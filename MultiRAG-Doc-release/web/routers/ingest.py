"""POST /api/ingest + GET /api/ingest/stream/{job_id}（SSE）"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from src.pipeline.ingest import run_ingest
from web.services.job_store import job_store
from web.services.query_service import get_embedder_if_loaded, invalidate_service, preload_service

router = APIRouter()
logger = logging.getLogger(__name__)

_PAPER_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@router.post("/ingest", status_code=202)
async def start_ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    paper_id: str,
    multimodal: bool = False,
    overwrite: bool = False,
    use_caption_model: bool = False,
) -> dict:
    """接收 PDF 文件，以后台任务执行 ingest，立即返回 job_id。"""
    if not _PAPER_ID_RE.match(paper_id):
        raise HTTPException(status_code=422, detail="paper_id 只允许字母、数字、下划线、连字符")

    # 将上传文件写入临时目录
    content = await file.read()
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(content)
    tmp.flush()
    tmp.close()
    tmp_path = Path(tmp.name)

    job_id = job_store.create()
    background_tasks.add_task(_run_ingest_task, job_id, tmp_path, paper_id, multimodal, overwrite, use_caption_model)
    return {"job_id": job_id}


async def _run_ingest_task(
    job_id: str,
    pdf_path: Path,
    paper_id: str,
    multimodal: bool,
    overwrite: bool,
    use_caption_model: bool = False,
) -> None:
    job_store.update_status(job_id, "running")
    loop = asyncio.get_running_loop()

    def push(event: dict) -> None:
        loop.call_soon_threadsafe(job_store.push_event, job_id, event)

    def on_progress(step: str, message: str) -> None:
        push({"type": "progress", "step": step, "message": message})

    try:
        embedder = get_embedder_if_loaded()
        result = await asyncio.to_thread(
            run_ingest,
            pdf_path,
            paper_id,
            multimodal,
            overwrite,
            use_caption_model=use_caption_model,
            embedder=embedder,
            on_progress=on_progress,
        )
        job = job_store.get(job_id)
        if job is not None:
            job["result"] = result
        job_store.update_status(job_id, "done")
        push({"type": "done", "result": result})
        await invalidate_service()
        asyncio.create_task(preload_service())
    except Exception as e:
        logger.exception("[ingest] job=%s 执行出错", job_id)
        job = job_store.get(job_id)
        if job is not None:
            job["error"] = str(e)
        job_store.update_status(job_id, "error")
        push({"type": "error", "message": str(e)})
    finally:
        try:
            pdf_path.unlink(missing_ok=True)
        except Exception:
            pass


@router.get("/ingest/stream/{job_id}")
async def ingest_stream(job_id: str) -> StreamingResponse:
    """SSE：推送 ingest 进度事件；支持重连后回放 events_log。"""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job 不存在或已被清理")

    async def event_stream():
        # 先回放历史事件
        for event in list(job["events_log"]):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        # 若任务已终态，不需要继续等待队列
        if job["status"] in ("done", "error"):
            return

        q: asyncio.Queue = job["queue"]
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=30.0)
            except asyncio.TimeoutError:
                yield "data: {\"type\": \"heartbeat\"}\n\n"
                if job["status"] in ("done", "error"):
                    break
                continue
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event["type"] in ("done", "error"):
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")
