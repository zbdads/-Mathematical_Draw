"""POST /api/query（非流式）+ POST /api/query/stream（SSE）
+ POST /api/query/decompose（非流式）+ POST /api/query/decompose/stream（SSE）
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.config import CFG
from src.query.decompose import run_decompose_query
from src.query.standard import run_standard_query
from web.services.query_service import get_service
from web.utils import image_path_to_url, serialize_answer, serialize_result

router = APIRouter()
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    paper_id: str | None = None
    generate_answer: bool = False


def _enrich_figures(figure_results: list[dict]) -> list[dict]:
    """将 figure 结果中的 image_path 转换为 image_url，过滤不合法条目。"""
    out = []
    for r in figure_results:
        url = image_path_to_url(r.get("image_path"))
        if url is None:
            logger.warning("[query] 过滤非法 image_path: %s", r.get("image_path"))
            continue
        out.append({**r, "image_url": url})
    return out


@router.post("/query")
async def query(req: QueryRequest) -> dict:
    service, _ = await get_service()
    top_k = req.top_k if req.top_k is not None else CFG.retriever.top_k
    result: dict[str, Any] = await asyncio.to_thread(
        run_standard_query,
        req.question,
        top_k,
        req.paper_id,
        req.generate_answer,
        None,
        None,
        service,
    )
    return {
        "text_results": [serialize_result(r) for r in result["text_results"]],
        "figure_results": _enrich_figures([serialize_result(r) for r in result["figure_results"]]),
        "answer": serialize_answer(result["answer"]),
        "guardrail_reason": result["guardrail_reason"],
    }


@router.post("/query/stream")
async def query_stream(req: QueryRequest) -> StreamingResponse:
    """SSE 流式查询：先推送检索结果，再逐 token 推送答案。"""
    queue: asyncio.Queue[dict] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_results_ready(results: list[dict]) -> None:
        text_r = [serialize_result(r) for r in results if r.get("modality") != "figure"]
        fig_r = _enrich_figures([serialize_result(r) for r in results if r.get("modality") == "figure"])
        loop.call_soon_threadsafe(
            queue.put_nowait,
            {"type": "retrieval_done", "text_results": text_r, "figure_results": fig_r},
        )

    def on_token(delta: str) -> None:
        if not isinstance(delta, str) or not delta:
            return
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "token", "delta": delta})

    async def run_query() -> None:
        try:
            service, _ = await get_service()
            top_k = req.top_k if req.top_k is not None else CFG.retriever.top_k
            result = await asyncio.to_thread(
                run_standard_query,
                req.question,
                top_k,
                req.paper_id,
                req.generate_answer,
                on_token if req.generate_answer else None,
                on_results_ready,
                service,
            )
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {
                    "type": "done",
                    "answer": serialize_answer(result["answer"]),
                    "guardrail_reason": result["guardrail_reason"],
                },
            )
        except Exception as e:
            logger.exception("[query/stream] 执行出错")
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})

    asyncio.create_task(run_query())

    async def event_stream():
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event["type"] in ("done", "error"):
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Decompose 路由 ────────────────────────────────────────────────────────────

class DecomposeQueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    paper_id: str | None = None
    generate_answer: bool = False


def _enrich_decompose_figures(fig_dicts: list[dict]) -> list[dict]:
    """为 decompose 图像结果附加 image_url，过滤不合法路径。"""
    out = []
    for r in fig_dicts:
        url = image_path_to_url(r.get("image_path"))
        if url is None:
            logger.warning("[decompose] 过滤非法 image_path: %s", r.get("image_path"))
            continue
        out.append({**r, "image_url": url})
    return out


def _split_and_enrich(all_results: list[dict]) -> tuple[list[dict], list[dict]]:
    """将 results 列表按 modality 拆分为 text_results 和 figure_results。"""
    text_results = [serialize_result(r) for r in all_results if r.get("modality") != "figure"]
    figure_results = _enrich_decompose_figures(
        [serialize_result(r) for r in all_results if r.get("modality") == "figure"]
    )
    return text_results, figure_results


@router.post("/query/decompose")
async def decompose_query(req: DecomposeQueryRequest) -> dict:
    """非流式 decompose 查询：QueryPlanner → FAISS → BGE rerank → AnswerAgent。"""
    result: dict[str, Any] = await asyncio.to_thread(
        run_decompose_query,
        req.question,
        req.top_k,
        req.paper_id,
        req.generate_answer,
    )
    text_results, figure_results = _split_and_enrich(result.get("results", []))
    plan = result.get("plan")
    plan_dict: dict = {}
    if plan is not None:
        try:
            plan_dict = asdict(plan)
        except Exception:
            pass
    return {
        "mode": "decompose",
        "text_results": text_results,
        "figure_results": figure_results,
        "answer": serialize_answer(result.get("answer")),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "plan": plan_dict,
    }


@router.post("/query/decompose/stream")
async def decompose_stream(req: DecomposeQueryRequest) -> StreamingResponse:
    """SSE 流式 decompose 查询，事件格式与 /api/query/stream 对齐。"""
    queue: asyncio.Queue[dict] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def plan_callback(plan_dict: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "plan", "plan": plan_dict})

    def retrieval_done_callback(text_dicts: list[dict], fig_dicts: list[dict]) -> None:
        text_r = [serialize_result(r) for r in text_dicts]
        fig_r = _enrich_decompose_figures([serialize_result(r) for r in fig_dicts])
        loop.call_soon_threadsafe(
            queue.put_nowait,
            {"type": "retrieval_done", "text_results": text_r, "figure_results": fig_r},
        )

    def on_token(delta: str) -> None:
        if not isinstance(delta, str) or not delta:
            return
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "token", "delta": delta})

    async def run_decompose() -> None:
        try:
            result = await asyncio.to_thread(
                run_decompose_query,
                req.question,
                req.top_k,
                req.paper_id,
                req.generate_answer,
                False,  # debug_decompose
                on_token if req.generate_answer else None,
                plan_callback,
                retrieval_done_callback,
            )
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {
                    "type": "done",
                    "answer": serialize_answer(result.get("answer")),
                    "guardrail_reason": result.get("guardrail_reason", ""),
                },
            )
        except Exception as e:
            logger.exception("[decompose/stream] 执行出错")
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})

    asyncio.create_task(run_decompose())

    async def event_stream():
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event["type"] in ("done", "error"):
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")
