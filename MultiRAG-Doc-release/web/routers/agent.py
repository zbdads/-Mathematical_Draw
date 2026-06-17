"""POST /api/agent/stream (SSE)

事件格式与 query.py 保持一致：data: {JSON}\n\n，用 type 字段区分事件类型。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.query.agent_mode import run_agent_query
from web.services.query_service import get_service
from web.utils import serialize_answer

router = APIRouter()
logger = logging.getLogger(__name__)


class AgentStreamRequest(BaseModel):
    question: str
    paper_id: str | None = None
    generate_answer: bool = True
    max_steps: int = 10
    debug: bool = False


@router.post("/agent/stream")
async def agent_stream(req: AgentStreamRequest) -> StreamingResponse:
    """SSE 流式 Agent 查询：推送节点事件 + 逐 token 答案。"""
    queue: asyncio.Queue[dict] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def node_callback(event_type: str, data: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, {"type": event_type, **data})

    def on_token(delta: str) -> None:
        if delta:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "token", "delta": delta})

    def debug_callback(data: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "debug_llm", **data})

    async def run_agent() -> None:
        try:
            service, _ = await get_service()
            result: dict[str, Any] = await asyncio.to_thread(
                run_agent_query,
                req.question,
                paper_id=req.paper_id or None,
                generate_answer=req.generate_answer,
                max_steps=req.max_steps,
                stream_callback=on_token if req.generate_answer else None,
                node_callback=node_callback,
                debug_callback=debug_callback if req.debug else None,
                service=service,
            )
            loop.call_soon_threadsafe(queue.put_nowait, {
                "type": "done",
                "answer": serialize_answer(result.get("answer")),
                "guardrail_reason": result.get("guardrail_reason", ""),
                "terminate_reason": result.get("terminate_reason", ""),
                "selected_evidence_count": result.get("selected_evidence_count", 0),
            })
        except Exception as e:
            logger.exception("[agent/stream] 执行出错")
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})

    asyncio.create_task(run_agent())

    async def event_stream():
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event["type"] in ("done", "error"):
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")
