"""Agent query mode wrapper."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.config import CFG
from src.query.service import QueryService


def run_agent_query(
    question: str,
    top_k: int | None = None,
    paper_id: str | None = None,
    generate_answer: bool = True,
    stream_callback: Callable[[str], None] | None = None,
    node_callback: Callable[[str, dict], None] | None = None,
    max_steps: int = 10,
    debug_callback: Callable[[dict], None] | None = None,
    debug_agent: bool = False,
    service: QueryService | None = None,
) -> dict[str, Any]:
    """运行 agentic RAG loop，返回结构化结果。"""
    try:
        from src.agent.langgraph_agent import LangGraphAgent
    except ImportError as exc:
        raise ImportError(
            "Agent 模式需要安装 langgraph / langchain-core。"
            "若只复现文本 RAG，请先使用 standard 或 decompose 模式。"
        ) from exc

    owns_service = service is None
    service = service or QueryService.from_disk()
    try:
        agent = LangGraphAgent(
            service=service,
            max_steps=max_steps,
        )
        result = agent.run(
            question=question,
            paper_id=paper_id,
            generate_answer=generate_answer,
            stream_callback=stream_callback,
            node_callback=node_callback,
            debug_callback=debug_callback,
            answer_language=CFG.generator.answer_language,
        )
    finally:
        if owns_service:
            service.close()

    if not debug_agent:
        result["agent_trace"] = []
    return result
