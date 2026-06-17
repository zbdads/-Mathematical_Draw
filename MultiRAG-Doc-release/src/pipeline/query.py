"""Unified query pipeline entrypoint."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.query.agent_mode import run_agent_query
from src.query.decompose import run_decompose_query
from src.query.service import QueryService
from src.query.standard import run_standard_query


def run_query(
    question: str,
    top_k: int | None,
    paper_id: str | None,
    generate_answer: bool,
    mode: str = "standard",
    stream_callback: Callable[[str], None] | None = None,
    on_results_ready: Callable[[list[dict]], None] | None = None,
    debug: bool = False,
    max_steps: int = 10,
) -> dict[str, Any]:
    """统一 query 入口，根据 mode 分发到 query 应用层实现。"""
    if mode == "standard":
        if top_k is None:
            raise ValueError("standard mode requires top_k")
        return run_standard_query(
            question=question,
            top_k=top_k,
            paper_id=paper_id,
            generate_answer=generate_answer,
            stream_callback=stream_callback,
            on_results_ready=on_results_ready,
        )

    if mode == "decompose":
        return run_decompose_query(
            question=question,
            top_k=top_k,
            paper_id=paper_id,
            generate_answer=generate_answer,
            debug_decompose=debug,
            stream_callback=stream_callback,
        )

    if mode == "agent":
        return run_agent_query(
            question=question,
            top_k=top_k,
            paper_id=paper_id,
            generate_answer=generate_answer,
            stream_callback=stream_callback,
            max_steps=max_steps,
            debug_agent=debug,
        )

    raise ValueError(f"Unsupported query mode: {mode}")

