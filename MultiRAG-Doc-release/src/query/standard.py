"""Standard query mode."""

from __future__ import annotations

from collections.abc import Callable

from src.config import CFG
from src.evaluation.guardrails import pre_generation_guardrail
from src.generator.answer_formatter import build_reject_answer
from src.generator.prompt_builder import build_messages
from src.query.answer_synthesizer import AnswerSynthesizer
from src.query.service import QueryService, flatten_dual_results


def run_standard_query(
    question: str,
    top_k: int,
    paper_id: str | None,
    generate_answer: bool,
    stream_callback: Callable[[str], None] | None = None,
    on_results_ready: Callable[[list[dict]], None] | None = None,
    service: QueryService | None = None,
) -> dict:
    """执行标准 query 流程。"""
    owns_service = service is None
    service = service or QueryService.from_disk()
    try:
        retrieval = service.retrieve_core(question, top_k, paper_id)
        text_results = retrieval["text_results"]
        figure_results = retrieval["figure_results"]
        results = flatten_dual_results(retrieval)

        if on_results_ready is not None:
            on_results_ready(results)

        if not generate_answer:
            return {
                "results": results,
                "text_results": text_results,
                "figure_results": figure_results,
                "answer": None,
                "guardrail_reason": "",
            }

        ok, reason = pre_generation_guardrail(question, results)
        if not ok:
            return {
                "results": results,
                "text_results": text_results,
                "figure_results": figure_results,
                "answer": build_reject_answer(reason),
                "guardrail_reason": reason,
            }

        messages = build_messages(
            question=question,
            evidence=results,
            answer_language=CFG.generator.answer_language,
        )
        outcome = AnswerSynthesizer().answer_from_dict_evidence(
            question=question,
            evidence_dicts=results,
            generate_answer=generate_answer,
            messages=messages,
            stream_callback=stream_callback,
        )
        outcome["text_results"] = text_results
        outcome["figure_results"] = figure_results
        return outcome
    finally:
        if owns_service:
            service.close()
