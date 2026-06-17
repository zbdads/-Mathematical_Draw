"""Grounded answer synthesis for planned query flows."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.agent.evidence_scorer import ScoredEvidence
from src.config import CFG
from src.evaluation.citation_check import check_citation
from src.generator.answer_formatter import (
    FormattedAnswer,
    build_reject_answer,
    format_answer,
)
from src.generator.answer_prompt_selector import render_system_prompt
from src.generator.llm_client import generate, generate_stream


class AnswerSynthesizer:
    """基于精选证据生成 grounded 回答，附带 citation 校验。"""

    def build_messages(
        self,
        question: str,
        evidence: list[ScoredEvidence],
        question_type: str | None = None,
        answer_language: str = "English",
    ) -> list[dict[str, str]]:
        return _build_answer_messages(
            question=question,
            evidence=evidence,
            question_type=question_type,
            answer_language=answer_language,
        )

    def answer(
        self,
        question: str,
        evidence: list[ScoredEvidence],
        generate_answer: bool,
        question_type: str | None = None,
        stream_callback: Callable[[str], None] | None = None,
        answer_language: str = "English",
        debug_callback: Callable[[dict], None] | None = None,
    ) -> dict[str, Any]:
        evidence_dicts = [se.to_evidence_dict() for se in evidence]
        messages = self.build_messages(
            question=question,
            evidence=evidence,
            question_type=question_type,
            answer_language=answer_language,
        )
        return self.answer_from_dict_evidence(
            question=question,
            evidence_dicts=evidence_dicts,
            generate_answer=generate_answer,
            messages=messages,
            stream_callback=stream_callback,
            debug_callback=debug_callback,
        )

    def answer_from_dict_evidence(
        self,
        question: str,
        evidence_dicts: list[dict[str, Any]],
        generate_answer: bool,
        messages: list[dict[str, str]],
        stream_callback: Callable[[str], None] | None = None,
        debug_callback: Callable[[dict], None] | None = None,
    ) -> dict[str, Any]:
        """基于已准备好的 messages 和 evidence dict 执行回答生成。"""
        if not generate_answer:
            return {"results": evidence_dicts, "answer": None, "guardrail_reason": ""}

        if stream_callback is not None:
            raw_answer = generate_stream(
                messages=messages,
                model=CFG.generator.model_name,
                temperature=CFG.generator.temperature,
                max_tokens=CFG.generator.max_new_tokens,
                on_token=stream_callback,
            )
        else:
            raw_answer = generate(
                messages=messages,
                model=CFG.generator.model_name,
                temperature=CFG.generator.temperature,
                max_tokens=CFG.generator.max_new_tokens,
            )

        if debug_callback is not None:
            debug_callback({"call_id": "answer_prompt", "prompt": messages, "raw_output": raw_answer})

        formatted = format_answer(raw_answer, evidence=evidence_dicts, question=question)

        if formatted.is_parse_failed:
            reason = "; ".join(formatted.parse_errors)
            return {
                "results": evidence_dicts,
                "answer": formatted,
                "guardrail_reason": reason,
            }

        citation_report = check_citation(formatted.citations, formatted.parse_errors, evidence_dicts)
        if not citation_report["ok"]:
            reason = "; ".join(citation_report["errors"])
            return {
                "results": evidence_dicts,
                "answer": build_reject_answer(reason=reason, question=question),
                "guardrail_reason": reason,
            }

        return {"results": evidence_dicts, "answer": formatted, "guardrail_reason": ""}


def _build_answer_messages(
    question: str,
    evidence: list[ScoredEvidence],
    question_type: str | None = None,
    answer_language: str = "English",
) -> list[dict[str, str]]:
    evidence_blocks: list[str] = []
    included_chunk_ids: list[str] = []
    sub_query_to_chunks: dict[str, list[str]] = {}

    for se in evidence:
        raw_page = se.page
        page_num = raw_page[0] if isinstance(raw_page, list) and raw_page else raw_page
        if page_num in ([], "", None):
            page_num = "N/A"
        content = se.content.strip()
        score_str = (
            f"{se.llm_relevance_score}/10"
            if se.llm_relevance_score >= 0
            else f"rerank={se.final_score:.4f}"
        )
        block = (
            f"[{se.chunk_id}] ({se.paper_id}, page {page_num}, "
            f"score={score_str})\n{content}"
        )
        evidence_blocks.append(block)
        included_chunk_ids.append(se.chunk_id)
        for sq in se.matched_sub_queries or [se.source_query]:
            sub_query_to_chunks.setdefault(sq, []).append(se.chunk_id)

    valid_keys = ", ".join(f"[{cid}]" for cid in included_chunk_ids)
    system_content = render_system_prompt(
        question_type=question_type,
        valid_keys=valid_keys,
        answer_language=answer_language,
    )

    evidence_section = (
        "\n\n".join(evidence_blocks) if evidence_blocks else "(no relevant evidence found)"
    )
    if sub_query_to_chunks:
        mapping_lines = ["Sub-query to retrieved chunk IDs:"]
        for sq, chunk_ids in sub_query_to_chunks.items():
            unique_chunk_ids = ", ".join(dict.fromkeys(f"[{cid}]" for cid in chunk_ids))
            mapping_lines.append(f"- {sq}: {unique_chunk_ids}")
        mapping_section = "\n".join(mapping_lines)
    else:
        mapping_section = "Sub-query to retrieved chunk IDs:\n(none)"
    user_content = (
        f"Original user question: {question}\n\n"
        f"{mapping_section}\n\n"
        f"Retrieved evidence (deduplicated by chunk, sorted by relevance):\n"
        f"{evidence_section}"
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
