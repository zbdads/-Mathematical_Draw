"""答案格式化模块。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

INSUFFICIENT_ANSWER = "I cannot answer based on the available evidence."


@dataclass
class ParsedCitation:
    """解析后的 citation。"""

    paper_id: str
    page: int
    chunk_id: str


@dataclass
class FormattedAnswer:
    """结构化答案。"""

    question: str
    answer: str
    raw_output: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    citation_map: dict[str, dict[str, Any]] = field(default_factory=dict)
    confidence: float = 0.0
    parse_errors: list[str] = field(default_factory=list)

    @property
    def is_insufficient(self) -> bool:
        return self.answer.strip() == INSUFFICIENT_ANSWER

    @property
    def is_parse_failed(self) -> bool:
        return any("format" in e.lower() for e in self.parse_errors)


def build_reject_answer(reason: str, question: str = "") -> FormattedAnswer:
    """构造统一的拒答结果。"""
    return FormattedAnswer(
        question=question,
        answer=INSUFFICIENT_ANSWER,
        citations=[],
        parse_errors=[reason] if reason else [],
    )


_ANSWER_CITATION_PATTERN = re.compile(
    r"Answer:\s*(.*?)\n\s*Citations:\s*(.*)$",
    flags=re.IGNORECASE | re.DOTALL,
)


def _build_evidence_lookup(evidence: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for chunk in evidence:
        cid = str(chunk.get("chunk_id", ""))
        if cid:
            lookup[cid] = chunk
    return lookup


def _parse_citations(
    citation_text: str,
    evidence_lookup: dict[str, dict[str, Any]],
) -> tuple[list[ParsedCitation], list[str]]:
    citations: list[ParsedCitation] = []
    errors: list[str] = []

    if citation_text.strip() in {"", "[]"}:
        return citations, errors

    chunk_ids = re.findall(r"\[([^\[\]]+)\]", citation_text)
    if not chunk_ids:
        errors.append("Citations field has no bracketed citation blocks")
        return citations, errors

    for cid in chunk_ids:
        cid = cid.strip()
        info = evidence_lookup.get(cid)
        if info:
            raw_page = info.get("page", 0)
            page = raw_page[0] if isinstance(raw_page, list) and raw_page else raw_page
            if page in ([], "", None):
                page = -1
            citations.append(
                ParsedCitation(
                    paper_id=str(info.get("paper_id", "")),
                    page=int(page),
                    chunk_id=cid,
                )
            )
        else:
            errors.append(f"Unknown citation key: [{cid}]")

    return citations, errors


def _normalize_citations(
    parsed_citations: list[ParsedCitation],
    evidence_lookup: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    citations: list[dict[str, Any]] = []
    citation_map: dict[str, dict[str, Any]] = {}
    chunk_to_index: dict[str, int] = {}

    for citation in parsed_citations:
        existing_index = chunk_to_index.get(citation.chunk_id)
        if existing_index is not None:
            continue

        index = len(citations) + 1
        chunk_to_index[citation.chunk_id] = index
        evidence = evidence_lookup.get(citation.chunk_id, {})
        item = {
            "index": index,
            "paper_id": citation.paper_id,
            "page": citation.page,
            "chunk_id": citation.chunk_id,
            "modality": evidence.get("modality", "text"),
            "content": evidence.get("content", ""),
            "caption": evidence.get("caption", ""),
            "section": evidence.get("section", ""),
            "score": evidence.get("score"),
            "rerank_score": evidence.get("rerank_score"),
            "figure_id": evidence.get("figure_id"),
            "image_path": evidence.get("image_path"),
        }
        citations.append(item)
        citation_map[str(index)] = {
            "index": index,
            "chunk_id": citation.chunk_id,
            "paper_id": citation.paper_id,
            "page": citation.page,
        }

    return citations, citation_map


def _replace_inline_citations(answer_text: str, citations: list[dict[str, Any]]) -> str:
    if not answer_text or not citations:
        return answer_text

    rendered = answer_text
    for citation in citations:
        chunk_id = str(citation.get("chunk_id", "")).strip()
        index = citation.get("index")
        if not chunk_id or not index:
            continue
        rendered = re.sub(
            rf"\[{re.escape(chunk_id)}\]",
            f"[{index}]",
            rendered,
        )
    return rendered


def format_answer(
    raw: str,
    evidence: list[dict[str, Any]],
    question: str = "",
) -> FormattedAnswer:
    """格式化 LLM 输出为结构化答案。"""
    evidence_lookup = _build_evidence_lookup(evidence)

    text = (raw or "").strip()
    if not text:
        return FormattedAnswer(
            question=question,
            answer=INSUFFICIENT_ANSWER,
            raw_output="",
            citations=[],
            parse_errors=["Empty model response"],
        )

    if text == INSUFFICIENT_ANSWER:
        return FormattedAnswer(
            question=question, answer=text, raw_output=text, citations=[],
        )

    match = _ANSWER_CITATION_PATTERN.search(text)
    if not match:
        return FormattedAnswer(
            question=question,
            answer=text,
            raw_output=text,
            citations=[],
            parse_errors=["Model output does not follow 'Answer/Citations' format"],
        )

    answer = match.group(1).strip()
    citation_text = match.group(2).strip()
    parsed_citations, parse_errors = _parse_citations(citation_text, evidence_lookup)
    citations, citation_map = _normalize_citations(parsed_citations, evidence_lookup)
    answer = _replace_inline_citations(answer, citations)

    return FormattedAnswer(
        question=question,
        answer=answer if answer else INSUFFICIENT_ANSWER,
        raw_output=text,
        citations=citations,
        citation_map=citation_map,
        parse_errors=parse_errors,
    )


def render_answer(formatted: FormattedAnswer) -> str:
    """将结构化答案渲染为 CLI 文本。"""
    if formatted.citations:
        citations = ", ".join(
            f"[{c.get('index', i + 1)}]"
            for i, c in enumerate(formatted.citations)
        )
    else:
        citations = "[]"
    return f"Answer: {formatted.answer}\nCitations: {citations}"
