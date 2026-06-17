"""Citation 验证模块。"""

from __future__ import annotations

from typing import Any


def _normalize_page(raw_page: Any) -> int:
    if isinstance(raw_page, list):
        return raw_page[0] if raw_page else -1
    if isinstance(raw_page, int):
        return raw_page
    return -1


def _evidence_key_map(evidence: list[dict[str, Any]]) -> set[tuple[str, int, str]]:
    keys: set[tuple[str, int, str]] = set()
    for item in evidence:
        keys.add(
            (
                str(item.get("paper_id", "")),
                _normalize_page(item.get("page", -1)),
                str(item.get("chunk_id", "")),
            )
        )
    return keys


def check_citation(
    citations: list[dict[str, Any]],
    parse_errors: list[str],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """验证答案中的 citation 是否与证据一致。"""
    errors: list[str] = list(parse_errors)
    evidence_keys = _evidence_key_map(evidence)

    missing: list[dict[str, Any]] = []
    for citation in citations:
        key = (
            str(citation.get("paper_id", "")),
            _normalize_page(citation.get("page", -1)),
            str(citation.get("chunk_id", "")),
        )
        if key not in evidence_keys:
            missing.append(citation)

    if missing:
        errors.append("Citations contain items not present in retrieved evidence")

    if not citations and not parse_errors:
        errors.append("Non-refusal answer must contain at least one citation")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "missing": missing,
        "citation_count": len(citations),
    }
