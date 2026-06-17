"""Extract LLMOPT-style model cards from existing paper chunks."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from src.config import CFG
from src.generator.llm_client import generate
from src.modeling.prompts import (
    MODEL_CARD_SYSTEM_PROMPT,
    MODEL_FIELD_SYSTEM_PROMPT,
    build_model_card_user_prompt,
    build_model_field_user_prompt,
)
from src.modeling.schema import FIVE_ELEMENT_FIELDS, ModelCard, ModelEvidenceRef

_MODEL_KEYWORDS = (
    "model",
    "formulation",
    "objective",
    "constraint",
    "constraints",
    "decision variable",
    "variable",
    "parameter",
    "sets",
    "minimize",
    "maximize",
    "min ",
    "max ",
    "s.t.",
    "subject to",
    "mixed integer",
    "integer programming",
    "linear programming",
    "routing",
    "scheduling",
    "q-learning",
    "algorithm",
)

_FIELD_KEYWORDS = {
    "sets": (
        "sets",
        "set of",
        "index",
        "indices",
        "nodes",
        "patients",
        "jobs",
        "machines",
        "caregivers",
        "vehicles",
        "stages",
        "arcs",
        "periods",
        "customers",
        "centers",
        "locations",
        "\u2208",
        "\u2200",
    ),
    "parameters": (
        "parameters",
        "parameter",
        "constant",
        "coefficient",
        "cost",
        "time",
        "capacity",
        "demand",
        "due",
        "weight",
        "service",
        "travel",
        "processing",
        "big number",
        "big-m",
        "given",
    ),
    "variables": (
        "decision variables",
        "decision variable",
        "binary variable",
        "binary decision",
        "integer variable",
        "continuous variable",
        "variable",
        "otherwise",
        "{0,1}",
        "0-1",
    ),
    "objective": (
        "objective",
        "objective function",
        "minimize",
        "minimizing",
        "minimization",
        "maximize",
        "maximizing",
        "maximization",
        "min ",
        "max ",
        "operation cost",
        "tardiness",
        "delivery efficiency",
    ),
    "constraints": (
        "constraints",
        "constraint",
        "subject to",
        "s.t.",
        "eq.",
        "equation",
        "ensure",
        "stipulate",
        "flow conservation",
        "capacity",
        "time window",
        "workload balance",
        "skill requirements",
        "\u2264",
        "\u2265",
        "\u2211",
        "\u2200",
    ),
}

_FIELD_MAX_TOKENS = {
    "sets": 900,
    "parameters": 1500,
    "variables": 1500,
    "objective": 1200,
    "constraints": 2200,
}


def _chunk_id(chunk: dict[str, Any]) -> str:
    return str(chunk.get("id") or chunk.get("chunk_id") or "").strip()


def _chunk_content(chunk: dict[str, Any]) -> str:
    if chunk.get("modality", "text") == "figure":
        return str(chunk.get("caption_merged") or chunk.get("caption") or chunk.get("content") or "")
    return str(chunk.get("content") or chunk.get("caption_merged") or "")


def _compact_chunk(chunk: dict[str, Any], max_chars: int) -> dict[str, Any]:
    copy = dict(chunk)
    content = _chunk_content(copy)[:max_chars]
    copy["content"] = content
    if copy.get("modality") == "figure":
        copy["caption_merged"] = content
    return copy


def _has_math_signal(text: str) -> bool:
    lower = text.lower()
    return bool(
        re.search(r"\b(eq\.|equation|subject to)\b|\(\d+\)|<=|>=|=", lower)
        or any(op in text for op in ("\u2208", "\u2264", "\u2265", "\u2211", "\u2200"))
    )


def select_modeling_chunks(
    chunks: list[dict[str, Any]],
    *,
    max_chunks: int = 18,
    max_chars_per_chunk: int = 1200,
) -> list[dict[str, Any]]:
    """Select likely modeling-relevant chunks from a paper."""
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for pos, chunk in enumerate(chunks):
        modality = chunk.get("modality", "text")
        content = _chunk_content(chunk)
        lower = str(content).lower()
        score = 0
        for keyword in _MODEL_KEYWORDS:
            if keyword in lower:
                score += 2 if keyword in {"objective", "constraint", "decision variable", "s.t."} else 1
        section = str(chunk.get("section", "")).lower()
        if any(key in section for key in ("model", "formulation", "method", "algorithm")):
            score += 2
        if modality in {"equation", "table", "figure"}:
            score += 1
        if score > 0:
            scored.append((score, -pos, chunk))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [chunk for _, _, chunk in scored[:max_chunks]]
    if not selected:
        selected = chunks[: min(max_chunks, len(chunks))]

    compacted: list[dict[str, Any]] = []
    for chunk in selected:
        compacted.append(_compact_chunk(chunk, max_chars_per_chunk))
    return compacted


def select_modeling_field_chunks(
    chunks: list[dict[str, Any]],
    field: str,
    *,
    max_chunks: int = 12,
    max_chars_per_chunk: int = 1600,
) -> list[dict[str, Any]]:
    """Select chunks likely to contain one five-element modeling field."""
    keywords = _FIELD_KEYWORDS.get(field, ())
    scored: list[tuple[int, int, int, dict[str, Any]]] = []
    for pos, chunk in enumerate(chunks):
        content = _chunk_content(chunk)
        lower = content.lower()
        score = 0
        for keyword in keywords:
            if keyword.lower() in lower:
                score += 3
        for keyword in _MODEL_KEYWORDS:
            if keyword in lower:
                score += 1
        section = str(chunk.get("section", "")).lower()
        if any(key in section for key in ("model", "formulation", "method", "algorithm")):
            score += 2
        if _has_math_signal(content):
            score += 2
        if chunk.get("modality") in {"equation", "table", "figure"}:
            score += 1
        if field == "objective" and re.search(r"\b(min|max)(imize|imization|imisation)?\b", lower):
            score += 4
        if field == "constraints" and re.search(r"\b(eq\.|constraints?|subject to|stipulate|ensure)\b|\(\d+\)", lower):
            score += 4
        if field == "variables" and re.search(r"\b(if|whether)\b.+\b(otherwise|0)\b", lower):
            score += 4
        if score > 0:
            scored.append((score, -pos, pos, chunk))

    if not scored:
        return select_modeling_chunks(
            chunks,
            max_chunks=max_chunks,
            max_chars_per_chunk=max_chars_per_chunk,
        )

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected_positions: list[int] = []
    seen_positions: set[int] = set()

    def add_position(pos: int) -> None:
        if pos < 0 or pos >= len(chunks) or pos in seen_positions:
            return
        seen_positions.add(pos)
        selected_positions.append(pos)

    for score, _, pos, chunk in scored:
        if len(selected_positions) >= max_chunks:
            break
        if score >= 8 and chunk.get("modality") != "figure":
            add_position(pos - 1)
        add_position(pos)
        if score >= 8 and chunk.get("modality") != "figure":
            add_position(pos + 1)

    for _, _, pos, _ in scored:
        if len(selected_positions) >= max_chunks:
            break
        add_position(pos)

    return [
        _compact_chunk(chunks[pos], max_chars_per_chunk)
        for pos in sorted(selected_positions[:max_chunks])
    ]


def build_evidence_blocks(chunks: list[dict[str, Any]]) -> list[str]:
    """Render source chunks for model-card extraction."""
    blocks: list[str] = []
    for chunk in chunks:
        chunk_id = _chunk_id(chunk) or "unknown"
        raw_page = chunk.get("page", -1)
        page = raw_page[0] if isinstance(raw_page, list) and raw_page else raw_page
        modality = chunk.get("modality", "text")
        content = _chunk_content(chunk).strip()
        if not content:
            continue
        blocks.append(
            f"[{chunk_id}] page={page}, modality={modality}\n{content}"
        )
    return blocks


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return [str(value).strip()]


def _dedupe_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = re.sub(r"\s+", " ", value).strip()
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _merge_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            merged.append(value)
    return merged


def _parse_confidence(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _generate_json_response(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    label: str,
) -> tuple[dict[str, Any] | None, str, str | None]:
    raw = ""
    last_error = ""
    for attempt in range(1, 4):
        try:
            raw = generate(
                messages=messages,
                model=CFG.generator.model_name,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return _extract_json_object(raw), raw, None
        except Exception as exc:
            last_error = str(exc)
            print(f"[model-card] {label} failed attempt {attempt}/3: {exc}")
            time.sleep(3 * attempt)
    return None, raw, last_error or "unknown_error"


def _sanitize_card_data(data: dict[str, Any], paper_id: str, source_chunk_ids: list[str]) -> ModelCard:
    refs: list[ModelEvidenceRef] = []
    valid_source_ids = set(source_chunk_ids)
    for ref in data.get("evidence_refs", []):
        if not isinstance(ref, dict):
            continue
        chunk_id = str(ref.get("chunk_id", "")).strip()
        if chunk_id and chunk_id not in valid_source_ids:
            continue
        refs.append(
            ModelEvidenceRef(
                chunk_id=chunk_id,
                page=-1,
                modality="text",
                role=str(ref.get("role", "other")).strip() or "other",
                quote=str(ref.get("quote", "")).strip()[:500],
            )
        )

    return ModelCard(
        paper_id=paper_id,
        title=str(data.get("title", "")).strip(),
        problem_type=str(data.get("problem_type", "")).strip(),
        application_domain=str(data.get("application_domain", "")).strip(),
        model_name=str(data.get("model_name", "")).strip(),
        sets=_dedupe_texts(_as_list(data.get("sets"))),
        parameters=_dedupe_texts(_as_list(data.get("parameters"))),
        variables=_dedupe_texts(_as_list(data.get("variables"))),
        objective=_dedupe_texts(_as_list(data.get("objective"))),
        constraints=_dedupe_texts(_as_list(data.get("constraints"))),
        assumptions=_dedupe_texts(_as_list(data.get("assumptions"))),
        algorithm=_dedupe_texts(_as_list(data.get("algorithm"))),
        source_chunk_ids=source_chunk_ids,
        evidence_refs=refs,
        confidence=_parse_confidence(data.get("confidence", 0.0)),
        warnings=_as_list(data.get("warnings")),
    )


def _sanitize_field_data(
    data: dict[str, Any],
    *,
    field: str,
    source_chunk_ids: list[str],
) -> dict[str, Any]:
    valid_source_ids = set(source_chunk_ids)
    items: list[str] = []
    refs: list[ModelEvidenceRef] = []

    raw_items = data.get("items", [])
    if isinstance(raw_items, str):
        raw_items = [raw_items]
    if not isinstance(raw_items, list):
        raw_items = []

    for raw_item in raw_items:
        if isinstance(raw_item, dict):
            text = str(raw_item.get("text", "")).strip()
            chunk_id = str(raw_item.get("chunk_id", "")).strip()
            quote = str(raw_item.get("quote", "")).strip()[:500]
        else:
            text = str(raw_item).strip()
            chunk_id = ""
            quote = ""
        if not text:
            continue
        items.append(text)
        if chunk_id in valid_source_ids:
            refs.append(
                ModelEvidenceRef(
                    chunk_id=chunk_id,
                    page=-1,
                    modality="text",
                    role=field,
                    quote=quote,
                )
            )

    if not items:
        items.extend(_as_list(data.get(field)))

    return {
        "items": _dedupe_texts(items),
        "refs": refs,
        "confidence": _parse_confidence(data.get("confidence", 0.0)),
        "warnings": _as_list(data.get("warnings")),
    }


def extract_model_field_from_chunks(
    paper_id: str,
    chunks: list[dict[str, Any]],
    field: str,
    *,
    max_chunks: int = 12,
) -> dict[str, Any]:
    """Extract one five-element modeling field with focused evidence."""
    selected = select_modeling_field_chunks(
        chunks,
        field,
        max_chunks=max_chunks,
    )
    source_chunk_ids = [
        _chunk_id(chunk)
        for chunk in selected
        if _chunk_id(chunk)
    ]
    blocks = build_evidence_blocks(selected)
    messages = [
        {"role": "system", "content": MODEL_FIELD_SYSTEM_PROMPT},
        {"role": "user", "content": build_model_field_user_prompt(paper_id, field, blocks)},
    ]
    data, raw, error = _generate_json_response(
        messages,
        max_tokens=min(CFG.generator.max_new_tokens, _FIELD_MAX_TOKENS.get(field, 1400)),
        label=f"{field} extraction",
    )
    if error or data is None:
        return {
            "items": [],
            "refs": [],
            "source_chunk_ids": source_chunk_ids,
            "selected_chunks": selected,
            "confidence": 0.0,
            "warnings": [f"{field}_field_failed: {error}"],
        }

    result = _sanitize_field_data(
        data,
        field=field,
        source_chunk_ids=source_chunk_ids,
    )
    result["source_chunk_ids"] = source_chunk_ids
    result["selected_chunks"] = selected
    return result


def enrich_evidence_refs(card: ModelCard, chunk_by_id: dict[str, dict[str, Any]]) -> ModelCard:
    """Fill page/modality for evidence refs from source chunks."""
    enriched: list[ModelEvidenceRef] = []
    for ref in card.evidence_refs:
        chunk = chunk_by_id.get(ref.chunk_id, {})
        raw_page = chunk.get("page", ref.page)
        page = raw_page if raw_page not in (None, "") else -1
        enriched.append(
            ModelEvidenceRef(
                chunk_id=ref.chunk_id,
                page=page,
                modality=chunk.get("modality", ref.modality),
                role=ref.role,
                quote=ref.quote,
            )
        )
    card.evidence_refs = enriched
    return card


def extract_model_card_from_chunks(
    paper_id: str,
    chunks: list[dict[str, Any]],
    *,
    max_chunks: int = 18,
    field_level: bool = True,
) -> ModelCard:
    """Extract a structured model card for one paper via the configured LLM."""
    selected = select_modeling_chunks(chunks, max_chunks=max_chunks)
    source_chunk_ids = [
        _chunk_id(chunk)
        for chunk in selected
        if _chunk_id(chunk)
    ]
    blocks = build_evidence_blocks(selected)
    messages = [
        {"role": "system", "content": MODEL_CARD_SYSTEM_PROMPT},
        {"role": "user", "content": build_model_card_user_prompt(paper_id, blocks)},
    ]
    data, raw, error = _generate_json_response(
        messages,
        max_tokens=min(CFG.generator.max_new_tokens, 2200),
        label="overview extraction",
    )
    if error or data is None:
        card = ModelCard(
            paper_id=paper_id,
            source_chunk_ids=source_chunk_ids,
            confidence=0.0,
            warnings=[f"model_card_llm_failed: {error}", raw[:500] if raw else ""],
        )
        return card
    try:
        card = _sanitize_card_data(data, paper_id, source_chunk_ids)
    except Exception as exc:
        card = ModelCard(
            paper_id=paper_id,
            source_chunk_ids=source_chunk_ids,
            confidence=0.0,
            warnings=[f"model_card_json_parse_failed: {exc}", raw[:500]],
        )
    chunk_by_id = {
        _chunk_id(chunk): chunk
        for chunk in selected
    }
    if not field_level:
        return enrich_evidence_refs(card, chunk_by_id)

    field_confidences: list[float] = []
    for field in FIVE_ELEMENT_FIELDS:
        field_result = extract_model_field_from_chunks(
            paper_id,
            chunks,
            field,
            max_chunks=max(8, min(max_chunks, 14)),
        )
        selected_field_chunks = field_result.get("selected_chunks", [])
        for chunk in selected_field_chunks:
            cid = _chunk_id(chunk)
            if cid:
                chunk_by_id[cid] = chunk
        card.source_chunk_ids = _merge_unique(
            card.source_chunk_ids + field_result.get("source_chunk_ids", [])
        )

        items = field_result.get("items", [])
        if items:
            setattr(card, field, items)
        refs = field_result.get("refs", [])
        if refs:
            card.evidence_refs = [
                ref for ref in card.evidence_refs if ref.role != field
            ] + refs
        confidence = float(field_result.get("confidence", 0.0) or 0.0)
        if confidence > 0:
            field_confidences.append(confidence)
        for warning in field_result.get("warnings", []):
            if warning:
                card.warnings.append(f"{field}: {warning}")

    card.warnings = _dedupe_texts(card.warnings)
    coverage = sum(1 for field in FIVE_ELEMENT_FIELDS if getattr(card, field)) / len(FIVE_ELEMENT_FIELDS)
    if field_confidences:
        field_confidence = sum(field_confidences) / len(field_confidences)
        card.confidence = max(
            0.0,
            min(1.0, 0.45 * card.confidence + 0.45 * field_confidence + 0.10 * coverage),
        )
    return enrich_evidence_refs(card, chunk_by_id)
