"""Model-aware chunk extraction for optimization papers.

This module builds extra retrieval chunks for mathematical model regions:
model construction sections, notation/parameter tables, objective blocks,
constraint blocks, equation blocks, and model-related figures. These chunks
sit beside ordinary text chunks and make modeling queries less dependent on
generic sentence splitting.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from src.ingestion.chunk_id import model_region_chunk_id

_SECTION_KEYWORDS = (
    "problem statement",
    "problem description",
    "model construction",
    "mathematical model",
    "model formulation",
    "problem formulation",
    "formulation",
    "notation",
    "notations",
    "assumptions",
    "decision variables",
    "parameters",
    "constraints",
    "objective",
)

_NOTATION_KEYWORDS = (
    "notation",
    "notations",
    "parameter",
    "parameters",
    "decision variable",
    "decision variables",
    "sets",
    "indices",
    "symbol",
    "description",
    "table",
)

_OBJECTIVE_KEYWORDS = (
    "objective",
    "objective function",
    "minimize",
    "minimization",
    "minimise",
    "minimisation",
    "maximize",
    "maximization",
    "maximise",
    "maximisation",
    "total cost",
    "tardiness",
)

_CONSTRAINT_KEYWORDS = (
    "constraint",
    "constraints",
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
    "skill requirement",
)

_PROBLEM_KEYWORDS = (
    "problem statement",
    "problem description",
    "problem definition",
    "problem formulation",
    "we consider",
    "this study considers",
)

_ASSUMPTION_KEYWORDS = (
    "assumption",
    "assumptions",
    "assume",
    "without loss of generality",
)

_MATH_SYMBOLS = ("∑", "∀", "∈", "≤", "≥", "≠", "⋅", "∪")

_MODEL_ELEMENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "problem_definition": _PROBLEM_KEYWORDS,
    "sets": (
        "sets",
        "set of",
        "indices",
        "index set",
        "nodes",
        "patients",
        "caregivers",
        "jobs",
        "machines",
        "vehicles",
    ),
    "parameters": (
        "parameter",
        "parameters",
        "given",
        "travel time",
        "service time",
        "processing time",
        "capacity",
        "cost",
        "demand",
        "time window",
        "big-m",
    ),
    "variables": (
        "decision variable",
        "decision variables",
        "binary variable",
        "continuous variable",
        "integer variable",
        "otherwise",
        "{0,1}",
        "0-1",
    ),
    "objective": _OBJECTIVE_KEYWORDS,
    "constraints": _CONSTRAINT_KEYWORDS,
    "assumptions": _ASSUMPTION_KEYWORDS,
}

_OPERATOR_HINT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "assignment": (
        "assign",
        "assignment",
        "allocated",
        "allocation",
        "served by",
        "assigned to",
        "分配",
    ),
    "routing_flow": (
        "route",
        "routing",
        "arc",
        "travel",
        "depot",
        "flow conservation",
        "path",
        "路径",
        "路线",
    ),
    "time_window": (
        "time window",
        "earliest",
        "latest",
        "ready time",
        "due time",
        "时间窗",
    ),
    "time_propagation": (
        "arrival",
        "departure",
        "start time",
        "completion",
        "service duration",
        "precedence",
        "big-m",
        "到达",
        "离开",
    ),
    "waiting_time": (
        "waiting time",
        "wait",
        "delay",
        "tardiness",
        "等待",
        "延迟",
    ),
    "capacity": (
        "capacity",
        "workload",
        "working time",
        "route duration",
        "maximum duration",
        "容量",
        "工作量",
    ),
    "skill_matching": (
        "skill",
        "qualification",
        "competence",
        "requirement",
        "技能",
        "资质",
    ),
    "outsourcing": (
        "outsourcing",
        "outsource",
        "external service",
        "rejection",
        "unserved",
        "外包",
    ),
    "priority_class": (
        "vip",
        "priority",
        "ordinary patient",
        "patient class",
        "优先级",
    ),
    "synchronization": (
        "synchronized",
        "synchronization",
        "simultaneous",
        "paired visit",
        "同步",
    ),
    "break_scheduling": (
        "lunch",
        "break",
        "rest",
        "午休",
        "休息",
    ),
    "overtime": (
        "overtime",
        "extra working",
        "late work",
        "加班",
    ),
    "multi_center": (
        "multi-center",
        "multi center",
        "multiple centers",
        "healthcare center",
        "depot assignment",
        "多中心",
        "护理站",
    ),
    "preference_matching": (
        "preference",
        "gender preference",
        "patient preference",
        "caregiver preference",
        "偏好",
    ),
    "multi_objective": (
        "multi-objective",
        "weighted sum",
        "trade-off",
        "pareto",
        "多目标",
    ),
}

_DOMAIN_SIGNAL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "home_health_care": (
        "home health care",
        "home healthcare",
        "hhc",
        "caregiver",
        "caregivers",
        "patient",
        "patients",
        "nurse",
        "nurses",
        "home visit",
        "居家医疗",
        "家庭医疗",
        "上门护理",
        "护理员",
        "患者",
    ),
    "production_scheduling": (
        "job shop",
        "flow shop",
        "production scheduling",
        "machine",
        "machines",
        "job",
        "jobs",
        "operation",
        "operations",
        "车间调度",
        "生产调度",
        "机器",
        "工序",
    ),
    "routing_scheduling": (
        "routing",
        "scheduling",
        "vehicle routing",
        "route",
        "time window",
        "路径",
        "调度",
    ),
}

_HHC_SIGNAL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "patient": ("patient", "patients", "患者"),
    "caregiver": ("caregiver", "caregivers", "nurse", "nurses", "护理员", "护士"),
    "visit": ("visit", "visits", "home visit", "service request", "上门", "服务"),
    "route": ("route", "routing", "travel", "depot", "路径", "路线"),
    "time": (
        "arrival",
        "departure",
        "start time",
        "time window",
        "service duration",
        "waiting time",
        "到达",
        "时间窗",
        "等待",
    ),
    "skill": ("skill", "qualification", "requirement", "技能", "资质"),
    "outsourcing": ("outsourcing", "outsource", "external service", "外包"),
    "priority": ("vip", "priority", "ordinary patient", "优先级"),
    "workload": ("workload", "working time", "route duration", "工作量"),
    "synchronization": ("synchronized", "synchronization", "simultaneous", "同步"),
    "break": ("lunch", "break", "rest", "午休", "休息"),
    "overtime": ("overtime", "加班"),
    "center": ("healthcare center", "depot", "multi-center", "护理站"),
    "preference": ("preference", "偏好"),
}

_MODEL_ELEMENT_ORDER = (
    "problem_definition",
    "sets",
    "parameters",
    "variables",
    "objective",
    "constraints",
    "assumptions",
    "formula",
    "table",
    "figure",
)

_NOISY_SECTION_KEYWORDS = (
    "references",
    "bibliography",
    "funding",
    "credit author",
    "competing interest",
    "data availability",
    "acknowledg",
)

_ALGORITHM_NOISE_KEYWORDS = (
    "algorithm",
    "q-learning",
    "q-table",
    "local search",
    "population",
    "mutation",
    "crossover",
    "fitness",
    "reward",
    "roulette",
    "pseudocode",
)

_EXPERIMENT_NOISE_KEYWORDS = (
    "test instance",
    "computational experiment",
    "numerical experiment",
    "parameter setting",
    "sensitivity",
    "friedman",
    "wilcoxon",
    "statistical",
    "managerial implication",
    "results",
)

_REFERENCE_CUE_KEYWORDS = (
    "doi.org",
    "http",
    "comput.",
    "trans. res.",
    "oper. res.",
    "eur. j.",
    "expert syst.",
    "int. trans.",
    "swarm evol.",
    "j. manuf.",
    "machine learn.",
    "omega",
)


def build_model_region_chunks(
    chunks: list[dict[str, Any]],
    paper_id: str,
    *,
    max_region_chars: int = 5000,
) -> list[dict[str, Any]]:
    """Create model-region chunks from already-built paper chunks."""
    source_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("paper_id") == paper_id
        and chunk.get("modality") != "math_model"
        and not str(chunk.get("chunk_type", "")).startswith("model_region")
    ]
    if not source_chunks:
        return []

    candidates: list[dict[str, Any]] = []
    candidates.extend(_build_section_region_chunks(source_chunks, paper_id, max_region_chars))
    candidates.extend(_build_equation_region_chunks(source_chunks, paper_id))
    candidates.extend(_build_table_region_chunks(source_chunks, paper_id, max_region_chars))
    candidates.extend(_build_figure_region_chunks(source_chunks, paper_id))

    deduped = _dedupe_region_chunks(candidates)
    for idx, chunk in enumerate(deduped):
        chunk["id"] = model_region_chunk_id(paper_id, idx)
        chunk["end"] = len(chunk.get("content", ""))
    return deduped


def remove_existing_model_region_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop generated model-region chunks while preserving model cards."""
    return [
        chunk
        for chunk in chunks
        if chunk.get("modality") != "model_region"
        and not str(chunk.get("chunk_type", "")).startswith("model_region")
    ]


def _build_section_region_chunks(
    chunks: list[dict[str, Any]],
    paper_id: str,
    max_region_chars: int,
) -> list[dict[str, Any]]:
    text_chunks = [
        (idx, chunk)
        for idx, chunk in enumerate(chunks)
        if chunk.get("modality", "text") == "text"
    ]
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for pos, chunk in text_chunks:
        text = str(chunk.get("content", ""))
        if _is_noisy_region(text, str(chunk.get("section", ""))):
            continue
        score = _model_region_score(text, str(chunk.get("section", "")))
        if _is_core_formulation_text(text) and score >= 8:
            scored.append((score, pos, chunk))

    if not scored:
        return []

    scored.sort(key=lambda item: item[1])
    grouped_positions = _group_adjacent_positions([pos for _, pos, _ in scored])
    by_pos = {pos: chunk for _, pos, chunk in scored}
    result: list[dict[str, Any]] = []
    for positions in grouped_positions:
        group_chunks = [by_pos[pos] for pos in positions if pos in by_pos]
        if not group_chunks:
            continue
        merged = _merge_source_chunks(
            paper_id=paper_id,
            source_chunks=group_chunks,
            region_type="model_section",
            title="Mathematical model section",
            max_chars=max_region_chars,
        )
        if merged:
            result.extend(_split_long_region(merged, max_region_chars))
    return result


def _build_equation_region_chunks(
    chunks: list[dict[str, Any]],
    paper_id: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for chunk in chunks:
        modality = chunk.get("modality", "text")
        content = str(chunk.get("content", ""))
        if _is_noisy_region(content, str(chunk.get("section", ""))):
            continue
        if (modality == "equation" or _looks_like_equation_block(content)) and _is_core_formulation_text(content):
            region_type = _classify_model_region(content, str(chunk.get("section", "")))
            result.append(
                _make_region_chunk(
                    paper_id=paper_id,
                    content=_region_content_header(region_type, paper_id) + "\n" + content.strip(),
                    region_type=region_type,
                    pages=_as_pages(chunk.get("page", [])),
                    source_chunk_ids=[str(chunk.get("id", ""))],
                    score=_model_region_score(content, str(chunk.get("section", ""))),
                )
            )
    return result


def _build_table_region_chunks(
    chunks: list[dict[str, Any]],
    paper_id: str,
    max_region_chars: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for chunk in chunks:
        if chunk.get("modality") != "table":
            continue
        content = str(chunk.get("content", ""))
        caption = str(chunk.get("caption", ""))
        if _is_noisy_region(content + "\n" + caption, str(chunk.get("section", ""))):
            continue
        score = _keyword_score(content + "\n" + caption, _NOTATION_KEYWORDS)
        if score < 2 and not _has_math_signal(content):
            continue
        if not _is_core_formulation_text(content + "\n" + caption):
            continue
        body = "\n".join(part for part in (caption.strip(), content.strip()) if part)
        region = _make_region_chunk(
            paper_id=paper_id,
            content=_region_content_header("notation_table", paper_id) + "\n" + body[:max_region_chars],
            region_type="notation_table",
            pages=_as_pages(chunk.get("page", [])),
            source_chunk_ids=[str(chunk.get("id", ""))],
            score=score + 5,
        )
        result.append(region)
    return result


def _build_figure_region_chunks(
    chunks: list[dict[str, Any]],
    paper_id: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for chunk in chunks:
        if chunk.get("modality") != "figure":
            continue
        content = str(chunk.get("caption_merged") or chunk.get("content", ""))
        if _is_noisy_region(content, str(chunk.get("section", ""))):
            continue
        score = _model_region_score(content, str(chunk.get("section", "")))
        if score < 9 or not _is_core_formulation_text(content):
            continue
        region_type = "model_figure"
        result.append(
            _make_region_chunk(
                paper_id=paper_id,
                content=_region_content_header(region_type, paper_id) + "\n" + content.strip(),
                region_type=region_type,
                pages=_as_pages(chunk.get("page", [])),
                source_chunk_ids=[str(chunk.get("id", ""))],
                score=score,
                image_path=chunk.get("image_path"),
                bbox=chunk.get("bbox"),
            )
        )
    return result


def _merge_source_chunks(
    *,
    paper_id: str,
    source_chunks: list[dict[str, Any]],
    region_type: str,
    title: str,
    max_chars: int,
) -> dict[str, Any] | None:
    parts: list[str] = []
    source_ids: list[str] = []
    pages: list[int] = []
    score = 0
    for chunk in source_chunks:
        content = str(chunk.get("content", "")).strip()
        if not content:
            continue
        source_id = str(chunk.get("id", ""))
        source_ids.append(source_id)
        pages.extend(_as_pages(chunk.get("page", [])))
        score += _model_region_score(content, str(chunk.get("section", "")))
        parts.append(f"[{source_id}]\n{content}")
    if not parts:
        return None
    content = "\n\n".join([_region_content_header(region_type, paper_id, title), *parts])
    return _make_region_chunk(
        paper_id=paper_id,
        content=content[:max_chars],
        region_type=region_type,
        pages=sorted(set(pages)),
        source_chunk_ids=_unique(source_ids),
        score=score,
    )


def _make_region_chunk(
    *,
    paper_id: str,
    content: str,
    region_type: str,
    pages: list[int],
    source_chunk_ids: list[str],
    score: int,
    image_path: str | None = None,
    bbox: Any = None,
) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "id": "",
        "paper_id": paper_id,
        "modality": "model_region",
        "chunk_type": f"model_region_{region_type}",
        "model_region_type": region_type,
        "content": content.strip(),
        "caption": "",
        "page": pages,
        "section": f"model_region:{region_type}",
        "start": 0,
        "end": 0,
        "source_chunk_ids": _unique([cid for cid in source_chunk_ids if cid]),
        "model_region_score": score,
        "embedding_text": [],
        "embedding_image": [],
    }
    if image_path:
        chunk["image_path"] = image_path
    if bbox is not None:
        chunk["bbox"] = bbox
    _annotate_model_region_chunk(chunk)
    return chunk


def _split_long_region(chunk: dict[str, Any], max_chars: int) -> list[dict[str, Any]]:
    content = str(chunk.get("content", ""))
    if len(content) <= max_chars:
        return [chunk]
    result: list[dict[str, Any]] = []
    step = max(1000, max_chars - 400)
    for start in range(0, len(content), step):
        part = dict(chunk)
        part["content"] = content[start : start + max_chars].strip()
        if not part["content"]:
            continue
        result.append(part)
    return result


def _dedupe_region_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for chunk in chunks:
        content = re.sub(r"\s+", " ", str(chunk.get("content", ""))).strip()
        if len(content) < 80:
            continue
        key = (
            str(chunk.get("paper_id", "")),
            str(chunk.get("chunk_type", "")),
            content[:260].casefold(),
        )
        current = best_by_key.get(key)
        if current is None or int(chunk.get("model_region_score", 0)) > int(
            current.get("model_region_score", 0)
        ):
            best_by_key[key] = chunk
    ordered = sorted(
        best_by_key.values(),
        key=lambda c: (
            min(_as_pages(c.get("page", [])) or [999999]),
            _region_sort_key(str(c.get("model_region_type", ""))),
            -int(c.get("model_region_score", 0)),
        ),
    )
    return ordered


def _group_adjacent_positions(positions: list[int]) -> list[list[int]]:
    if not positions:
        return []
    groups: list[list[int]] = []
    current = [positions[0]]
    for pos in positions[1:]:
        if pos <= current[-1] + 1:
            current.append(pos)
        else:
            groups.append(current)
            current = [pos]
    groups.append(current)
    return groups


def _model_region_score(text: str, section: str = "") -> int:
    joined = f"{section}\n{text}"
    score = 0
    score += _keyword_score(joined, _SECTION_KEYWORDS)
    score += _keyword_score(joined, _OBJECTIVE_KEYWORDS)
    score += _keyword_score(joined, _CONSTRAINT_KEYWORDS)
    score += _keyword_score(joined, _NOTATION_KEYWORDS)
    if _has_math_signal(text):
        score += 4
    if _looks_like_equation_block(text):
        score += 4
    if re.search(r"\b(eq\.|equation|constraint)s?\s*\(?\d+", text, flags=re.IGNORECASE):
        score += 2
    return score


def _annotate_model_region_chunk(chunk: dict[str, Any]) -> None:
    content = str(chunk.get("content", ""))
    section = str(chunk.get("section", ""))
    region_type = str(chunk.get("model_region_type", ""))
    elements = _detect_model_elements(content, section, region_type)
    operator_hints = _detect_keyword_groups(content, _OPERATOR_HINT_KEYWORDS)
    domain_signals = _detect_keyword_groups(content, _DOMAIN_SIGNAL_KEYWORDS)
    hhc_signals = _detect_keyword_groups(content, _HHC_SIGNAL_KEYWORDS)
    formula_count = _formula_signal_count(content)

    chunk["model_elements"] = elements
    chunk["operator_hints"] = operator_hints
    chunk["domain_signals"] = domain_signals
    chunk["hhc_signals"] = hhc_signals
    chunk["formula_signal_count"] = formula_count
    chunk["model_evidence_score"] = _model_evidence_score(
        chunk,
        elements=elements,
        operator_hints=operator_hints,
        domain_signals=domain_signals,
        hhc_signals=hhc_signals,
        formula_count=formula_count,
    )


def _detect_model_elements(text: str, section: str, region_type: str) -> list[str]:
    joined = f"{section}\n{text}"
    elements: set[str] = set()
    for element, keywords in _MODEL_ELEMENT_KEYWORDS.items():
        if _keyword_score(joined, keywords) > 0:
            elements.add(element)
    if _has_math_signal(text) or _looks_like_equation_block(text):
        elements.add("formula")
    if "table" in region_type:
        elements.add("table")
    if "figure" in region_type:
        elements.add("figure")
    if "objective" in region_type:
        elements.add("objective")
    if "constraint" in region_type:
        elements.add("constraints")
    if "notation" in region_type:
        elements.update({"sets", "parameters", "variables"})
    return [element for element in _MODEL_ELEMENT_ORDER if element in elements]


def _detect_keyword_groups(
    text: str,
    keyword_map: dict[str, tuple[str, ...]],
) -> list[str]:
    lower = text.lower()
    result = [
        group
        for group, keywords in keyword_map.items()
        if any(keyword.lower() in lower for keyword in keywords)
    ]
    return result


def _formula_signal_count(text: str) -> int:
    signals = 0
    signals += len(re.findall(r"\(\d+\)", text))
    signals += len(re.findall(r"(<=|>=|=)", text))
    signals += sum(text.count(symbol) for symbol in _MATH_SYMBOLS)
    signals += len(re.findall(r"\b(min|max)\b|\bs\.t\.\b|subject to", text, flags=re.IGNORECASE))
    return signals


def _model_evidence_score(
    chunk: dict[str, Any],
    *,
    elements: list[str],
    operator_hints: list[str],
    domain_signals: list[str],
    hhc_signals: list[str],
    formula_count: int,
) -> float:
    score = float(chunk.get("model_region_score", 0) or 0)
    score += min(formula_count, 12) * 0.8
    score += len(elements) * 1.2
    score += len(operator_hints) * 1.0
    score += len(domain_signals) * 1.0
    score += len(hhc_signals) * 0.8
    region_type = str(chunk.get("model_region_type", ""))
    if region_type in {"objective_block", "constraint_block", "notation_table", "notation_block"}:
        score += 3.0
    elif region_type == "model_section":
        score += 1.5
    return round(score, 4)


def _is_noisy_region(text: str, section: str = "") -> bool:
    lower = f"{section}\n{text}".lower()
    if any(keyword in lower for keyword in _NOISY_SECTION_KEYWORDS):
        return True
    reference_hits = len(re.findall(r"\b\d{4}[a-z]?\b|doi\.org|journal|vol\.|pp\.", lower))
    reference_cues = sum(1 for keyword in _REFERENCE_CUE_KEYWORDS if keyword in lower)
    if reference_hits >= 4 or reference_cues >= 2:
        return True
    algorithm_hits = sum(1 for keyword in _ALGORITHM_NOISE_KEYWORDS if keyword in lower)
    if algorithm_hits >= 2 and not _is_core_formulation_text(text):
        return True
    experiment_hits = sum(1 for keyword in _EXPERIMENT_NOISE_KEYWORDS if keyword in lower)
    if experiment_hits >= 2 and not _is_core_formulation_text(text):
        return True
    if "algorithm" in lower and ("input:" in lower or "output:" in lower) and not _is_core_formulation_text(text):
        return True
    return False


def _classify_model_region(text: str, section: str = "") -> str:
    joined = f"{section}\n{text}"
    objective_score = _keyword_score(joined, _OBJECTIVE_KEYWORDS)
    constraint_score = _keyword_score(joined, _CONSTRAINT_KEYWORDS)
    notation_score = _keyword_score(joined, _NOTATION_KEYWORDS)
    if objective_score >= max(constraint_score, notation_score) and objective_score > 0:
        return "objective_block"
    if constraint_score >= max(objective_score, notation_score) and constraint_score > 0:
        return "constraint_block"
    if notation_score > 0:
        return "notation_block"
    return "equation_block"


def _keyword_score(text: str, keywords: tuple[str, ...]) -> int:
    lower = text.lower()
    score = 0
    for keyword in keywords:
        if keyword in lower:
            score += 2 if len(keyword) > 8 else 1
    return score


def _has_math_signal(text: str) -> bool:
    return bool(
        any(symbol in text for symbol in _MATH_SYMBOLS)
        or re.search(r"\b(min|max)\s*[zcf]?\s*[=:]", text, flags=re.IGNORECASE)
        or re.search(r"\(\d+\)", text)
        or re.search(r"(<=|>=|=)", text)
    )


def _has_model_formula_signal(text: str) -> bool:
    lower = text.lower()
    if any(symbol in text for symbol in ("∑", "∀", "∈", "≤", "≥")):
        return True
    if re.search(r"\b(min|max)(imize|imization|imise|imisation)?\b", lower):
        return True
    if re.search(r"\bsubject to\b|\bs\.t\.\b|\bconstraint", lower):
        return True
    if re.search(r"\b(eq\.|equation)\s*\(?\d+", lower):
        return True
    return False


def _is_core_formulation_text(text: str) -> bool:
    lower = text.lower()
    if _looks_like_references(text):
        return False
    if any(
        keyword in lower
        for keyword in (
            "mathematical model",
            "mixed integer programming model below",
            "build a mixed integer programming model below",
            "can be formulated as",
            "model can be formulated as",
            "decision variables:",
            "description of ffspaw-js symbols",
            "description of symbols",
            "notations\n",
            "notations description",
            "s.t.",
            "subject to",
        )
    ):
        return True
    if re.search(r"\bmin\s*\(|\bmax\s+z\s*=", lower):
        return True
    equation_numbers = len(re.findall(r"\(\d+\)", text))
    math_symbol_hits = sum(1 for symbol in ("∑", "∀", "∈", "≤", "≥") if symbol in text)
    if equation_numbers >= 4 and math_symbol_hits >= 2:
        return True
    if "set of" in lower and "parameters" in lower and "decision variables" in lower:
        return True
    return False


def _looks_like_references(text: str) -> bool:
    lower = text.lower()
    reference_hits = len(re.findall(r"\b\d{4}[a-z]?\b|doi\.org|journal|vol\.|pp\.", lower))
    reference_cues = sum(1 for keyword in _REFERENCE_CUE_KEYWORDS if keyword in lower)
    author_pattern_hits = len(re.findall(r"\b[A-Z][a-z]+,\s+[A-Z]\.", text))
    return reference_hits >= 4 or reference_cues >= 2 or author_pattern_hits >= 4


def _looks_like_equation_block(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 8:
        return False
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return False
    math_lines = sum(1 for line in lines if _has_math_signal(line))
    if math_lines >= 2:
        return True
    if len(stripped) < 900 and math_lines >= 1 and re.search(r"(\bforall\b|∀|∈|≤|≥|∑)", stripped):
        return True
    return False


def _region_content_header(region_type: str, paper_id: str, title: str | None = None) -> str:
    labels = {
        "model_section": "Mathematical model section",
        "notation_table": "Notation / parameter / variable table",
        "notation_block": "Notation / parameter / variable block",
        "equation_block": "Equation block",
        "objective_block": "Objective function block",
        "constraint_block": "Constraint block",
        "model_figure": "Model-related figure or rendered page",
    }
    return f"{title or labels.get(region_type, region_type)} for paper {paper_id}."


def _region_sort_key(region_type: str) -> int:
    order = {
        "model_section": 0,
        "notation_table": 1,
        "notation_block": 2,
        "objective_block": 3,
        "constraint_block": 4,
        "equation_block": 5,
        "model_figure": 6,
    }
    return order.get(region_type, 99)


def _as_pages(value: Any) -> list[int]:
    if value in (None, ""):
        return []
    values = value if isinstance(value, list) else [value]
    pages: list[int] = []
    for item in values:
        try:
            pages.append(int(item))
        except (TypeError, ValueError):
            continue
    return pages


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def summarize_region_counts(chunks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        counts[str(chunk.get("model_region_type", "unknown"))] += 1
    return dict(sorted(counts.items()))
