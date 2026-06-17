"""Tool Schemas：agent 可调用工具的参数 schema。

Phase 1 只对 agent 开放 search_evidence。
expand_evidence / select_evidence 为系统内部动作，不在此注册。
finish / abort 为 route signal，不作为业务 tool。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ── 受控枚举 ─────────────────────────────────────────────────────────────────

ALLOWED_MODALITIES: set[str] = {"text", "figure", "text+figure"}
DEFAULT_MODALITY = "text+figure"

# Phase 1: 每步只允许单个 query
# Phase 1: 只允许单个 paper_id 过滤


# ── search_evidence ───────────────────────────────────────────────────────────

@dataclass
class SearchEvidenceArgs:
    """search_evidence 工具的受控参数。

    agent 填写 query / modalities / expected_evidence_count；
    paper_id 由代码侧管理，用户指定优先，agent 不可覆盖。
    """

    query: str
    modalities: Literal["text", "figure", "text+figure"] = DEFAULT_MODALITY
    expected_evidence_count: int = 3   # agent 声明本次期望拿几条，钳位到 [1, 5]
    # paper_id 由 controller 注入（用户指定优先），agent 请求值作为候选
    paper_id_hint: str | None = None

    def validate(self) -> None:
        """校验参数合法性，不合法时抛出 ValueError。"""
        if not self.query or not self.query.strip():
            raise ValueError("search_evidence: query 不能为空")
        if self.modalities not in ALLOWED_MODALITIES:
            raise ValueError(
                f"search_evidence: modalities={self.modalities!r} 不在受控枚举 {ALLOWED_MODALITIES} 中"
            )

    @classmethod
    def from_dict(cls, args: dict) -> "SearchEvidenceArgs":
        query = str(args.get("query", "")).strip()
        raw = args.get("modalities", [])
        if not isinstance(raw, list):
            raw = [raw] if isinstance(raw, str) else []
        has_text = "text" in raw
        has_figure = "figure" in raw
        if has_text and has_figure:
            modalities = "text+figure"
        elif has_figure:
            modalities = "figure"
        elif has_text:
            modalities = "text"
        else:
            modalities = DEFAULT_MODALITY
        paper_id_hint = args.get("paper_id_hint") or None
        expected_evidence_count = int(args.get("expected_evidence_count", 3))
        expected_evidence_count = max(1, min(5, expected_evidence_count))
        return cls(
            query=query,
            modalities=modalities,
            expected_evidence_count=expected_evidence_count,
            paper_id_hint=paper_id_hint,
        )


# ── expand_evidence（Phase 1 系统内部，不暴露给 agent）────────────────────────

ALLOWED_ANCHOR_SETS: set[str] = {"latest_hits", "selected", "citation_gap"}

@dataclass
class ExpandEvidenceArgs:
    """expand_evidence 的受控参数（仅系统内部使用）。

    anchor_set 受控枚举：latest_hits | selected | citation_gap
    Phase 1 默认 anchor_set="selected"（citation recovery fallback）。
    citation_gap 仅在 check_citation 提供结构化缺口信息时启用。
    """

    anchor_set: Literal["latest_hits", "selected", "citation_gap"] = "selected"
    window: int = 1   # Phase 1 只允许 window=1

    def validate(self) -> None:
        if self.anchor_set not in ALLOWED_ANCHOR_SETS:
            raise ValueError(
                f"expand_evidence: anchor_set={self.anchor_set!r} 不在受控枚举中"
            )
        if self.window != 1:
            raise ValueError("expand_evidence: Phase 1 只允许 window=1")

    @classmethod
    def from_dict(cls, args: dict) -> "ExpandEvidenceArgs":
        anchor_set = str(args.get("anchor_set", "selected"))
        if anchor_set not in ALLOWED_ANCHOR_SETS:
            anchor_set = "selected"
        return cls(anchor_set=anchor_set, window=1)
