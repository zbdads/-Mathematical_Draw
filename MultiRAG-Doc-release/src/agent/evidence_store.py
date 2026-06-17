"""Evidence Store：canonical 证据池，独立于 AgentState。

职责：
- 持有所有 EvidenceRecord 原始全文与底层元数据。
- 按 chunk_id/figure_id 去重。
- 记录每条证据的 source_queries、命中轮次（hit_count）、邻接 IDs。
- 支持邻域扩展（adjacent chunk 扩展，用于 citation recovery）。
- 供回答生成器直接消费（转换为 ScoredEvidence 列表）。

AgentState 不直接拥有原始证据全文，通过 record_id 引用 EvidenceStore。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.query.retrieval_orchestrator import EvidenceCandidate
    from src.agent.state import SelectedView

from src.agent.evidence_scorer import ScoredEvidence

# ── EvidenceRecord ─────────────────────────────────────────────────────────────

_PREVIEW_CHARS = 200


@dataclass
class EvidenceRecord:
    """canonical 证据条目，由 EvidenceStore 独立维护。"""

    record_id: str
    chunk_id: str
    paper_id: str
    page: int | list[int] | None
    modality: str
    content: str
    source_queries: list[str] = field(default_factory=list)
    hit_count: int = 0
    adjacent_ids: list[str] = field(default_factory=list)  # 邻接 chunk_id（元数据层）
    retrieval_score: float = 0.0
    figure_id: str | None = None
    image_path: str | None = None
    section: str = ""

    def content_preview(self) -> str:
        preview = self.content[:_PREVIEW_CHARS].replace("\n", " ")
        if len(self.content) > _PREVIEW_CHARS:
            preview += "..."
        return preview


# ── EvidenceStore ──────────────────────────────────────────────────────────────

class EvidenceStore:
    """持有 canonical evidence records，提供去重、查询、邻域扩展能力。"""

    def __init__(self) -> None:
        # chunk_id -> EvidenceRecord
        self._records: dict[str, EvidenceRecord] = {}
        # record_id -> chunk_id（反向查找）
        self._id_map: dict[str, str] = {}

    # ── 写入 ─────────────────────────────────────────────────────────────────

    def add_candidates(
        self,
        candidates: list["EvidenceCandidate"],
        source_query: str,
    ) -> list[str]:
        """批量写入 EvidenceCandidate，去重后返回新增的 record_ids。

        同一 chunk_id 已存在时只更新 source_queries / hit_count，不覆盖原文。
        """
        new_ids: list[str] = []
        for c in candidates:
            existing = self._records.get(c.chunk_id)
            if existing is not None:
                if source_query not in existing.source_queries:
                    existing.source_queries.append(source_query)
                existing.hit_count += 1
            else:
                record_id = str(uuid.uuid4())[:8]
                rec = EvidenceRecord(
                    record_id=record_id,
                    chunk_id=c.chunk_id,
                    paper_id=c.paper_id,
                    page=c.page,
                    modality=c.modality,
                    content=c.content,
                    source_queries=[source_query],
                    hit_count=1,
                    retrieval_score=c.retrieval_score,
                    figure_id=c.figure_id,
                    image_path=c.image_path,
                    section=getattr(c, "section", ""),
                )
                self._records[c.chunk_id] = rec
                self._id_map[record_id] = c.chunk_id
                new_ids.append(record_id)
        return new_ids

    # ── 查询 ─────────────────────────────────────────────────────────────────

    def get_by_chunk_id(self, chunk_id: str) -> EvidenceRecord | None:
        return self._records.get(chunk_id)

    def get_by_record_id(self, record_id: str) -> EvidenceRecord | None:
        chunk_id = self._id_map.get(record_id)
        if chunk_id is None:
            return None
        return self._records.get(chunk_id)

    def all_records(self) -> list[EvidenceRecord]:
        return list(self._records.values())

    def all_chunk_ids(self) -> list[str]:
        return list(self._records.keys())

    def count(self) -> int:
        return len(self._records)

    def count_text(self) -> int:
        """仅计算 text 赛道（非 figure）的证据数量，用于 evidence budget 管理。"""
        return sum(1 for r in self._records.values() if r.modality != "figure")

    def remove_by_chunk_ids(self, chunk_ids: list[str]) -> list[str]:
        """从 store 中移除指定 chunk_id，返回实际移除的列表。"""
        removed = []
        for chunk_id in chunk_ids:
            rec = self._records.pop(chunk_id, None)
            if rec is not None:
                self._id_map.pop(rec.record_id, None)
                removed.append(chunk_id)
        return removed
    # ── 邻域扩展（citation recovery 系统内部动作）────────────────────────────

    def expand_adjacent(
        self,
        anchor_chunk_ids: list[str],
        all_chunks: list[dict[str, Any]],
        window: int = 1,
    ) -> list[str]:
        """对 anchor 证据集做相邻 chunk 扩展，返回新增 record_ids。

        通过 all_chunks（按 paper_id + page 排序的全量 chunk 列表）查找邻接 chunk。
        仅扩展已存在于 chunk 库中的邻接条目，不做模型调用。
        邻接 chunk 不占 evidence_cap 计数（budget 在检索阶段已耗尽时 citation recovery 仍可追加）。

        Args:
            anchor_chunk_ids: 需要扩展的 chunk_id 列表。
            all_chunks: MetadataStore 返回的全量 chunk dict 列表（含 chunk_id 字段）。
            window: 每侧扩展的相邻 chunk 数量（Phase 1 仅支持 1）。

        Returns:
            新增 record_ids（已存在的不重复计入）。
        """
        # 建立 chunk_id -> index 的快速映射（按原始顺序）
        id_to_idx: dict[str, int] = {
            c.get("chunk_id", c.get("figure_id", "")): i
            for i, c in enumerate(all_chunks)
        }

        new_ids: list[str] = []
        for anchor_id in anchor_chunk_ids:
            idx = id_to_idx.get(anchor_id)
            if idx is None:
                continue
            for delta in range(-window, window + 1):
                if delta == 0:
                    continue
                neighbor_idx = idx + delta
                if neighbor_idx < 0 or neighbor_idx >= len(all_chunks):
                    continue
                neighbor_chunk = all_chunks[neighbor_idx]
                neighbor_id = neighbor_chunk.get("chunk_id", neighbor_chunk.get("figure_id", ""))
                if not neighbor_id or neighbor_id in self._records:
                    continue
                # 作为系统内部扩展写入（source_query 标记为 _expand）
                record_id = str(uuid.uuid4())[:8]
                rec = EvidenceRecord(
                    record_id=record_id,
                    chunk_id=neighbor_id,
                    paper_id=neighbor_chunk.get("paper_id", ""),
                    page=neighbor_chunk.get("page", -1),
                    modality=neighbor_chunk.get("modality", "text"),
                    content=neighbor_chunk.get("content", ""),
                    source_queries=["_expand"],
                    hit_count=0,
                    retrieval_score=0.0,
                    section=neighbor_chunk.get("section", ""),
                )
                # 更新 anchor 的 adjacent_ids 记录
                anchor_rec = self._records.get(anchor_id)
                if anchor_rec is not None:
                    anchor_rec.adjacent_ids.append(neighbor_id)

                self._records[neighbor_id] = rec
                self._id_map[record_id] = neighbor_id
                new_ids.append(record_id)

        return new_ids

    # ── 转换为 AgentState 视图 ────────────────────────────────────────────────

    def to_evidence_views(self) -> list["SelectedView"]:
        """转换为 AgentState.evidence 的视图列表（按 retrieval_score 降序）。"""
        from src.agent.state import SelectedView
        views = []
        for rec in sorted(self._records.values(), key=lambda r: r.retrieval_score, reverse=True):
            views.append(SelectedView(
                record_id=rec.record_id,
                chunk_id=rec.chunk_id,
                paper_id=rec.paper_id,
                page=rec.page,
                modality=rec.modality,
                content=rec.content,
                final_score=rec.retrieval_score,
                llm_relevance_score=-1,
                source_query=rec.source_queries[0] if rec.source_queries else "",
                matched_sub_queries=rec.source_queries,
                figure_id=rec.figure_id,
                image_path=rec.image_path,
            ))
        return views

    def get_all_as_scored_evidence(self) -> list[ScoredEvidence]:
        """全量 records 转为 ScoredEvidence 列表（按 retrieval_score 降序），供回答生成器消费。"""
        result = []
        for rec in sorted(self._records.values(), key=lambda r: r.retrieval_score, reverse=True):
            result.append(ScoredEvidence(
                chunk_id=rec.chunk_id,
                paper_id=rec.paper_id,
                page=rec.page if rec.page is not None else -1,
                modality=rec.modality,
                content=rec.content,
                section=rec.section,
                source_query=rec.source_queries[0] if rec.source_queries else "",
                retrieval_path=rec.modality,
                retrieval_score=rec.retrieval_score,
                llm_relevance_score=-1,
                focused_summary="",
                final_score=rec.retrieval_score,
                figure_id=rec.figure_id,
                image_path=rec.image_path,
                matched_sub_queries=rec.source_queries,
            ))
        return result
