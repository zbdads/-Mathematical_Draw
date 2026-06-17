"""Evidence Scorer：证据数据容器定义。

EvidenceScorer（LLM 打分）已废弃，改由 reranker 直接提供检索分数。
ScoredEvidence 作为数据容器继续供 answer_synthesizer.py 和 evidence_store.py 使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class ScoredEvidence:
    """经检索后的证据条目。

    新语义：
    - llm_relevance_score: 固定 -1（未打分）
    - focused_summary: 固定 ""（不使用）
    - final_score: 等于 retrieval_score（reranker 分）
    """

    chunk_id: str
    paper_id: str
    page: int | list[int]
    modality: str
    content: str
    section: str
    source_query: str
    retrieval_path: str
    retrieval_score: float
    llm_relevance_score: int    # 固定 -1（未打分）
    focused_summary: str        # 固定 ""（不使用）
    final_score: float          # 等于 retrieval_score
    figure_id: str | None = None
    image_path: str | None = None
    hit_paths: list[str] | None = None
    matched_sub_queries: list[str] | None = None

    def to_evidence_dict(self) -> dict[str, Any]:
        """转换为 format_answer / check_citation 所需的 dict 格式。"""
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "page": self.page,
            "modality": self.modality,
            "content": self.content,
            "section": self.section,
            "score": self.final_score,
            "retrieval_score": self.retrieval_score,
            "llm_relevance_score": self.llm_relevance_score,
            "figure_id": self.figure_id,
            "image_path": self.image_path,
            "matched_sub_queries": self.matched_sub_queries or [],
        }
