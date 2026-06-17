"""文本语义检索模块：top-k 检索，支持 paper_id 过滤。

当前检索器直接对输入 query 做向量召回，并在需要时按 paper_id 过滤结果。
若后续需要接入对话式检索，可在外层增加 query 改写和 token budget 控制。
"""

from __future__ import annotations

from typing import Any

from src.config import CFG
from src.index.metadata_store import MetadataStore
from src.index.vector_store import VectorStore
from src.ingestion.text_embedder import TextEmbedder


class TextRetriever:
    """基于 FAISS 的文本 top-k 语义检索器。"""

    def __init__(
        self,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
        embedder: TextEmbedder | None = None,
    ) -> None:
        self._vs = vector_store
        self._ms = metadata_store
        self._embedder = embedder or TextEmbedder()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        paper_id: str | None = None,
        query_vec: Any | None = None,
    ) -> list[dict[str, Any]]:
        """语义检索，返回 top-k 最相关 chunk。

        Args:
            query: 查询文本。
            top_k: 返回结果数量，默认取 CFG.retriever.top_k。
            paper_id: 若指定，则仅返回该论文的 chunk（后过滤）。
            query_vec: 可选的预计算 query embedding，用于同一问题多次按论文过滤时复用。

        Returns:
            [
                {
                    "rank": int,
                    "score": float,
                    "chunk_id": str,
                    "paper_id": str,
                    "modality": str,
                    "page": list[int],
                    "content": str,
                    "section": str,
                }
            ]
        """
        k = top_k or CFG.retriever.top_k

        # 若有 paper_id 过滤，扩大 FAISS 搜索范围以补偿过滤损耗
        search_k = k * 10 if paper_id else k

        qvec = query_vec if query_vec is not None else self._embedder.encode_query(query)
        scores, indices = self._vs.search(qvec, min(search_k, self._vs.ntotal))

        results: list[dict[str, Any]] = []
        rank = 1
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            chunk = self._ms.get(int(idx))
            if paper_id and chunk.get("paper_id") != paper_id:
                continue
            result = {
                "rank": rank,
                "score": float(score),
                "chunk_id": chunk.get("id", ""),
                "paper_id": chunk.get("paper_id", ""),
                "modality": chunk.get("modality", "text"),
                "page": chunk.get("page", -1),
                "content": chunk.get("content", ""),
                "section": chunk.get("section", ""),
            }
            for key in (
                "chunk_type",
                "source_chunk_ids",
                "evidence_refs",
                "model_card",
                "model_region_type",
                "model_region_score",
                "model_elements",
                "operator_hints",
                "domain_signals",
                "hhc_signals",
                "formula_signal_count",
                "model_evidence_score",
            ):
                if key in chunk:
                    result[key] = chunk[key]
            results.append(result)
            rank += 1
            if len(results) >= k:
                break

        return results
