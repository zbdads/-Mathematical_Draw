"""图像检索模块：基于 Qwen3-VL-Embedding-8B 的 text -> image 跨模态检索。

设计要点：
    - 图像索引（4096 维）与文本索引（1024 维）完全分开，不可混用。
    - encode_text_query 将文本查询映射到图像共享向量空间。
    - 索引类型：IndexFlatIP + L2 归一化 = cosine 相似度检索（精确暴力搜索）。
    - figure metadata 持久化为 JSON，与 FAISS 索引位置严格对齐。
    - ingest 时图像索引写入纯图像 embedding。
    - caption_index_qwenvl 写入 QwenVL 文本 embedding（caption_merged）。
    - 检索时先在 QwenVL 空间内做 image+caption 的 beta 融合，再与 BGE caption 路做 alpha 融合。

参考架构：
    TextRetriever（text_retriever.py）的 VectorStore + MetadataStore 模式，
    图像侧用 VectorStore(dim=4096) 和自管理的 figure metadata list 替代。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.config import CFG
from src.index.vector_store import VectorStore
from src.retrieval.fusion import merge_figure_with_caption_hits, merge_qwenvl_dual_path_hits

if TYPE_CHECKING:
    from src.ingestion.image_embedder import ImageEmbedder



class ImageRetriever:
    """基于 Qwen3-VL-Embedding-8B 的图像检索器，支持 text -> image top-k 语义检索。"""

    def __init__(
        self,
        embedder: ImageEmbedder | None = None,
        index_path: Path | None = None,
        metadata_path: Path | None = None,
        caption_index_path: Path | None = None,
    ) -> None:
        self._embedder = embedder  # 懒加载，首次调用时初始化
        self._embedder_lock = threading.Lock()
        self._index_path = index_path or CFG.paths.index_dir / "image_index.faiss"
        self._caption_index_path = (
            caption_index_path or CFG.paths.index_dir / "caption_index_qwenvl.faiss"
        )
        self._metadata_path = (
            metadata_path or CFG.paths.index_dir / "image_metadata.json"
        )
        self._vs: VectorStore | None = None
        self._caption_vs_qwenvl: VectorStore | None = None
        self._figures: list[dict[str, Any]] = []

    def _figure_meta_by_id(self, figure_id: str) -> dict[str, Any] | None:
        """Look up persisted figure metadata by ``figure_id``."""
        for fig in self._figures:
            if str(fig.get("figure_id", "")) == figure_id:
                return fig
        return None

    def enrich_results_with_metadata(
        self,
        hits: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Backfill figure metadata for merged hits from persisted image metadata.

        Caption/BGE-only hits may come from the text metadata store, where figure chunks
        historically did not persist ``image_path``. For those hits, recover the missing
        fields from ``image_metadata.json`` using ``figure_id``.
        """
        if not hits:
            return hits

        enriched: list[dict[str, Any]] = []
        for hit in hits:
            figure_id = str(hit.get("figure_id", hit.get("chunk_id", "")))
            if not figure_id:
                enriched.append(hit)
                continue

            meta = self._figure_meta_by_id(figure_id)
            if meta is None:
                enriched.append(hit)
                continue

            merged = dict(hit)
            if not merged.get("image_path"):
                merged["image_path"] = meta.get("image_path")
            if not merged.get("caption"):
                merged["caption"] = meta.get("caption", "")
            if not merged.get("caption_merged"):
                merged["caption_merged"] = meta.get("caption_merged", meta.get("caption", ""))
            if not merged.get("content"):
                merged["content"] = (
                    meta.get("caption_merged")
                    or meta.get("caption", "")
                )
            if merged.get("page") in (None, -1, [], ""):
                merged["page"] = meta.get("page", merged.get("page"))
            enriched.append(merged)

        return enriched

    # ── 内部工具 ────────────────────────────────────────────────────────────

    def _get_embedder(self) -> ImageEmbedder:
        """懒加载 ImageEmbedder，避免无检索调用时初始化模型。"""
        if self._embedder is None:
            with self._embedder_lock:
                if self._embedder is None:
                    from src.ingestion.image_embedder import ImageEmbedder as _IE
                    self._embedder = _IE()
        return self._embedder

    def _load_if_exists(self) -> bool:
        """Load persisted image retrieval state when both files are present."""
        if self._index_path.exists() and self._metadata_path.exists():
            self._vs = VectorStore.load(self._index_path)
            with open(self._metadata_path, encoding="utf-8") as f:
                self._figures = json.load(f)
            if self._caption_index_path.exists():
                self._caption_vs_qwenvl = VectorStore.load(self._caption_index_path)
            return True
        return False

    def ensure_loaded(self) -> bool:
        """Ensure persisted image retrieval resources are loaded into memory."""
        if self._vs is not None:
            return True
        return self._load_if_exists()

    # ── 只读属性 ────────────────────────────────────────────────────────────

    @property
    def ntotal(self) -> int:
        """已入库的 figure 数量。"""
        return self._vs.ntotal if self._vs else 0

    # ── 入库 ────────────────────────────────────────────────────────────────

    def add(
        self,
        new_figures: list[dict[str, Any]],
        embeddings: "np.ndarray",
        caption_embeddings_qwenvl: "np.ndarray | None" = None,
        overwrite: bool = False,
    ) -> int:
        """将预编码的 figure embeddings 写入索引并持久化。

        图像解码、磁盘写入、向量化由 src.ingestion.image_ingestor.ingest_images() 完成；
        本方法只负责索引管理（VectorStore + metadata list + 落盘）。

        Args:
            new_figures: ingest_images() 返回的 figure metadata 列表。
            embeddings:  shape (n, dim) float32 ndarray，与 new_figures 一一对齐。
            overwrite:   True 时清空已有索引重建；False 时追加入库。

        Returns:
            成功写入的 figure 数量。
        """
        if not new_figures:
            print("  [警告] 无有效图像，图像索引未建立。")
            return 0

        if overwrite or self._vs is None:
            # _vs 未在内存中：从磁盘加载或新建
            if not overwrite and self._index_path.exists() and self._metadata_path.exists():
                self._vs = VectorStore.load(self._index_path)
                with open(self._metadata_path, encoding="utf-8") as f:
                    self._figures = json.load(f)
                if self._caption_index_path.exists():
                    self._caption_vs_qwenvl = VectorStore.load(self._caption_index_path)
            else:
                self._vs = VectorStore(dim=embeddings.shape[1])
                self._caption_vs_qwenvl = None
                self._figures = []
        # else: _vs 已在内存（共享实例跨论文复用），直接追加，无需重新加载

        img_ids = np.arange(self._vs.ntotal, self._vs.ntotal + len(embeddings), dtype=np.int64)
        self._vs.add(embeddings, img_ids)

        if caption_embeddings_qwenvl is not None and len(caption_embeddings_qwenvl) > 0:
            if self._caption_vs_qwenvl is None:
                self._caption_vs_qwenvl = VectorStore(dim=caption_embeddings_qwenvl.shape[1])
            self._caption_vs_qwenvl.add(caption_embeddings_qwenvl, img_ids)

        self._figures.extend(new_figures)

        self.save()
        print(f"  [ImageRetriever] 入库完成，索引总向量数：{self._vs.ntotal}")
        return len(new_figures)

    # ── 检索 ────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        paper_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """text -> image 检索，返回 top-k figure 证据列表。

        若索引未加载，自动从磁盘加载。

        Args:
            query: 文本查询，如 "show architecture figure"。
            top_k: 返回结果数量，默认取 CFG.retriever.top_k。
            paper_id: 若指定，仅返回该论文的 figure（后过滤）。

        Returns:
            [
                {
                    "figure_id": str,
                    "paper_id": str,
                    "page": int,
                    "caption": str,
                    "image_path": str,
                    "score": float,    # cosine 相似度（0 ~ 1）
                    "modality": "figure",
                }
            ]
        """
        if not self.ensure_loaded() or self._vs is None or self._vs.ntotal == 0:
            return []

        k = top_k or CFG.retriever.top_k
        # 若有 paper_id 过滤，扩大搜索范围以补偿过滤损耗
        search_k = k * 10 if paper_id else k

        embedder = self._get_embedder()
        query_vec = embedder.encode_text_query(query)
        scores, indices = self._vs.search(query_vec, min(search_k, self._vs.ntotal))

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._figures[int(idx)]
            if paper_id and meta.get("paper_id") != paper_id:
                continue
            results.append(
                {
                    "figure_id": meta["figure_id"],
                    "paper_id": meta.get("paper_id", ""),
                    "page": meta["page"],
                    "caption": meta["caption"],
                    "image_path": meta.get("image_path", ""),
                    "score": float(score),
                    "modality": "figure",
                }
            )
            if len(results) >= k:
                break

        return results

    # ── 持久化 ──────────────────────────────────────────────────────────────

    def retrieve_qwenvl_caption(
        self,
        query: str,
        top_k: int | None = None,
        paper_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """在 QwenVL caption 索引中检索 figure。"""
        if (
            not self.ensure_loaded()
            or self._caption_vs_qwenvl is None
            or self._caption_vs_qwenvl.ntotal == 0
        ):
            return []

        k = top_k or CFG.retriever.top_k
        search_k = k * 10 if paper_id else k

        embedder = self._get_embedder()
        query_vec = embedder.encode_text_query(query)
        scores, indices = self._caption_vs_qwenvl.search(
            query_vec,
            min(search_k, self._caption_vs_qwenvl.ntotal),
        )

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._figures[int(idx)]
            if paper_id and meta.get("paper_id") != paper_id:
                continue
            caption_merged = meta.get("caption_merged", "")
            caption = meta.get("caption", "")
            results.append(
                {
                    "chunk_id": meta["figure_id"],
                    "figure_id": meta["figure_id"],
                    "paper_id": meta.get("paper_id", ""),
                    "page": meta["page"],
                    "caption": caption,
                    "content": caption_merged or caption,
                    "caption_merged": caption_merged or caption,
                    "image_path": meta.get("image_path", ""),
                    "score": float(score),
                    "modality": "figure",
                    "section": "",
                }
            )
            if len(results) >= k:
                break

        return results

    def retrieve_with_caption_hits(
        self,
        query: str,
        caption_hits: list[dict[str, Any]] | None = None,
        top_k: int | None = None,
        paper_id: str | None = None,
        alpha: float = 0.8,
        beta: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Retrieve figures with two-layer fusion (QwenVL beta + BGE alpha)."""
        image_hits = self.retrieve(query, top_k=top_k, paper_id=paper_id)
        caption_hits_qwenvl = self.retrieve_qwenvl_caption(query, top_k=top_k, paper_id=paper_id)
        figure_mm_hits = merge_qwenvl_dual_path_hits(
            image_hits_qwenvl=image_hits,
            caption_hits_qwenvl=caption_hits_qwenvl,
            beta=beta,
        )
        if not caption_hits:
            return self.enrich_results_with_metadata(figure_mm_hits)
        merged_hits = merge_figure_with_caption_hits(figure_mm_hits, caption_hits, alpha=alpha)
        return self.enrich_results_with_metadata(merged_hits)

    def save(self) -> None:
        """将向量索引和 figure metadata 持久化到磁盘。"""
        if self._vs is not None:
            self._vs.save(self._index_path)
        if self._caption_vs_qwenvl is not None:
            self._caption_vs_qwenvl.save(self._caption_index_path)
        self._metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._figures, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls,
        index_path: Path | None = None,
        metadata_path: Path | None = None,
        caption_index_path: Path | None = None,
        embedder: ImageEmbedder | None = None,
    ) -> "ImageRetriever":
        """从磁盘加载图像索引与 figure metadata，返回可直接检索的实例。"""
        retriever = cls(
            embedder=embedder,
            index_path=index_path,
            metadata_path=metadata_path,
            caption_index_path=caption_index_path,
        )
        retriever.ensure_loaded()
        return retriever

    @classmethod
    def load_if_available(
        cls,
        index_path: Path | None = None,
        metadata_path: Path | None = None,
        caption_index_path: Path | None = None,
        embedder: ImageEmbedder | None = None,
    ) -> "ImageRetriever | None":
        """Return a retriever only when persisted image artifacts exist."""
        retriever = cls(
            embedder=embedder,
            index_path=index_path,
            metadata_path=metadata_path,
            caption_index_path=caption_index_path,
        )
        if not retriever.ensure_loaded():
            return None
        return retriever
