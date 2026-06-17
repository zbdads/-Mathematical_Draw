"""FAISS 向量索引封装（IndexFlatIP，余弦相似度）。

当前实现聚焦本地单机检索，提供 add/search/save/load 四个核心能力。
如需扩展到多后端或支持精确删除，可在此接口上补充删除能力、后端工厂和
更通用的存储适配层。
"""

from __future__ import annotations

import threading
from pathlib import Path

import faiss
import numpy as np
try:
    import torch
except ImportError:  # pragma: no cover - CPU/API repro can run without torch
    torch = None

from src.config import CFG


def _faiss_gpu_available() -> bool:
    """判断当前 faiss 构建是否可用 GPU。"""
    if torch is None or not torch.cuda.is_available():
        return False
    if not hasattr(faiss, "StandardGpuResources"):
        return False
    try:
        return faiss.get_num_gpus() > 0
    except Exception:
        return False


class VectorStore:
    """FAISS IndexFlatIP 封装，使用 L2 归一化实现余弦相似度检索。"""

    def __init__(self, dim: int | None = None) -> None:
        self._dim = dim or CFG.embedder.embedding_dim
        self._gpu_resources = None
        self._lock = threading.Lock()
        self._index = self._to_runtime_index(
            faiss.IndexIDMap(faiss.IndexFlatIP(self._dim))
        )

    def _to_runtime_index(self, index: faiss.Index) -> faiss.Index:
        """若可用则把 CPU index 迁移到 GPU。"""
        if not _faiss_gpu_available():
            return index

        self._gpu_resources = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, index)
        print("  [faiss] 使用 GPU index: cuda:0")
        return gpu_index

    @staticmethod
    def _to_cpu_index(index: faiss.Index) -> faiss.Index:
        """持久化前将 GPU index 拉回 CPU。"""
        if hasattr(faiss, "index_gpu_to_cpu") and not isinstance(index, faiss.IndexFlatIP):
            try:
                return faiss.index_gpu_to_cpu(index)
            except Exception:
                pass
        return index

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    def add(self, embeddings: np.ndarray, ids: np.ndarray | list[int]) -> None:
        """向索引中添加向量（须已 L2 归一化）。

        Args:
            embeddings: (n, dim) float32 归一化向量。
            ids: (n,) int64 自定义 ID，与 MetadataStore 的 chunk_id 对应。
        """
        assert embeddings.ndim == 2, "embeddings 必须是 2D 数组"
        assert embeddings.shape[1] == self._dim, (
            f"维度不匹配：期望 {self._dim}，实际 {embeddings.shape[1]}"
        )
        ids_arr = np.asarray(ids, dtype=np.int64)
        assert len(ids_arr) == len(embeddings), "ids 与 embeddings 数量不一致"
        self._index.add_with_ids(embeddings, ids_arr)

    def search(
        self, query_vec: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """检索最相近的 top_k 向量。

        Args:
            query_vec: (1, dim) 归一化查询向量。
            top_k: 返回结果数量。

        Returns:
            (scores, indices)，均为 shape (1, top_k) 的 ndarray。
        """
        with self._lock:
            scores, indices = self._index.search(query_vec, top_k)
        return scores, indices

    def save(self, path: Path) -> None:
        """将索引写入文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._to_cpu_index(self._index), str(path))

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        """从文件加载索引。"""
        store = cls.__new__(cls)
        store._gpu_resources = None
        store._lock = threading.Lock()
        cpu_index = faiss.read_index(str(path))
        store._dim = cpu_index.d
        store._index = store._to_runtime_index(cpu_index)
        return store
