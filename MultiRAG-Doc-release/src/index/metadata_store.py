"""Chunk 元数据存取，chunk_id 与 VectorStore 的 IndexIDMap 显式绑定。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MetadataStore:
    """管理 chunk 字典，chunk_id 由本类分配并同步传给 VectorStore.add()。"""

    def __init__(self) -> None:
        self._chunks: dict[int, dict[str, Any]] = {}
        self._next_id: int = 0
        self._paper_index: dict[str, list[int]] = {}    # paper_id -> [chunk_ids]
        self._modality_index: dict[str, list[int]] = {}  # modality -> [chunk_ids]

    def __len__(self) -> int:
        return len(self._chunks)

    def add_chunks(self, chunks: list[dict[str, Any]]) -> list[int]:
        """追加 chunk 记录，返回分配的 chunk_id 列表（传给 VectorStore.add）。"""
        ids = list(range(self._next_id, self._next_id + len(chunks)))
        for chunk_id, chunk in zip(ids, chunks):
            self._chunks[chunk_id] = chunk
            pid = chunk.get("paper_id", "")
            if pid:
                self._paper_index.setdefault(pid, []).append(chunk_id)
            mod = chunk.get("modality", "text")
            self._modality_index.setdefault(mod, []).append(chunk_id)
        self._next_id += len(chunks)
        return ids

    def get(self, chunk_id: int) -> dict[str, Any]:
        """按 chunk_id 取 chunk（O(1) 字典查询）。"""
        return self._chunks[chunk_id].copy()

    def get_all(self) -> list[dict[str, Any]]:
        """返回全部 chunk 记录（只读副本）。"""
        return [chunk.copy() for chunk in self._chunks.values()]

    def paper_ids(self) -> list[str]:
        """返回已入库的所有 paper_id。"""
        return list(self._paper_index.keys())

    def filter_by_paper(self, paper_id: str) -> list[int]:
        """按 paper_id 返回对应的 chunk_id 列表（O(1) 倒排表查询）。"""
        return list(self._paper_index.get(paper_id, []))

    def filter_by_modality(self, modality: str) -> list[int]:
        """按 modality 返回对应的 chunk_id 列表（O(1) 倒排表查询）。"""
        return list(self._modality_index.get(modality, []))

    def _rebuild_indices(self) -> None:
        """从 _chunks 重建倒排表（load 后调用）。"""
        self._paper_index = {}
        self._modality_index = {}
        for chunk_id, chunk in self._chunks.items():
            pid = chunk.get("paper_id", "")
            if pid:
                self._paper_index.setdefault(pid, []).append(chunk_id)
            mod = chunk.get("modality", "text")
            self._modality_index.setdefault(mod, []).append(chunk_id)

    def save(self, path: Path) -> None:
        """序列化为 JSON 文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "next_id": self._next_id,
            "chunks": {str(k): v for k, v in self._chunks.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "MetadataStore":
        """从 JSON 文件加载并重建倒排表。"""
        store = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        store._chunks = {int(k): v for k, v in data["chunks"].items()}
        store._next_id = data["next_id"]
        store._rebuild_indices()
        return store
