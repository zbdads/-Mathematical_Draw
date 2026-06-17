"""GET /api/papers"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from src.config import CFG
from src.index.metadata_store import MetadataStore

router = APIRouter()

_COMBINED_CHUNKS_PATH: Path = CFG.paths.chunks_dir / "all_chunks.json"


@router.get("/papers")
async def list_papers() -> dict:
    """返回已入库的 paper_id 列表。冷启动/空库时返回空列表，不抛 500。"""
    if not _COMBINED_CHUNKS_PATH.exists():
        return {"paper_ids": []}
    try:
        ms = MetadataStore.load(_COMBINED_CHUNKS_PATH)
        return {"paper_ids": ms.paper_ids()}
    except Exception:
        return {"paper_ids": []}
