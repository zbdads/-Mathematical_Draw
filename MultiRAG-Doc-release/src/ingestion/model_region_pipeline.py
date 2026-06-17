"""Offline pipeline for rebuilding model-region chunks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.config import CFG
from src.index.metadata_store import MetadataStore
from src.index.vector_store import VectorStore
from src.ingestion.model_region_chunker import (
    build_model_region_chunks,
    remove_existing_model_region_chunks,
    summarize_region_counts,
)
from src.ingestion.text_embedder import TextEmbedder

COMBINED_CHUNKS_PATH = CFG.paths.chunks_dir / "all_chunks.json"
TEXT_INDEX_PATH = CFG.paths.index_dir / "text_index.faiss"


def _group_by_paper(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        paper_id = str(chunk.get("paper_id", "")).strip()
        if paper_id:
            grouped[paper_id].append(chunk)
    return dict(grouped)


def _embed_text_chunks(
    chunks: list[dict[str, Any]],
    embedder: TextEmbedder,
) -> Any:
    """Return embeddings, reusing existing per-chunk vectors when dimensions match."""
    import numpy as np

    dim = embedder.dim
    embeddings: list[list[float] | None] = []
    missing_indices: list[int] = []
    missing_texts: list[str] = []
    for idx, chunk in enumerate(chunks):
        existing = chunk.get("embedding_text")
        if isinstance(existing, list) and len(existing) == dim:
            embeddings.append(existing)
            continue
        embeddings.append(None)
        missing_indices.append(idx)
        missing_texts.append(str(chunk.get("content", "")))

    if missing_texts:
        new_embeddings = embedder.encode(missing_texts)
        for idx, emb in zip(missing_indices, new_embeddings):
            values = emb.tolist()
            embeddings[idx] = values
            chunks[idx]["embedding_text"] = values

    return np.array(embeddings, dtype="float32")


def rebuild_model_region_chunks(
    *,
    paper_id: str | None = None,
    embedder: TextEmbedder | None = None,
) -> dict[str, Any]:
    """Recreate model-region chunks and rebuild the text index."""
    if not COMBINED_CHUNKS_PATH.exists():
        raise FileNotFoundError(f"缺少 chunks 文件：{COMBINED_CHUNKS_PATH}")

    old_ms = MetadataStore.load(COMBINED_CHUNKS_PATH)
    base_chunks = remove_existing_model_region_chunks(old_ms.get_all())
    grouped = _group_by_paper(
        [
            chunk
            for chunk in base_chunks
            if chunk.get("modality") != "math_model"
        ]
    )
    if paper_id:
        grouped = {paper_id: grouped.get(paper_id, [])}
        if not grouped[paper_id]:
            raise ValueError(f"paper_id={paper_id!r} 不存在或没有可用 chunk")

    generated: list[dict[str, Any]] = []
    per_paper_counts: dict[str, dict[str, int]] = {}
    for pid, paper_chunks in grouped.items():
        region_chunks = build_model_region_chunks(paper_chunks, pid)
        generated.extend(region_chunks)
        per_paper_counts[pid] = summarize_region_counts(region_chunks)
        print(f"[model-region] {pid}: {len(region_chunks)} chunks {per_paper_counts[pid]}")

    if paper_id:
        # Preserve model-region chunks from other papers during single-paper rebuild.
        existing_other_regions = [
            chunk
            for chunk in old_ms.get_all()
            if chunk.get("modality") == "model_region"
            and chunk.get("paper_id") != paper_id
        ]
        all_chunks = base_chunks + existing_other_regions + generated
    else:
        all_chunks = base_chunks + generated

    embedder = embedder or TextEmbedder()
    text_chunks = [chunk for chunk in all_chunks if chunk.get("modality") != "figure"]
    embeddings = _embed_text_chunks(text_chunks, embedder)

    vs = VectorStore(dim=embedder.dim)
    ms = MetadataStore()
    all_ids = ms.add_chunks(all_chunks)
    text_ids = [
        cid
        for cid, chunk in zip(all_ids, all_chunks)
        if chunk.get("modality") != "figure"
    ]
    vs.add(embeddings, text_ids)
    vs.save(TEXT_INDEX_PATH)
    ms.save(COMBINED_CHUNKS_PATH)

    return {
        "paper_id": paper_id or "ALL",
        "region_chunks": len(generated),
        "per_paper_counts": per_paper_counts,
        "chunks_total": len(all_chunks),
        "text_index_total": vs.ntotal,
        "model_region_chunks": sum(
            1 for chunk in all_chunks if chunk.get("modality") == "model_region"
        ),
        "math_model_chunks": sum(
            1 for chunk in all_chunks if chunk.get("modality") == "math_model"
        ),
    }
