"""Persistence and indexing for optimization model cards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import CFG
from src.index.metadata_store import MetadataStore
from src.index.vector_store import VectorStore
from src.ingestion.text_embedder import TextEmbedder
from src.modeling.schema import FIVE_ELEMENT_FIELDS, ModelCard

MODEL_CARD_DIR = CFG.paths.root / "database" / "model_cards"
COMBINED_CHUNKS_PATH = CFG.paths.chunks_dir / "all_chunks.json"
TEXT_INDEX_PATH = CFG.paths.index_dir / "text_index.faiss"


def save_model_card(card: ModelCard, path: Path | None = None) -> Path:
    """Save a model card to JSON and return the path."""
    target = path or MODEL_CARD_DIR / f"{card.paper_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(card.to_dict(), f, ensure_ascii=False, indent=2)
    return target


def load_model_card(path: Path) -> ModelCard:
    with open(path, encoding="utf-8") as f:
        return ModelCard.from_dict(json.load(f))


def load_all_model_cards(model_card_dir: Path | None = None) -> list[ModelCard]:
    root = model_card_dir or MODEL_CARD_DIR
    if not root.exists():
        return []
    return [load_model_card(path) for path in sorted(root.glob("*.json"))]


def card_to_markdown(card: ModelCard) -> str:
    """Render a model card as retrieval-friendly text."""
    lines = [
        f"Optimization model card for paper {card.paper_id}.",
        f"Title: {card.title or 'Not specified'}.",
        f"Problem type: {card.problem_type or 'Not specified'}.",
        f"Application domain: {card.application_domain or 'Not specified'}.",
        f"Model name: {card.model_name or 'Not specified'}.",
        "",
    ]
    labels = {
        "sets": "Sets",
        "parameters": "Parameters",
        "variables": "Decision Variables",
        "objective": "Objective Function",
        "constraints": "Constraints",
        "assumptions": "Assumptions",
        "algorithm": "Solution Algorithm",
    }
    for field, label in labels.items():
        values = getattr(card, field)
        lines.append(f"## {label}")
        if values:
            lines.extend(f"- {value}" for value in values)
        else:
            lines.append("- Not found in extracted evidence.")
        lines.append("")
    if card.evidence_refs:
        lines.append("## Evidence References")
        for ref in card.evidence_refs:
            quote = f": {ref.quote}" if ref.quote else ""
            lines.append(f"- [{ref.chunk_id}] {ref.role}, page {ref.page}{quote}")
        lines.append("")
    if card.warnings:
        lines.append("## Extraction Warnings")
        lines.extend(f"- {warning}" for warning in card.warnings)
    return "\n".join(lines).strip()


def card_to_chunks(card: ModelCard) -> list[dict[str, Any]]:
    """Convert a model card to one summary chunk plus field-specific chunks."""
    chunks: list[dict[str, Any]] = []
    refs = [ref.__dict__ for ref in card.evidence_refs]
    base = {
        "paper_id": card.paper_id,
        "modality": "math_model",
        "page": [],
        "section": "LLMOPT-style model card",
        "start": 0,
        "end": 0,
        "caption": "",
        "embedding_text": [],
        "embedding_image": [],
        "model_card": card.to_dict(),
        "source_chunk_ids": card.source_chunk_ids,
        "evidence_refs": refs,
    }
    summary = card_to_markdown(card)
    chunks.append(
        {
            **base,
            "id": f"{card.paper_id}_model_card",
            "chunk_type": "model_card",
            "content": summary,
            "end": len(summary),
        }
    )

    for field in FIVE_ELEMENT_FIELDS:
        values = getattr(card, field)
        if not values:
            continue
        label = {
            "sets": "Sets",
            "parameters": "Parameters",
            "variables": "Decision Variables",
            "objective": "Objective Function",
            "constraints": "Constraints",
        }[field]
        content = "\n".join(
            [
                f"{label} for paper {card.paper_id}.",
                f"Problem type: {card.problem_type or 'Not specified'}.",
                *(f"- {value}" for value in values),
                "",
                "Source evidence: "
                + ", ".join(f"[{cid}]" for cid in card.source_chunk_ids[:8]),
            ]
        ).strip()
        chunks.append(
            {
                **base,
                "id": f"{card.paper_id}_model_{field}",
                "chunk_type": field,
                "content": content,
                "end": len(content),
            }
        )
    return chunks


def remove_existing_model_chunks(ms: MetadataStore) -> list[dict[str, Any]]:
    """Return chunks excluding generated model-card chunks.

    Model-region chunks are extracted from the original paper evidence and should
    remain searchable alongside model cards.
    """
    return [
        chunk
        for chunk in ms.get_all()
        if chunk.get("modality") != "math_model"
        and not str(chunk.get("id", "")).endswith("_model_card")
    ]


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


def rebuild_text_index_with_model_cards(
    cards: list[ModelCard],
    *,
    embedder: TextEmbedder | None = None,
) -> dict[str, int]:
    """Append model-card chunks to metadata and rebuild the text FAISS index."""
    if not COMBINED_CHUNKS_PATH.exists():
        raise FileNotFoundError(f"缺少 chunks 文件：{COMBINED_CHUNKS_PATH}")

    old_ms = MetadataStore.load(COMBINED_CHUNKS_PATH)
    chunks = remove_existing_model_chunks(old_ms)
    for card in cards:
        chunks.extend(card_to_chunks(card))

    embedder = embedder or TextEmbedder()
    text_chunks = [chunk for chunk in chunks if chunk.get("modality") != "figure"]
    embeddings = _embed_text_chunks(text_chunks, embedder)

    vs = VectorStore(dim=embedder.dim)
    ms = MetadataStore()
    all_ids = ms.add_chunks(chunks)
    text_ids = [
        cid
        for cid, chunk in zip(all_ids, chunks)
        if chunk.get("modality") != "figure"
    ]
    vs.add(embeddings, text_ids)
    vs.save(TEXT_INDEX_PATH)
    ms.save(COMBINED_CHUNKS_PATH)

    return {
        "cards": len(cards),
        "chunks_total": len(chunks),
        "text_index_total": vs.ntotal,
        "model_chunks": sum(1 for chunk in chunks if chunk.get("modality") == "math_model"),
    }
