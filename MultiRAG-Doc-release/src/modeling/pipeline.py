"""Pipeline for building and indexing optimization model cards."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from src.index.metadata_store import MetadataStore
from src.modeling.extractor import extract_model_card_from_chunks
from src.modeling.store import (
    COMBINED_CHUNKS_PATH,
    MODEL_CARD_DIR,
    load_all_model_cards,
    load_model_card,
    rebuild_text_index_with_model_cards,
    save_model_card,
)


def _group_chunks_by_paper(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        if chunk.get("modality") == "math_model":
            continue
        paper_id = str(chunk.get("paper_id", "")).strip()
        if paper_id:
            grouped[paper_id].append(chunk)
    return dict(grouped)


def build_model_cards_from_chunks(
    *,
    chunks_path: Path | None = None,
    model_card_dir: Path | None = None,
    paper_id: str | None = None,
    max_chunks: int = 18,
    field_level: bool = True,
    rebuild_index: bool = True,
) -> dict[str, Any]:
    """Extract model cards from the current chunk store and optionally index them."""
    source_path = chunks_path or COMBINED_CHUNKS_PATH
    target_dir = model_card_dir or MODEL_CARD_DIR
    ms = MetadataStore.load(source_path)
    grouped = _group_chunks_by_paper(ms.get_all())

    if paper_id:
        grouped = {paper_id: grouped.get(paper_id, [])}
        if not grouped[paper_id]:
            raise ValueError(f"paper_id={paper_id!r} 不存在或没有可用 chunk")

    cards = []
    saved_paths = []
    for pid, chunks in grouped.items():
        print(f"[model-card] extracting {pid} from {len(chunks)} chunks")
        target_path = target_dir / f"{pid}.json"
        card = extract_model_card_from_chunks(
            pid,
            chunks,
            max_chunks=max_chunks,
            field_level=field_level,
        )
        if (
            card.confidence <= 0.0
            and target_path.exists()
            and not any(
                getattr(card, field)
                for field in ("sets", "parameters", "variables", "objective", "constraints")
            )
        ):
            existing = load_model_card(target_path)
            if existing.confidence > 0.0 or any(
                getattr(existing, field)
                for field in ("sets", "parameters", "variables", "objective", "constraints")
            ):
                print(f"[model-card] keeping existing {target_path} after failed extraction")
                card = existing
                path = target_path
            else:
                path = save_model_card(card, target_path)
        else:
            path = save_model_card(card, target_path)
        cards.append(card)
        saved_paths.append(str(path))
        print(
            f"[model-card] saved {path} "
            f"confidence={card.confidence:.2f} "
            f"vars={len(card.variables)} obj={len(card.objective)} cons={len(card.constraints)}"
        )

    index_stats = {}
    if rebuild_index:
        cards_for_index = load_all_model_cards(target_dir)
        index_stats = rebuild_text_index_with_model_cards(cards_for_index)
        print(f"[model-card] indexed stats: {index_stats}")

    return {
        "cards": [card.to_dict() for card in cards],
        "paths": saved_paths,
        "index_stats": index_stats,
    }
