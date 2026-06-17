"""Chunk ID 生成模块：集中管理所有 modality 的 chunk ID 格式。"""

from __future__ import annotations


def text_chunk_id(paper_id: str, idx: int) -> str:
    return f"{paper_id}_c{idx}"


def figure_chunk_id(paper_id: str, page: int, idx: int) -> str:
    return f"{paper_id}_fig_{page}_{idx}"


def table_chunk_id(paper_id: str, idx: int) -> str:
    return f"{paper_id}_t{idx}"


def equation_chunk_id(paper_id: str, idx: int) -> str:
    return f"{paper_id}_e{idx}"


def model_region_chunk_id(paper_id: str, idx: int) -> str:
    return f"{paper_id}_mr{idx}"


def assign_figure_ids(figures: list[dict], paper_id: str) -> None:
    """为 figure 列表原地分配 figure_id，格式与 ImageRetriever 一致。

    按 page 分组计数，保证同页内 idx 递增。
    """
    page_counters: dict[int, int] = {}
    for fig in figures:
        page = int(fig.get("page", 0))
        idx = page_counters.get(page, 0)
        page_counters[page] = idx + 1
        fig["figure_id"] = figure_chunk_id(paper_id, page, idx)
