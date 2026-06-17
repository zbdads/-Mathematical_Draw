"""多模态 chunk 组装模块：figure / table / equation 分块 + 总入口。

文本分块编排见 chunker.py，文本分割算法见 text_chunker.py。
"""

from __future__ import annotations

from typing import Any

from src.ingestion.chunk_id import (
    equation_chunk_id,
    figure_chunk_id,
    table_chunk_id,
)
from src.ingestion.chunker import build_text_chunks
from src.ingestion.model_region_chunker import build_model_region_chunks


def build_figure_chunks(
    figures: list[dict[str, Any]],
    paper_id: str,
) -> list[dict[str, Any]]:
    """将 figure 列表转为 chunk 记录。"""
    result: list[dict[str, Any]] = []
    for idx, fig in enumerate(figures):
        page = int(fig["page"])
        caption_original = fig.get("caption", "")
        content = fig.get("caption_merged") or caption_original
        figure_id = fig.get("figure_id", figure_chunk_id(paper_id, 0, idx))
        result.append(
            {
                "id": figure_id,
                "figure_id": figure_id,
                "paper_id": paper_id,
                "modality": "figure",
                "content": content,
                "caption": caption_original,
                "caption_merged": content,
                "page": [page],
                "section": "",
                "start": 0,
                "end": len(content),
                "bbox": fig.get("bbox"),
                "embedding_text": [],
                "embedding_image": [],
            }
        )
    return result


def build_table_chunks(
    tables: list[dict[str, Any]],
    paper_id: str,
) -> list[dict[str, Any]]:
    """将 table 列表转为 chunk 记录。"""
    result: list[dict[str, Any]] = []
    for idx, table in enumerate(tables):
        page = int(table["page"])
        caption = table.get("caption", "")
        markdown = table.get("markdown", "")
        result.append(
            {
                "id": table_chunk_id(paper_id, idx),
                "paper_id": paper_id,
                "modality": "table",
                "content": markdown,
                "caption": caption,
                "page": [page],
                "section": "",
                "start": 0,
                "end": len(markdown),
                "embedding_text": [],
                "embedding_image": [],
            }
        )
    return result


def build_equation_chunks(
    equations: list[dict[str, Any]],
    paper_id: str,
) -> list[dict[str, Any]]:
    """将 equation 列表转为 chunk 记录。"""
    result: list[dict[str, Any]] = []
    for idx, equation in enumerate(equations):
        page = int(equation["page"])
        content = equation.get("content", "")
        result.append(
            {
                "id": equation_chunk_id(paper_id, idx),
                "paper_id": paper_id,
                "modality": "equation",
                "content": content,
                "caption": "",
                "page": [page],
                "section": "",
                "start": 0,
                "end": len(content),
                "bbox": equation.get("bbox"),
                "embedding_text": [],
                "embedding_image": [],
            }
        )
    return result


def build_chunks(
    pages: list[dict[str, Any]],
    paper_id: str,
    chunk_size: int = 600,
    overlap_sentences: int = 2,
    min_chunk_size: int = 200,
    prepend_header: bool = True,
    figures: list[dict[str, Any]] | None = None,
    tables: list[dict[str, Any]] | None = None,
    equations: list[dict[str, Any]] | None = None,
    include_model_regions: bool = True,
) -> list[dict[str, Any]]:
    """将解析结果转换为统一的 chunk 记录列表。

    文本来自 parse_pdf()，图片/表格/公式来自 parse_pdf_multimodal()（可选）。
    所有 modality 统一进同一列表，供后续向量化和入库使用。

    Args:
        pages:             parse_pdf() 返回值。
        paper_id:          论文标识符（如 "RAG_2020"）。
        chunk_size:        文本块大小（字符数上限）。
        overlap_sentences: 相邻文本块重叠句子数。
        min_chunk_size:    过短 section 的合并阈值（字符数）。
        prepend_header:    非首 sub-chunk 是否注入 section 标题。
        figures:           parse_pdf_figures() 或 parse_pdf_multimodal()[0]，None 表示跳过。
        tables:            parse_pdf_tables() 或 parse_pdf_multimodal()[1]，None 表示跳过。
        equations:         parse_pdf_multimodal()[2]，None 表示跳过。
        include_model_regions: 是否额外生成数学模型区域 chunk。

    Returns:
        完整 chunk 记录列表。
    """
    result = build_text_chunks(
        pages, paper_id, chunk_size, overlap_sentences, min_chunk_size, prepend_header
    )

    if figures:
        result.extend(build_figure_chunks(figures, paper_id))

    if tables:
        result.extend(build_table_chunks(tables, paper_id))

    if equations:
        result.extend(build_equation_chunks(equations, paper_id))

    if include_model_regions:
        result.extend(build_model_region_chunks(result, paper_id))

    return result
