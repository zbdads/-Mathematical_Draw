"""文本分块编排模块：跨页拼接 + section 感知分块。

多模态 chunk 组装见 multimodal_chunker.py，文本分割算法见 text_chunker.py。
"""

from __future__ import annotations

import bisect
from typing import Any

from src.ingestion.chunk_id import text_chunk_id
from src.ingestion.text_chunker import (
    extract_section_map,
    lookup_section,
    section_aware_chunk,
)


def build_text_chunks(
    pages: list[dict[str, Any]],
    paper_id: str,
    chunk_size: int = 600,
    overlap_sentences: int = 2,
    min_chunk_size: int = 200,
    prepend_header: bool = True,
) -> list[dict[str, Any]]:
    """将 parse_pdf() 返回的页列表转换为文本 chunk 记录列表。

    跨页拼接全文后，按 section 边界优先分块（section_aware_chunk），
    长 section 内部按句子边界细分。

    Args:
        pages:             parse_pdf() 返回值。
        paper_id:          论文标识符（如 "RAG_2020"）。
        chunk_size:        文本块大小（字符数上限）。
        overlap_sentences: 相邻文本块重叠句子数。
        min_chunk_size:    过短 section 的合并阈值（字符数）。
        prepend_header:    非首 sub-chunk 是否注入 section 标题。

    Returns:
        文本 chunk 记录列表。
    """
    result: list[dict[str, Any]] = []

    full_text, page_offsets = _concat_pages(pages)
    section_map = extract_section_map(full_text)

    for idx, c in enumerate(
        section_aware_chunk(full_text, chunk_size, overlap_sentences, min_chunk_size, prepend_header)
    ):
        start_page = _lookup_page(page_offsets, c["start"])
        end_page = _lookup_page(page_offsets, max(c["start"], c["end"] - 1))
        pages_list = (
            [start_page]
            if start_page == end_page
            else list(range(start_page, end_page + 1))
        )
        section = lookup_section(section_map, c["start"])
        result.append(
            {
                "id": text_chunk_id(paper_id, idx),
                "paper_id": paper_id,
                "modality": "text",
                "content": c["text"],
                "caption": "",
                "page": pages_list,
                "section": section,
                "start": c["start"],
                "end": c["end"],
                "embedding_text": [],
                "embedding_image": [],
            }
        )

    return result


def _concat_pages(
    pages: list[dict[str, Any]],
) -> tuple[str, list[tuple[int, int]]]:
    """将 parse_pdf() 返回的页列表拼接为全文，保留页码映射。

    Returns:
        full_text: 所有页文本以 "\\n\\n" 拼接的全文字符串。
        page_offsets: [(start_offset, page_no), ...] 按 start_offset 升序。
    """
    parts: list[str] = []
    page_offsets: list[tuple[int, int]] = []
    offset = 0
    for d in pages:
        text = d.get("text", "")
        if not text:
            continue
        page_offsets.append((offset, int(d["page"])))
        parts.append(text)
        offset += len(text) + 2  # +2 为 "\n\n" 分隔符长度
    full_text = "\n\n".join(parts)
    return full_text, page_offsets


def _lookup_page(page_offsets: list[tuple[int, int]], offset: int) -> int:
    """在 page_offsets 中二分查找 offset 所在页码。"""
    if not page_offsets:
        return 1
    offsets = [p[0] for p in page_offsets]
    idx = bisect.bisect_right(offsets, offset) - 1
    return page_offsets[max(idx, 0)][1]
