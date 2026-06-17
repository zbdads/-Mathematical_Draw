"""文本分块策略集合。

对外接口：
    sentence_boundary_chunk  — Step 1：句子边界分块
    section_aware_chunk      — Step 3：section 边界优先分块
    extract_section_map      — 从全文提取 section 标题索引
    lookup_section           — 按 offset 查找所属 section

内部辅助：_split_sentences, _parse_sections, _merge_short_sections
"""

from __future__ import annotations

import bisect
import re
from typing import Any


def section_aware_chunk(
    markdown_text: str,
    chunk_size: int,
    overlap_sentences: int = 2,
    min_chunk_size: int = 100,
    prepend_header: bool = True,
) -> list[dict[str, Any]]:
    """按 section 边界分块，长 section 内部按句子边界细分。

    分块逻辑：
      1. 按 ## / ### 标题先做粗切，识别各 section 的文本范围。
      2. 过短 section（body 字符数 < min_chunk_size）与前一 section 合并。
      3. 短 section（body ≤ chunk_size）整体作为一个 chunk。
      4. 长 section（body > chunk_size）内部用 sentence_boundary_chunk 细分；
         非首 sub-chunk 若 prepend_header=True，则在文本前注入 section 标题。

    start / end 始终指向 markdown_text 中的原始字符偏移量，不受 prepend_header 影响。

    Args:
        markdown_text:     含 ## / ### 标题的 Markdown 文本。
        chunk_size:        块大小（字符数上限）。
        overlap_sentences: 长 section 内部细分时的句子重叠数。
        min_chunk_size:    过短 section 阈值（字符数），小于此值与前一 section 合并。
        prepend_header:    是否在长 section 的非首 sub-chunk 前注入 section 标题。

    Returns:
        [{"text": str, "start": int, "end": int}, ...]
    """
    if not markdown_text:
        return []

    raw_sections = _parse_sections(markdown_text)
    sections = _merge_short_sections(raw_sections, markdown_text, min_chunk_size)

    chunks: list[dict[str, Any]] = []
    for sec in sections:
        header: str = sec["header"]
        body_start: int = sec["body_start"]
        body_end: int = sec["body_end"]
        body = markdown_text[body_start:body_end]

        if not body.strip():
            continue
        
        if len(body) <= chunk_size:
            chunks.append({"text": body, "start": body_start, "end": body_end})
        else:
            # 太长的 section 使用 sentence overlap 逻辑
            sub_chunks = sentence_boundary_chunk(body, chunk_size, overlap_sentences)
            for i, sc in enumerate(sub_chunks):
                text = sc["text"]
                if prepend_header and header and i > 0:
                    text = f"{header}\n{text}"
                chunks.append({
                    "text": text,
                    "start": body_start + sc["start"],
                    "end": body_start + sc["end"],
                })

    return chunks


def sentence_boundary_chunk(
    text: str,
    chunk_size: int,
    overlap_sentences: int = 2,
) -> list[dict[str, Any]]:
    """按句子边界分块，保证不在句中截断。

    Args:
        text: 待分块的文本。
        chunk_size: 块大小（字符数上限）。
        overlap_sentences: 相邻块重叠的句子数。

    Returns:
        [{"text": str, "start": int, "end": int}, ...]
    """
    if not text:
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[dict[str, Any]] = []
    current: list[tuple[str, int, int]] = []
    current_len = 0

    for sent_text, sent_start, sent_end in sentences:
        sent_len = len(sent_text)

        # 超长单句：fallback 到字符级切分
        if sent_len > chunk_size:
            if current:
                chunks.append({
                    "text": "".join(s[0] for s in current),
                    "start": current[0][1],
                    "end": current[-1][2],
                })
                current = []
                current_len = 0
            for sub_start in range(0, sent_len, chunk_size):
                sub_end = min(sub_start + chunk_size, sent_len)
                sub_text = sent_text[sub_start:sub_end]
                if sub_text.strip():
                    chunks.append({
                        "text": sub_text,
                        "start": sent_start + sub_start,
                        "end": sent_start + sub_end,
                    })
            continue

        # 加入当前句子会超限：先提交，再保留 overlap
        if current and current_len + sent_len > chunk_size:
            chunks.append({
                "text": "".join(s[0] for s in current),
                "start": current[0][1],
                "end": current[-1][2],
            })
            current = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current_len = sum(len(s[0]) for s in current)

        current.append((sent_text, sent_start, sent_end))
        current_len += sent_len

    if current:
        chunks.append({
            "text": "".join(s[0] for s in current),
            "start": current[0][1],
            "end": current[-1][2],
        })

    return chunks


def extract_section_map(text: str) -> list[tuple[int, str]]:
    """从文本中提取 Markdown 标题，返回按 offset 升序的 [(offset, section_title), ...]。

    匹配 parse_pdf() 生成的 ## / ### 开头标题行。
    空列表表示文本不含标题信息（graceful degradation）。
    """
    sections: list[tuple[int, str]] = []
    for m in re.finditer(r"^#{1,3}\s+(.+)$", text, re.MULTILINE):
        sections.append((m.start(), m.group(1).strip()))
    return sections


def lookup_section(section_map: list[tuple[int, str]], offset: int) -> str:
    """查找 offset 之前最近的 section 标题，找不到时返回空串。"""
    if not section_map:
        return ""
    offsets = [s[0] for s in section_map]
    idx = bisect.bisect_right(offsets, offset) - 1
    return section_map[idx][1] if idx >= 0 else ""


def _parse_sections(text: str) -> list[dict[str, Any]]:
    """将 Markdown 文本按 ## / ### 标题切分为 section 列表。

    Returns:
        [{"header": str, "body_start": int, "body_end": int}, ...]

        - header：标题行原文（含 ## 前缀），首段无标题时为空串。
        - body_start / body_end：标题行之后内容在 text 中的字符偏移。
    """
    sections: list[dict[str, Any]] = []
    header_re = re.compile(r"^#{1,3}\s+.+$", re.MULTILINE)
    matches = list(header_re.finditer(text))

    # 首段：第一个 header 之前的内容
    first_header_pos = matches[0].start() if matches else len(text)
    if text[:first_header_pos].strip():
        sections.append({"header": "", "body_start": 0, "body_end": first_header_pos})

    for i, m in enumerate(matches):
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append({
            "header": m.group(0).strip(),
            "body_start": body_start,
            "body_end": body_end,
        })

    return sections


def _merge_short_sections(
    sections: list[dict[str, Any]],
    text: str,
    min_chunk_size: int,
) -> list[dict[str, Any]]:
    """将过短的 section 与前一个合并（扩展 body_end 至短 section 末尾）。

    合并后前一 section 的 body_end 延伸，涵盖短 section 的标题行和正文。
    header 保持前一 section 的标题不变。
    """
    if not sections:
        return []

    result: list[dict[str, Any]] = [dict(sections[0])]
    for sec in sections[1:]:
        body = text[sec["body_start"]:sec["body_end"]]
        if len(body.strip()) < min_chunk_size:
            # 扩展前一 section 的范围以涵盖当前短 section（含其标题行）
            result[-1] = dict(result[-1])
            result[-1]["body_end"] = sec["body_end"]
        else:
            result.append(dict(sec))

    return result


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    """将文本切分为句子，返回 (sentence_text, start, end) 列表。"""
    result: list[tuple[str, int, int]] = []
    prev = 0
    for m in re.finditer(r'[.!?]+\s+|\n{2,}', text):
        end = m.end()
        frag = text[prev:end]
        if frag.strip():
            result.append((frag, prev, end))
        prev = end
    tail = text[prev:]
    if tail.strip():
        result.append((tail, prev, len(text)))
    return result
