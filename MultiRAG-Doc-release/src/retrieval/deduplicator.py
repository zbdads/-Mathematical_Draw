"""去重与冲突标注模块。

去重：基于 embedding cosine 相似度，相似度超过阈值的 chunk 视为重复，保留 score 最高的版本。
冲突标注：对同一 paper_id 内容不重复但分数相近的 chunk 不做合并，仅在输出中增加
          conflict_note 字段，标明同主题下存在多篇来源及其 paper_id + page 信息。

冲突标注触发条件（保守策略）：
    - 同一问题下有 ≥2 篇不同论文的 chunk 进入结果。
    - 各论文 agg_score 差距在 conflict_margin 以内（分数相近，说明均有较强相关性，
      出现分歧的可能性较高）。
"""

from __future__ import annotations

from typing import Any

import numpy as np


def deduplicate_chunks(
    chunks: list[dict[str, Any]],
    threshold: float = 0.95,
) -> list[dict[str, Any]]:
    """基于 embedding cosine 相似度去重。

    保留每个相似簇中 score 最高的 chunk（非最先出现），确保质量最优。
    若 chunk 无 "embedding" 字段，则跳过相似度计算，直接返回原列表。

    Args:
        chunks: 含 "embedding"（list[float]）和 "score"（float）字段的 chunk 列表。
        threshold: cosine 相似度阈值，超过此值视为重复（默认 0.95）。

    Returns:
        去重后的 chunk 列表（保持原有字段，移除 "embedding" 以减小输出体积）。
    """
    if not chunks:
        return chunks

    if "embedding" not in chunks[0]:
        return chunks

    # 按 score 降序，优先保留高分 chunk
    sorted_chunks = sorted(chunks, key=lambda c: -c["score"])
    embs = np.array([c["embedding"] for c in sorted_chunks], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embs = embs / norms

    kept_indices: list[int] = []
    dropped: set[int] = set()

    for i in range(len(sorted_chunks)):
        if i in dropped:
            continue
        kept_indices.append(i)
        sims = embs[i] @ embs.T  # shape: (n,)
        for j in range(i + 1, len(sorted_chunks)):
            if j not in dropped and sims[j] > threshold:
                dropped.add(j)

    result = []
    for idx in kept_indices:
        chunk = {k: v for k, v in sorted_chunks[idx].items() if k != "embedding"}
        result.append(chunk)

    return result


def annotate_conflicts(
    chunks: list[dict[str, Any]],
    conflict_margin: float = 0.05,
) -> list[dict[str, Any]]:
    """对跨论文结果进行冲突标注（不裁判，只标出分歧）。

    当同一问题下有 ≥2 篇不同论文且它们的 agg_score 差距在 conflict_margin 以内时，
    为每个相关 chunk 添加 conflict_note 字段，列出所有竞争来源。

    Args:
        chunks: 经过 rank_multi_doc 处理后的 chunk 列表（应含 paper_id、agg_score）。
        conflict_margin: agg_score 差值阈值，差值 ≤ conflict_margin 的论文视为存在竞争。

    Returns:
        原 chunk 列表（in-place 修改 conflict_note 字段后返回）。
    """
    if not chunks:
        return chunks

    # 收集各论文的 agg_score
    paper_agg: dict[str, float] = {}
    for c in chunks:
        pid = c.get("paper_id", "")
        if pid and pid not in paper_agg:
            paper_agg[pid] = c.get("agg_score", c.get("score", 0.0))

    if len(paper_agg) < 2:
        return chunks

    top_score = max(paper_agg.values())
    competing_papers = {
        pid: score
        for pid, score in paper_agg.items()
        if top_score - score <= conflict_margin
    }

    if len(competing_papers) < 2:
        return chunks

    # 构建 conflict_note 说明
    sources = sorted(competing_papers.items(), key=lambda x: -x[1])
    note_parts = [f"{pid}(score={s:.4f})" for pid, s in sources]
    conflict_note = "Conflicting sources: " + ", ".join(note_parts)

    for chunk in chunks:
        if chunk.get("paper_id", "") in competing_papers:
            chunk["conflict_note"] = conflict_note

    return chunks


def deduplicate_and_annotate(
    chunks: list[dict[str, Any]],
    dedup_threshold: float = 0.95,
    conflict_margin: float = 0.05,
) -> list[dict[str, Any]]:
    """去重 + 冲突标注一体化接口。

    先去重，再做冲突标注。

    Args:
        chunks: 含 "embedding"、"score"、"paper_id"、"agg_score" 的 chunk 列表。
        dedup_threshold: cosine 相似度去重阈值。
        conflict_margin: agg_score 差值冲突判定阈值。

    Returns:
        处理后的 chunk 列表。
    """
    deduped = deduplicate_chunks(chunks, threshold=dedup_threshold)
    return annotate_conflicts(deduped, conflict_margin=conflict_margin)
