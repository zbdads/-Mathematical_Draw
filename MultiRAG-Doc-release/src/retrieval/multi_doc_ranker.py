"""多文档排序与聚合模块（doc-level + chunk-level）。

流程：
    1. doc-level aggregation：将同一论文的多个 chunk 分数聚合为论文级分数。
    2. doc-level 粗筛：保留 top_papers 篇论文。
    3. chunk-level 精排：在保留论文内按 chunk 分数重新排序，tie-breaking 用 chunk_id 保证稳定性。

支持聚合策略：max / mean / sum（默认 max，适合多 chunk 中有一个高度相关的场景）。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal


AggStrategy = Literal["max", "mean", "sum"]


def aggregate_by_paper(
    chunks: list[dict[str, Any]],
    strategy: AggStrategy = "max",
) -> list[dict[str, Any]]:
    """将 chunk 列表按论文聚合，返回论文级排序结果。

    Args:
        chunks: 检索器返回的 chunk 列表，每条须含 paper_id、score、chunk_id 字段。
        strategy: 聚合策略 "max" | "mean" | "sum"。

    Returns:
        按 agg_score 降序排列的论文列表，每条格式：
        {
            "paper_id": str,
            "agg_score": float,
            "chunks": list[dict],   # 该论文下按 score 降序排列的 chunk
        }
    """
    paper_scores: dict[str, list[float]] = defaultdict(list)
    paper_chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for c in chunks:
        pid = c["paper_id"]
        paper_scores[pid].append(c["score"])
        paper_chunks[pid].append(c)

    results: list[dict[str, Any]] = []
    for pid, scores in paper_scores.items():
        if strategy == "max":
            agg = max(scores)
        elif strategy == "sum":
            agg = sum(scores)
        else:  # mean
            agg = sum(scores) / len(scores)

        # chunk-level 内部精排：score 降序，score 相同时按 chunk_id 字典序（稳定 tie-breaking）
        sorted_chunks = sorted(
            paper_chunks[pid],
            key=lambda c: (-c["score"], c.get("chunk_id", "")),
        )
        results.append({"paper_id": pid, "agg_score": agg, "chunks": sorted_chunks})

    return sorted(results, key=lambda x: (-x["agg_score"], x["paper_id"]))


def rank_multi_doc(
    chunks: list[dict[str, Any]],
    strategy: AggStrategy = "max",
    top_papers: int | None = None,
    top_chunks_per_paper: int | None = None,
) -> list[dict[str, Any]]:
    """多文档双层排序主接口。

    先做 doc-level 聚合粗筛，再在每篇论文内做 chunk-level 精排，
    最终展平为带 rank 字段的 chunk 列表。

    Args:
        chunks: 检索器返回的 chunk 列表。
        strategy: doc-level 聚合策略 "max" | "mean" | "sum"。
        top_papers: 保留前 N 篇论文，None 表示全部保留。
        top_chunks_per_paper: 每篇论文最多保留 N 个 chunk，None 表示全部保留。

    Returns:
        展平后的 chunk 列表，新增字段：
            "doc_rank"   - 论文级排名（从 1 开始）
            "rank"       - 全局 chunk 排名（从 1 开始）
            "agg_score"  - 该论文的聚合分数
    """
    paper_results = aggregate_by_paper(chunks, strategy=strategy)

    if top_papers is not None:
        paper_results = paper_results[:top_papers]

    flat: list[dict[str, Any]] = []
    global_rank = 1
    for doc_rank, paper in enumerate(paper_results, start=1):
        selected_chunks = paper["chunks"]
        if top_chunks_per_paper is not None:
            selected_chunks = selected_chunks[:top_chunks_per_paper]
        for chunk in selected_chunks:
            flat.append(
                {
                    **chunk,
                    "rank": global_rank,
                    "doc_rank": doc_rank,
                    "agg_score": paper["agg_score"],
                }
            )
            global_rank += 1

    return flat
