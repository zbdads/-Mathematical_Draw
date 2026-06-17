"""检索评估指标

Recall@k / Precision@k / Reciprocal Rank / MRR

输入统一为 List[str]（doc_id 列表，按排名从高到低）。
"""

from __future__ import annotations

from typing import List


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Recall@k：top-k 中命中相关 doc 的比例（相对于全部相关 doc）。

    Recall@k = |retrieved[:k] ∩ relevant| / |relevant|
    """
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant_set)
    return hits / len(relevant_set)


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Precision@k：top-k 中相关 doc 占 k 的比例。

    Precision@k = |retrieved[:k] ∩ relevant| / k
    """
    if k == 0:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant_set)
    return hits / k


def reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """单个 query 的 Reciprocal Rank (RR)。

    RR = 1 / rank of first relevant doc（未命中则 0）
    """
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    retrieved_list: List[List[str]],
    relevant_list: List[List[str]],
) -> float:
    """Mean Reciprocal Rank (MRR)，跨多个 query 取均值。

    MRR = (1/N) Σ_i  1 / rank_i
    """
    assert len(retrieved_list) == len(relevant_list), "query 数量须一致"
    if not retrieved_list:
        return 0.0
    rr_sum = sum(
        reciprocal_rank(ret, rel)
        for ret, rel in zip(retrieved_list, relevant_list)
    )
    return rr_sum / len(retrieved_list)
