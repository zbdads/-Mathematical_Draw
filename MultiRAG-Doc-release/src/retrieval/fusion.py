"""多路召回融合算法

Score Normalization：
    normalize_minmax / normalize_zscore / normalize_softmax / normalize_sigmoid

Fusion 策略：
    fusion_weighted_sum / fusion_rrf / fusion_rsf
    fusion_borda_count / fusion_comb / fusion_dbsf

低阶 API：输入格式统一为 Dict[str, float]（doc_id → score），输出为 List[tuple[str, float]] 降序。
高阶 API：
    reciprocal_rank_fusion  — 将多路检索器结果融合排序
    merge_dual_path_hits    — 按 figure_id 合并文本路与图像路 figure 命中
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────
# Score Normalization
# ─────────────────────────────────────────────

def normalize_minmax(scores: List[float], eps: float = 1e-8) -> List[float]:
    """Min-Max 归一化，映射到 [0, 1]。

    s' = (s - s_min) / (s_max - s_min)
    """
    s_min = min(scores)
    s_max = max(scores)
    return [(s - s_min) / (s_max - s_min + eps) for s in scores]


def normalize_zscore(scores: List[float], eps: float = 1e-8) -> List[float]:
    """Z-score 标准化。

    s' = (s - μ) / σ
    """
    n = len(scores)
    mu = sum(scores) / n
    sigma = math.sqrt(sum((s - mu) ** 2 for s in scores) / n)
    return [(s - mu) / (sigma + eps) for s in scores]


def normalize_softmax(scores: List[float]) -> List[float]:
    """Softmax 归一化，强化 top 结果，输出和为 1。

    s'_i = exp(s_i) / Σ exp(s_j)
    """
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def normalize_sigmoid(scores: List[float]) -> List[float]:
    """Sigmoid 归一化，映射到 (0, 1)，适合 neural retriever 输出。

    s' = 1 / (1 + exp(-s))
    """
    return [1.0 / (1.0 + math.exp(-s)) for s in scores]


# ─────────────────────────────────────────────
# Fusion 策略（低阶 API：Dict[str, float] 输入）
# ─────────────────────────────────────────────

def fusion_weighted_sum(
    retrieval_results: List[Dict[str, float]],
    weights: Optional[List[float]] = None,
    top_k: int = 10,
    normalize: bool = True,
) -> List[tuple[str, float]]:
    """加权求和融合（Score Normalization + Weighted Sum）。

    先对每路分数做 min-max 归一化，再线性组合：
        score = α·s'_text + β·s'_image + ...

    Args:
        retrieval_results: 每路检索结果，格式为 [{doc_id: score, ...}, ...]
        weights:    每路权重，默认均等
        top_k:      返回 top-k 结果
        normalize:  是否先做 min-max 归一化
    """
    n_paths = len(retrieval_results)
    if weights is None:
        weights = [1.0 / n_paths] * n_paths
    assert len(weights) == n_paths, "weights 长度须与 retrieval_results 一致"

    fused: Dict[str, float] = {}
    for path_scores, w in zip(retrieval_results, weights):
        if not path_scores:
            continue
        if normalize:
            ids = list(path_scores.keys())
            norm_vals = normalize_minmax(list(path_scores.values()))
            path_scores = dict(zip(ids, norm_vals))
        for doc_id, score in path_scores.items():
            fused[doc_id] = fused.get(doc_id, 0.0) + w * score

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]


def fusion_rrf(
    retrieval_results: List[Dict[str, float]],
    k: int = 60,
    top_k: int = 10,
) -> List[tuple[str, float]]:
    """Reciprocal Rank Fusion (RRF)。

    完全忽略原始分数，只依赖排名：
        score(d) = Σ_i  1 / (k + rank_i(d))

    k=60 为经典默认值（Cormack et al. 2009）。
    """
    fused: Dict[str, float] = {}
    for path_scores in retrieval_results:
        ranked = sorted(path_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]


def fusion_rsf(
    retrieval_results: List[Dict[str, float]],
    top_k: int = 10,
) -> List[tuple[str, float]]:
    """Relative Score Fusion (RSF)。

    对各路做 min-max 归一化后累加，保留分数的相对分布比例。
    比简单 weighted sum 更稳定，能保留 score 语义信息。
    """
    fused: Dict[str, float] = {}
    for path_scores in retrieval_results:
        if not path_scores:
            continue
        ids = list(path_scores.keys())
        norm_vals = normalize_minmax(list(path_scores.values()))
        for doc_id, norm_score in zip(ids, norm_vals):
            fused[doc_id] = fused.get(doc_id, 0.0) + norm_score

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]


def fusion_borda_count(
    retrieval_results: List[Dict[str, float]],
    top_k: int = 10,
) -> List[tuple[str, float]]:
    """Borda Count rank aggregation。

    按名次赋分再求和：score(d) = Σ_i (N - rank_i(d))。
    思路与 RRF 类似，但对 top 名次的区分度不如 RRF 敏感。
    """
    fused: Dict[str, float] = {}
    for path_scores in retrieval_results:
        ranked = sorted(path_scores.items(), key=lambda x: x[1], reverse=True)
        n = len(ranked)
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + float(n - rank)

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]


def fusion_comb(
    retrieval_results: List[Dict[str, float]],
    method: str = "sum",
    top_k: int = 10,
    normalize: bool = True,
) -> List[tuple[str, float]]:
    """CombSUM / CombMAX / CombMIN 融合。

    先做 min-max 归一化，再对各路分数取聚合：
        CombSUM:  Σ s_i
        CombMAX:  max(s_i)
        CombMIN:  min(s_i)

    Args:
        method: "sum" | "max" | "min"
    """
    assert method in ("sum", "max", "min"), "method 须为 sum / max / min"

    doc_scores: Dict[str, List[float]] = {}
    for path_scores in retrieval_results:
        if not path_scores:
            continue
        if normalize:
            ids = list(path_scores.keys())
            norm_vals = normalize_minmax(list(path_scores.values()))
            path_scores = dict(zip(ids, norm_vals))
        for doc_id, score in path_scores.items():
            doc_scores.setdefault(doc_id, []).append(score)

    agg = {"sum": sum, "max": max, "min": min}[method]
    fused = {doc_id: agg(scores) for doc_id, scores in doc_scores.items()}
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]


def fusion_dbsf(
    retrieval_results: List[Dict[str, float]],
    top_k: int = 10,
) -> List[tuple[str, float]]:
    """Distribution-Based Score Fusion (DBSF)。

    用 z-score 对各路分数做分布对齐，再累加。
    适合 vector + lexical 混合检索（如 BM25 0–30 vs CLIP 0–0.35）。
    """
    fused: Dict[str, float] = {}
    for path_scores in retrieval_results:
        if not path_scores:
            continue
        ids = list(path_scores.keys())
        norm_vals = normalize_zscore(list(path_scores.values()))
        for doc_id, norm_score in zip(ids, norm_vals):
            fused[doc_id] = fused.get(doc_id, 0.0) + norm_score

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ─────────────────────────────────────────────
# 高阶 API：适配检索器返回格式
# ─────────────────────────────────────────────

def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    k: int = 60,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion，将多路检索器结果融合排序。

    Args:
        result_lists: 各路检索器返回的结果列表（格式同 TextRetriever.retrieve()）。
        k: RRF 平滑参数，默认 60。
        top_k: 返回 top-k 结果。

    Returns:
        融合后的 chunk 列表（按 fusion_score 降序），新增 "fusion_score" 字段，
        "rank" 字段重新编号，"score" 保留原路 score 中最高值。
    """
    # 将各路 list[dict] 转为 {chunk_id: score}，同时建立 chunk_id → 原始 dict 的映射
    path_score_maps: list[Dict[str, float]] = []
    chunk_meta: dict[str, dict[str, Any]] = {}

    for result_list in result_lists:
        score_map: Dict[str, float] = {}
        for item in result_list:
            cid = item["chunk_id"]
            score_map[cid] = item["score"]
            # 保留第一次遇到的 metadata（后续路若重复则跳过）
            if cid not in chunk_meta:
                chunk_meta[cid] = {k: v for k, v in item.items() if k not in ("rank", "score")}
        path_score_maps.append(score_map)

    fused_ranked = fusion_rrf(path_score_maps, k=k, top_k=top_k)

    results: list[dict[str, Any]] = []
    for rank, (chunk_id, fusion_score) in enumerate(fused_ranked, start=1):
        meta = chunk_meta.get(chunk_id, {"chunk_id": chunk_id})
        results.append({
            "rank": rank,
            "score": fusion_score,
            "fusion_score": fusion_score,
            **meta,
        })

    return results


def _normalize_hits_by_id(
    hits: list[dict[str, Any]],
    id_key: str,
) -> dict[str, float]:
    """对命中列表按指定 ID 键做 min-max 归一化。"""
    if not hits:
        return {}
    raw_scores = [float(h.get("score", 0.0)) for h in hits]
    s_min, s_max = min(raw_scores), max(raw_scores)
    denom = s_max - s_min or 1e-8
    return {
        str(h[id_key]): (float(h.get("score", 0.0)) - s_min) / denom
        for h in hits
        if h.get(id_key)
    }


def merge_qwenvl_dual_path_hits(
    image_hits_qwenvl: list[dict[str, Any]],
    caption_hits_qwenvl: list[dict[str, Any]],
    beta: float = 0.5,
) -> list[dict[str, Any]]:
    """QwenVL 空间内双路融合：image 路 + caption 路。"""
    image_norm = _normalize_hits_by_id(image_hits_qwenvl, "figure_id")
    caption_norm = _normalize_hits_by_id(caption_hits_qwenvl, "figure_id")

    figure_map: dict[str, dict[str, Any]] = {}
    image_ids: set[str] = set()
    caption_ids: set[str] = set()

    for hit in image_hits_qwenvl:
        fid = str(hit.get("figure_id", ""))
        if not fid:
            continue
        image_ids.add(fid)
        figure_map.setdefault(
            fid,
            {"metadata": hit, "image_score_qwenvl": 0.0, "caption_score_qwenvl": 0.0},
        )
        figure_map[fid]["image_score_qwenvl"] = image_norm.get(fid, 0.0)

    for hit in caption_hits_qwenvl:
        fid = str(hit.get("figure_id", ""))
        if not fid:
            continue
        caption_ids.add(fid)
        figure_map.setdefault(
            fid,
            {"metadata": hit, "image_score_qwenvl": 0.0, "caption_score_qwenvl": 0.0},
        )
        figure_map[fid]["caption_score_qwenvl"] = caption_norm.get(fid, 0.0)

    results: list[dict[str, Any]] = []
    for fid, item in figure_map.items():
        figure_mm_score = (
            beta * item["image_score_qwenvl"]
            + (1 - beta) * item["caption_score_qwenvl"]
        )
        meta = {k: v for k, v in item["metadata"].items() if k not in ("score", "rank")}
        meta.setdefault("chunk_id", fid)
        meta.setdefault("figure_id", fid)
        meta.setdefault("modality", "figure")
        meta.setdefault("content", meta.get("caption_merged") or meta.get("caption", ""))
        meta.setdefault("section", "")

        hit_paths: list[str] = []
        if fid in image_ids:
            hit_paths.append("image_qwenvl")
        if fid in caption_ids:
            hit_paths.append("caption_qwenvl")

        results.append(
            {
                **meta,
                "score": figure_mm_score,
                "figure_mm_score": figure_mm_score,
                "image_score_qwenvl": item["image_score_qwenvl"],
                "caption_score_qwenvl": item["caption_score_qwenvl"],
                "hit_paths": hit_paths,
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def merge_figure_with_caption_hits(
    figure_mm_hits: list[dict[str, Any]],
    caption_hits_bge: list[dict[str, Any]],
    alpha: float = 0.8,
) -> list[dict[str, Any]]:
    """Figure 外层融合：QwenVL 内层结果 + BGE caption 路。"""
    mm_norm = _normalize_hits_by_id(figure_mm_hits, "figure_id")
    bge_norm = _normalize_hits_by_id(caption_hits_bge, "chunk_id")

    figure_map: dict[str, dict[str, Any]] = {}
    mm_ids: set[str] = set()
    bge_ids: set[str] = set()

    for hit in figure_mm_hits:
        fid = str(hit.get("figure_id", hit.get("chunk_id", "")))
        if not fid:
            continue
        mm_ids.add(fid)
        figure_map.setdefault(fid, {"metadata": hit, "mm_score_norm": 0.0, "bge_score_norm": 0.0})
        figure_map[fid]["mm_score_norm"] = mm_norm.get(fid, 0.0)

    for hit in caption_hits_bge:
        fid = str(hit.get("chunk_id", hit.get("figure_id", "")))
        if not fid:
            continue
        bge_ids.add(fid)
        figure_map.setdefault(fid, {"metadata": hit, "mm_score_norm": 0.0, "bge_score_norm": 0.0})
        figure_map[fid]["bge_score_norm"] = bge_norm.get(fid, 0.0)

    results: list[dict[str, Any]] = []
    for fid, item in figure_map.items():
        final_figure_score = (
            alpha * item["mm_score_norm"]
            + (1 - alpha) * item["bge_score_norm"]
        )
        meta = {k: v for k, v in item["metadata"].items() if k not in ("score", "rank")}
        meta.setdefault("chunk_id", fid)
        meta.setdefault("figure_id", fid)
        meta.setdefault("modality", "figure")
        meta.setdefault("content", meta.get("caption_merged") or meta.get("caption", ""))
        meta.setdefault("section", "")

        hit_paths: list[str] = list(meta.get("hit_paths", []))
        if fid in bge_ids:
            hit_paths.append("caption_bge")
        if fid in mm_ids:
            hit_paths.append("figure_mm")
        hit_paths = list(dict.fromkeys(hit_paths))

        results.append(
            {
                **meta,
                "score": final_figure_score,
                "final_figure_score": final_figure_score,
                "figure_mm_score_norm": item["mm_score_norm"],
                "caption_score_bge_norm": item["bge_score_norm"],
                "hit_paths": hit_paths,
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def merge_dual_path_hits(
    text_figure_hits: list[dict[str, Any]],
    image_hits: list[dict[str, Any]],
    alpha: float = 0.8,
) -> list[dict[str, Any]]:
    """按 figure_id 合并文本路与图像路的 figure 命中，融合双路分数。

    设计要点：
        - 文本路（caption_merged 经 SentenceTransformer 编码）权重 = 1 - alpha
        - 图像路（CLIP image encoder）权重 = alpha
        - 两路都命中同一 figure 时合并为一条结果并增强分数
        - 只命中一路时仅获得该路的加权分数

    Args:
        text_figure_hits: TextRetriever 返回的 modality=="figure" 子集；
                          使用 chunk_id 字段作为 figure_id 键。
        image_hits:       ImageRetriever.retrieve() 返回的列表；
                          使用 figure_id 字段作为键。
        alpha:            图像路权重（0–1），默认 0.8（image 路主导，ablation 实验最优值）。

    Returns:
        融合后的 figure 结果列表（按融合分数降序），每条新增：
            "score"       融合分数
            "hit_paths"   命中路径列表，如 ["text", "image"]
            "chunk_id"    等于 figure_id（与文本结果格式对齐）
    """
    # ── 分别做 min-max 归一化，并记录各路实际命中的 figure_id 集合 ──────────
    def _norm(hits: list[dict[str, Any]], id_key: str) -> dict[str, float]:
        if not hits:
            return {}
        raw = [h["score"] for h in hits]
        s_min, s_max = min(raw), max(raw)
        denom = s_max - s_min or 1e-8
        return {h[id_key]: (h["score"] - s_min) / denom for h in hits}

    text_norm = _norm(text_figure_hits, "chunk_id")
    image_norm = _norm(image_hits, "figure_id")
    text_ids: set[str] = {h["chunk_id"] for h in text_figure_hits}
    image_ids: set[str] = {h["figure_id"] for h in image_hits}

    # ── 按 figure_id 聚合元数据 ────────────────────────────────────────────
    figure_map: dict[str, dict[str, Any]] = {}

    for hit in text_figure_hits:
        fid = hit["chunk_id"]
        if fid not in figure_map:
            figure_map[fid] = {
                "metadata": hit,
                "text_score": 0.0,
                "image_score": 0.0,
            }
        figure_map[fid]["text_score"] = text_norm[fid]

    for hit in image_hits:
        fid = hit["figure_id"]
        if fid not in figure_map:
            figure_map[fid] = {
                "metadata": hit,
                "text_score": 0.0,
                "image_score": 0.0,
            }
        figure_map[fid]["image_score"] = image_norm[fid]

    # ── 计算融合分数并组装结果 ─────────────────────────────────────────────
    results: list[dict[str, Any]] = []
    for fid, data in figure_map.items():
        fused_score = alpha * data["image_score"] + (1 - alpha) * data["text_score"]
        # hit_paths 按实际是否在源列表中判断（不依赖归一化后的分数值）
        hit_paths: list[str] = []
        if fid in text_ids:
            hit_paths.append("text")
        if fid in image_ids:
            hit_paths.append("image")

        meta = {k: v for k, v in data["metadata"].items() if k not in ("score", "rank")}
        meta.setdefault("chunk_id", fid)
        meta.setdefault("figure_id", fid)
        meta.setdefault("content", meta.get("caption_merged") or meta.get("caption", ""))
        meta.setdefault("section", "")
        results.append({
            **meta,
            "score": fused_score,
            "hit_paths": hit_paths,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results
