"""PeerQA 召回评测脚本。

前置条件：
    1. 已运行 peerqa_ingest.py，生成 database/peerqa_index/ 和 database/peerqa_chunks/
    2. 已运行 peerqa_gt_mapper.py，生成 database/testset/peerqa_eval_testset.json

用法（通过 CLI）：
    python -m src.cli eval-retrieval \\
        --peerqa-testset database/testset/peerqa_eval_testset.json \\
        --index-dir database/peerqa_index \\
        --chunks database/peerqa_chunks/all_chunks.json \\
        --top-k 20 --config-tag peerqa_baseline
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from src.evaluation.recall_metrics import precision_at_k, recall_at_k, reciprocal_rank
from src.index.metadata_store import MetadataStore
from src.index.vector_store import VectorStore
from src.ingestion.text_embedder import TextEmbedder
from src.retrieval.text_retriever import TextRetriever

_K_VALUES = [5, 10, 20]

_DEFAULT_PEERQA_TESTSET = Path("database/testset/peerqa_eval_testset.json")
_DEFAULT_INDEX_DIR = Path("database/peerqa_index")
_DEFAULT_CHUNKS = Path("database/peerqa_chunks/all_chunks.json")
_DEFAULT_OUTPUT_DIR = Path("database/eval_results")


# ── Sanity check ──────────────────────────────────────────────────────────────

def _build_chunk_id_set(ms: MetadataStore) -> set[str]:
    return {chunk["id"] for chunk in ms.get_all() if "id" in chunk}


def validate_gt_ids(testset: list[dict], chunk_id_set: set[str]) -> None:
    """检查 testset 中所有 GT IDs 是否在当前索引中存在，缺失则直接报错。"""
    missing = []
    for q in testset:
        for gt_id in q.get("ground_truth_ids", []):
            if gt_id not in chunk_id_set:
                missing.append((q.get("id", "?"), gt_id))
    if missing:
        raise ValueError(
            f"GT ID 缺失 {len(missing)} 个，请先重新运行 peerqa_gt_mapper.py。"
            f"\n缺失样例：{missing[:5]}"
        )


# ── 指标计算与聚合 ────────────────────────────────────────────────────────────

def _compute_record_metrics(retrieved_ids: list[str], gt_ids: list[str]) -> dict[str, float]:
    record: dict[str, float] = {
        "mrr": reciprocal_rank(retrieved_ids, gt_ids),
        "precision@5": precision_at_k(retrieved_ids, gt_ids, 5),
    }
    for k in _K_VALUES:
        record[f"recall@{k}"] = recall_at_k(retrieved_ids, gt_ids, k)
    return record


def _metrics_for_group(records: list[dict]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        base = {f"recall@{k}": 0.0 for k in _K_VALUES}
        base.update({"mrr": 0.0, "precision@5": 0.0, "count": 0})
        return base
    result: dict[str, Any] = {"count": n}
    for k in _K_VALUES:
        result[f"recall@{k}"] = round(sum(r[f"recall@{k}"] for r in records) / n, 4)
    result["mrr"] = round(sum(r["mrr"] for r in records) / n, 4)
    result["precision@5"] = round(sum(r["precision@5"] for r in records) / n, 4)
    return result


def _aggregate_peerqa_results(records: list[dict]) -> dict[str, Any]:
    mapping_groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        mapping_groups[r.get("mapping_status", "unknown")].append(r)

    return {
        "overall": _metrics_for_group(records),
        "by_mapping_status": {
            s: _metrics_for_group(g) for s, g in sorted(mapping_groups.items())
        },
    }


# ── 评测 ──────────────────────────────────────────────────────────────────────

def _to_safe_paper_id(original_id: str) -> str:
    return "peerqa_" + original_id.replace("/", "__")


def eval_peerqa(
    text_retriever: TextRetriever,
    testset: list[dict],
    top_k: int = 20,
) -> dict[str, Any]:
    """对 peerqa_eval_testset 跑文本召回评测，返回聚合结果。"""
    records = []
    for q in testset:
        retrieved = text_retriever.retrieve(q["query"], top_k=top_k)
        retrieved_ids = [r["chunk_id"] for r in retrieved]
        gt_ids = q["ground_truth_ids"]
        record: dict[str, Any] = {
            "id": q["id"],
            "mapping_status": q.get("mapping_status", "unknown"),
        }
        record.update(_compute_record_metrics(retrieved_ids, gt_ids))
        records.append(record)
    return _aggregate_peerqa_results(records)


def eval_peerqa_intra_paper(
    text_retriever: TextRetriever,
    testset: list[dict],
    ms: MetadataStore,
    top_k: int = 20,
) -> dict[str, Any]:
    """论文内部召回评测：每条 query 只在来源论文的 chunks 内检索。

    从全局索引检索 fetch_n 个结果（fetch_n ≥ 最大单篇 chunk 数），
    再按 paper_id 过滤、保留原始排名后取 top_k。
    """
    chunks_by_paper: dict[str, set[str]] = {}
    for chunk in ms.get_all():
        pid = chunk.get("paper_id", "")
        if pid not in chunks_by_paper:
            chunks_by_paper[pid] = set()
        chunks_by_paper[pid].add(chunk["id"])

    max_paper_chunks = max((len(ids) for ids in chunks_by_paper.values()), default=top_k)
    fetch_n = max(max_paper_chunks + 50, top_k * 10)
    print(f"[eval] intra-paper: max chunks/paper={max_paper_chunks}, fetch_n={fetch_n}")

    records = []
    for q in testset:
        safe_pid = _to_safe_paper_id(q.get("original_paper_id", ""))
        paper_chunk_ids = chunks_by_paper.get(safe_pid, set())

        retrieved_all = text_retriever.retrieve(q["query"], top_k=fetch_n)
        retrieved_ids = [
            r["chunk_id"] for r in retrieved_all if r["chunk_id"] in paper_chunk_ids
        ][:top_k]

        gt_ids = q["ground_truth_ids"]
        record: dict[str, Any] = {
            "id": q["id"],
            "mapping_status": q.get("mapping_status", "unknown"),
        }
        record.update(_compute_record_metrics(retrieved_ids, gt_ids))
        records.append(record)
    return _aggregate_peerqa_results(records)


# ── 主入口 ────────────────────────────────────────────────────────────────────

def run_retrieval_eval(
    peerqa_testset_path: Path | None = None,
    index_dir: Path | None = None,
    chunks_path: Path | None = None,
    top_k: int = 20,
    config_tag: str = "peerqa_baseline",
    output_dir: Path | None = None,
    intra_paper: bool = False,
) -> dict[str, Any]:
    """PeerQA 召回评测主入口：加载资源 → sanity check → 评测 → 保存结果。

    Args:
        peerqa_testset_path: peerqa_eval_testset.json 路径。
        index_dir:    peerqa FAISS 索引目录（含 text_index.faiss）。
        chunks_path:  peerqa all_chunks.json 路径。
        top_k:        召回数量，默认 20。
        config_tag:   结果标识，写入输出文件名与 JSON。
        output_dir:   结果目录，默认 database/eval_results/。
        intra_paper:  若为 True，每条 query 只在来源论文的 chunks 内检索。

    Returns:
        评测结果 dict（同时持久化到 output_dir）。
    """
    peerqa_testset_path = peerqa_testset_path or _DEFAULT_PEERQA_TESTSET
    index_dir = index_dir or _DEFAULT_INDEX_DIR
    chunks_path = chunks_path or _DEFAULT_CHUNKS
    output_dir = output_dir or _DEFAULT_OUTPUT_DIR

    index_path = index_dir / "text_index.faiss"

    for p, label in [
        (peerqa_testset_path, "peerqa_eval_testset"),
        (index_path, "text_index.faiss"),
        (chunks_path, "all_chunks.json"),
    ]:
        if not p.exists():
            raise FileNotFoundError(
                f"{label} 不存在：{p}\n"
                "请先运行 peerqa_ingest.py 和 peerqa_gt_mapper.py。"
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] 加载 PeerQA 索引：{index_path}")
    vs = VectorStore.load(index_path)
    ms = MetadataStore.load(chunks_path)
    embedder = TextEmbedder()
    retriever = TextRetriever(vs, ms, embedder)
    chunk_id_set = _build_chunk_id_set(ms)

    print(f"[eval] 加载 PeerQA 测试集：{peerqa_testset_path}")
    with open(peerqa_testset_path, encoding="utf-8") as f:
        testset: list[dict] = json.load(f)

    # testset 由 peerqa_gt_mapper 生成时已过滤掉 failed 条目，此处仅保留 answerable=True
    testset = [q for q in testset if q.get("answerable", True)]
    print(f"[eval] query 数：{len(testset)}，运行 sanity check...")

    validate_gt_ids(testset, chunk_id_set)
    mode_label = "intra-paper" if intra_paper else "global"
    print(f"[eval] sanity check 通过，开始 PeerQA 召回评测（{mode_label}）...")

    if intra_paper:
        eval_result = eval_peerqa_intra_paper(retriever, testset, ms, top_k=top_k)
    else:
        eval_result = eval_peerqa(retriever, testset, top_k=top_k)
    ov = eval_result["overall"]
    print(
        f"[eval] PeerQA 召回完成  "
        f"recall@5={ov['recall@5']:.3f}  "
        f"recall@10={ov['recall@10']:.3f}  "
        f"recall@20={ov['recall@20']:.3f}  "
        f"mrr={ov['mrr']:.3f}  (n={ov['count']})"
    )

    result: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_tag": config_tag,
        "dataset": "peerqa",
        "top_k": top_k,
        "eval_size": len(testset),
        "retrieval_mode": "intra_paper" if intra_paper else "global",
        "text_retrieval": eval_result,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{config_tag}_{ts}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[eval] 结果已保存：{output_path}")

    return result
