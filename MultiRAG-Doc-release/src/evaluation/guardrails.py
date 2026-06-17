"""Pre/post-generation guardrail logic."""

from __future__ import annotations

from src.config import CFG


def pre_generation_guardrail(
    question: str,
    evidence: list[dict[str, object]],
) -> tuple[bool, str]:
    """检查证据是否足够支撑生成。

    Returns:
        (ok, reason) — ok=False 时 reason 描述拒答原因。
    """
    cfg = CFG.guardrails

    if len(evidence) < cfg.min_results:
        return False, f"retrieved results too few: {len(evidence)} < {cfg.min_results}"

    # 用 FAISS 余弦相似度衡量绝对相关性；融合后 score 变为 RRF 分，需优先取 faiss_score
    best_score = max(float(r.get("faiss_score", r.get("score", -1.0))) for r in evidence)
    if best_score < cfg.min_top1_score:
        return False, f"top1 score too low: {best_score:.4f} < {cfg.min_top1_score:.4f}"

    total_chars = sum(len(str(item.get("content", ""))) for item in evidence)
    if total_chars < cfg.min_total_chars:
        return False, f"evidence too short: {total_chars} < {cfg.min_total_chars}"


    return True, ""
