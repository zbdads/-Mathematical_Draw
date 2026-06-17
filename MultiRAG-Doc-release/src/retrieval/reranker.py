"""Cross-encoder 精排模块。

用法：
    reranker = Reranker()
    ranked = reranker.rerank(query, candidates, top_k=5)

流程：
    1. 将 (query, chunk["content"]) 拼成 pair 列表。
    2. CrossEncoder.predict(pairs) 批量推理，返回 logit 分数。
    3. 按分数降序重排，截取 top_k，并将 rerank_score 写回 chunk。
"""

from __future__ import annotations

from typing import Any


_DEFAULT_MODEL = "BAAI/bge-reranker-base"


class Reranker:
    """基于 sentence-transformers CrossEncoder 的重排序器。

    Args:
        model_name: HuggingFace 模型名称或本地路径。
        device:     推理设备，None 则自动选 CUDA / CPU。
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import CrossEncoder
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = CrossEncoder(model_name, device=device)
        print(f"  [Reranker] model={model_name}  device={device}")

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """对候选 chunk 做精排。

        Args:
            query:      检索问题字符串。
            candidates: chunk 列表，每条须含 "content" 字段。
            top_k:      精排后保留条数。

        Returns:
            按 rerank_score 降序排列的 chunk 列表（每条追加 rerank_score 字段）。
        """
        if not candidates:
            return []

        pairs = [(query, c["content"]) for c in candidates]
        scores: list[float] = self._model.predict(pairs).tolist()

        ranked = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for score, chunk in ranked[:top_k]:
            results.append({**chunk, "rerank_score": round(float(score), 6)})
        return results
