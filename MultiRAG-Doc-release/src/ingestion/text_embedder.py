"""文本 Embedding 模块：基于 FlagEmbedding BGEM3FlagModel 的文本向量化。"""

from __future__ import annotations

import gc
import hashlib
import os
import re
import threading
import time

import faiss
import numpy as np
import requests
try:
    import torch
except ImportError:  # pragma: no cover - lightweight API repro may omit torch
    torch = None

from src.config import CFG
from src.openai_compat import build_openai_client


def _select_torch_device() -> str:
    """优先使用 CUDA，其次回退到 CPU。"""
    return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


def _is_hash_mode(mode: str) -> bool:
    return mode in {"hash", "demo", "local_hash"}


def _is_qwen_vl_embedding_model(model_name: str) -> bool:
    return model_name.lower().strip() in {
        "qwen2.5-vl-embedding",
        "qwen3-vl-embedding",
        "qwen3-vl-embedding-8b",
        "multimodal-embedding-v1",
        "tongyi-embedding-vision-flash",
        "tongyi-embedding-vision-plus",
    }


def _clip_embedding_text(text: str, *, limit: int = 1900) -> str:
    """Keep API embedding inputs inside conservative provider limits."""
    value = text or "NULL"
    if len(value) <= limit:
        return value
    head = value[: int(limit * 0.7)]
    tail = value[-int(limit * 0.3) :]
    return f"{head}\n...\n{tail}"


def _batch_by_text_budget(texts: list[str], *, max_items: int, max_chars: int) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for text in texts:
        size = len(text)
        if current and (len(current) >= max_items or current_chars + size > max_chars):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(text)
        current_chars += size
    if current:
        batches.append(current)
    return batches


def _tokenize_for_hashing(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text.lower())
    bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    return tokens + bigrams


def _hash_token(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    raw = int.from_bytes(digest, "little", signed=False)
    return raw % dim, 1.0 if (raw >> 63) == 0 else -1.0


class TextEmbedder:
    """基于 BGE-M3 的文本 Embedder（dense retrieval）。"""

    def __init__(self, model_name: str | None = None) -> None:
        self._lock = threading.Lock()
        self._mode = CFG.embedder.mode.lower().strip()
        self._dim = CFG.embedder.embedding_dim
        self._client = None
        self._model = None

        if _is_hash_mode(self._mode):
            print(f"  [embedder-hash] dim={self._dim}（本地 smoke-test 检索，非正式语义 embedding）")
            return

        if self._mode == "api":
            api_model_name = CFG.embedder.api_model_name or (model_name or CFG.embedder.text_model_name)
            api_base_url = CFG.embedder.api_base_url or CFG.generator.base_url
            self._api_model_name = api_model_name
            self._api_dimensions = CFG.embedder.api_dimensions
            self._api_base_url = api_base_url
            self._api_key = (
                os.environ.get(CFG.embedder.api_key_env)
                or os.environ.get("EMBEDDING_API_KEY")
                or os.environ.get("LLM_API_KEY")
            )
            self._api_url = CFG.embedder.image_api_url
            if _is_qwen_vl_embedding_model(api_model_name):
                if not self._api_key:
                    raise EnvironmentError(
                        f"请在项目根目录 .env 文件中设置 {CFG.embedder.api_key_env}"
                    )
                if not self._api_url:
                    raise EnvironmentError("请在 config.yml 中设置 embedder.image_api_url")
                print(
                    "  [embedder-api-multimodal] "
                    f"model={api_model_name}, url={self._api_url}, dim={self._dim}"
                )
            else:
                self._client = build_openai_client(
                    base_url=api_base_url,
                    api_key_env=CFG.embedder.api_key_env,
                    fallback_api_key_env="LLM_API_KEY",
                )
                print(
                    "  [embedder-api] "
                    f"model={api_model_name}, base_url={api_base_url}, dim={self._dim}"
                )
            return

        from FlagEmbedding import BGEM3FlagModel

        name = model_name or CFG.embedder.text_model_name
        model_path = CFG.paths.models_dir / name
        use_fp16 = torch is not None and torch.cuda.is_available()
        source = str(model_path) if model_path.exists() else name
        device = _select_torch_device()
        self._model = BGEM3FlagModel(source, use_fp16=use_fp16, devices=[device])
        print(f"  [embedder] model={source}, device={device}, fp16={use_fp16}")

    @property
    def dim(self) -> int:
        return self._dim

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.size == 0:
            return embeddings.astype("float32")
        if embeddings.shape[1] != self._dim:
            raise ValueError(
                "embedding_dim 配置与实际 API 返回维度不一致："
                f"config.yml={self._dim}, actual={embeddings.shape[1]}。"
                "请将 config.yml 中 embedder.embedding_dim 改为 API 模型的真实维度后重试。"
            )
        faiss.normalize_L2(embeddings)
        return embeddings

    def _encode_dense(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Encode texts and return normalized dense vectors."""
        if not texts:
            return np.zeros((0, self._dim), dtype="float32")

        if _is_hash_mode(self._mode):
            return self._encode_dense_hash(texts)

        if self._mode == "api":
            return self._encode_dense_api(texts, batch_size)

        with self._lock:
            output = self._model.encode(
                texts,
                batch_size=batch_size,
                max_length=512,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        embeddings = np.array(output["dense_vecs"], dtype="float32")
        return self._normalize_embeddings(embeddings)

    def _encode_dense_api(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Encode texts via an OpenAI-compatible embeddings API."""
        if _is_qwen_vl_embedding_model(self._api_model_name):
            return self._encode_dense_qwenvl_api(texts)

        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = [_clip_embedding_text(text) for text in texts[i : i + batch_size]]
            request_kwargs: dict = {
                "model": self._api_model_name,
                "input": batch,
                "encoding_format": "float",
            }
            if self._api_dimensions is not None:
                request_kwargs["dimensions"] = self._api_dimensions
            response = None
            last_exc: Exception | None = None
            for attempt in range(1, 4):
                try:
                    response = self._client.embeddings.create(**request_kwargs)
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt >= 3:
                        raise
                    time.sleep(1.5 * attempt)
            if response is None:
                raise last_exc or RuntimeError("embedding API request failed")
            ordered = sorted(response.data, key=lambda item: item.index)
            embeddings = np.array([item.embedding for item in ordered], dtype="float32")
            all_embeddings.append(self._normalize_embeddings(embeddings))
        return np.vstack(all_embeddings)

    def _encode_dense_qwenvl_api(self, texts: list[str]) -> np.ndarray:
        """Encode texts via DashScope multimodal embedding API."""
        all_embeddings: list[np.ndarray] = []
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        total = len(texts)
        done = 0
        batches = _batch_by_text_budget(
            [_clip_embedding_text(text) for text in texts],
            max_items=max(1, CFG.embedder.batch_size),
            max_chars=9000,
        )
        for batch in batches:
            payload = {
                "model": self._api_model_name,
                "input": {"contents": [{"text": text or "NULL"} for text in batch]},
            }
            response = None
            last_exc: Exception | None = None
            for attempt in range(1, 7):
                try:
                    with self._lock:
                        response = requests.post(
                            self._api_url,
                            headers=headers,
                            json=payload,
                            timeout=120,
                        )
                    if response.status_code < 500 and response.status_code != 429:
                        break
                except requests.RequestException as exc:
                    last_exc = exc
                wait_seconds = 15 * attempt if response is not None and response.status_code == 429 else 8 * attempt
                time.sleep(wait_seconds)
            if response is None:
                raise RuntimeError(f"DashScope 多模态 embedding 请求失败：{last_exc}") from last_exc
            if response.status_code >= 400:
                raise RuntimeError(
                    "DashScope 多模态 embedding 调用失败："
                    f"HTTP {response.status_code} {response.text[:800]}"
            )
            if total > 1:
                done += len(batch)
                print(f"  [embedder-api-multimodal] text embedding {done}/{total}")
            all_embeddings.append(
                self._normalize_embeddings(
                    np.array(self._extract_qwenvl_embeddings(response.json()), dtype="float32")
                )
            )
            time.sleep(1.0)
        return np.vstack(all_embeddings)

    @classmethod
    def _extract_qwenvl_embeddings(cls, data: dict) -> list[list[float]]:
        output = data.get("output", data)
        embeddings = output.get("embeddings") if isinstance(output, dict) else None
        if isinstance(embeddings, list) and embeddings:
            result: list[list[float]] = []
            for item in embeddings:
                if isinstance(item, dict):
                    emb = item.get("embedding") or item.get("vector")
                    if emb is not None:
                        result.append(emb)
                elif isinstance(item, list):
                    result.append(item)
            if result:
                return result

        data_list = data.get("data")
        if isinstance(data_list, list) and data_list:
            result = [
                item.get("embedding")
                for item in data_list
                if isinstance(item, dict) and item.get("embedding") is not None
            ]
            if result:
                return result

        return [cls._extract_qwenvl_embedding(data)]

    @staticmethod
    def _extract_qwenvl_embedding(data: dict) -> list[float]:
        output = data.get("output", data)
        embeddings = output.get("embeddings") if isinstance(output, dict) else None
        if isinstance(embeddings, list) and embeddings:
            first = embeddings[0]
            if isinstance(first, dict):
                emb = first.get("embedding") or first.get("vector")
                if emb is not None:
                    return emb
            if isinstance(first, list):
                return first

        data_list = data.get("data")
        if isinstance(data_list, list) and data_list:
            first = data_list[0]
            if isinstance(first, dict):
                emb = first.get("embedding")
                if emb is not None:
                    return emb

        raise RuntimeError(f"无法解析多模态 embedding API 返回：{str(data)[:800]}")

    def _encode_dense_hash(self, texts: list[str]) -> np.ndarray:
        """Deterministic local fallback for dependency and UI smoke tests."""
        embeddings = np.zeros((len(texts), self._dim), dtype="float32")
        for row, text in enumerate(texts):
            for token in _tokenize_for_hashing(text):
                col, sign = _hash_token(token, self._dim)
                embeddings[row, col] += sign
        return self._normalize_embeddings(embeddings)

    def encode(self, texts: list[str]) -> np.ndarray:
        """批量编码文本，返回 L2 归一化后的 (n, dim) float32 矩阵。"""
        return self._encode_dense(texts, batch_size=CFG.embedder.batch_size)

    def encode_query(self, query: str) -> np.ndarray:
        """编码单条查询，返回 (1, dim) 归一化向量。"""
        return self._encode_dense([query], batch_size=1)

    def release(self) -> None:
        """释放模型，回收 GPU 显存。"""
        if self._mode == "api":
            self._client = None
            return
        if _is_hash_mode(self._mode):
            return
        del self._model
        self._model = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[TextEmbedder] 模型已释放")
