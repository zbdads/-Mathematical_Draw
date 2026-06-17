"""图像 Embedding 模块：基于 Qwen3-VL-Embedding-8B 的图像向量化。"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
import requests
from dotenv import load_dotenv

try:
    import torch
    import torch.nn.functional as F
    from transformers.modeling_outputs import ModelOutput
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLModel,
        Qwen3VLPreTrainedModel,
    )
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
except ImportError:  # pragma: no cover - API mode does not need local VL deps
    torch = None
    F = None
    ModelOutput = object
    Qwen3VLConfig = object
    Qwen3VLModel = None
    Qwen3VLPreTrainedModel = object
    Qwen3VLProcessor = None

from src.config import CFG
from src.ingestion.image_api_utils import pil_image_to_data_url

load_dotenv(CFG.paths.root / ".env")


@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    """Qwen3-VL embedding head output."""

    last_hidden_state: torch.FloatTensor | None = None
    attention_mask: torch.Tensor | None = None


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    """Minimal embedding wrapper aligned with the official Qwen3-VL-Embedding script."""

    config_class = Qwen3VLConfig
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False

    def __init__(self, config: Qwen3VLConfig):
        if Qwen3VLModel is None:
            raise ImportError("本地 Qwen3-VL embedding 依赖未安装，请使用 embedder.image_mode: api")
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> Qwen3VLForEmbeddingOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


class ImageEmbedder:
    """基于 Qwen3-VL-Embedding-8B 的多模态 Embedder，支持图像批量编码与文本查询编码。

    Qwen3-VL-Embedding 统一向量空间：图像 encoder 与文本 encoder 共享同一空间（4096 维），
    因此文本 query 可以直接与图像 embedding 做 cosine / inner product 比较。

    与 CLIP 的区别：
        - 支持在图像 embedding 时融合 caption 文本，使 embedding 同时携带视觉与语义信息。
        - 向量维度更高（4096 vs 512），语义表达能力更强。
        - 采用 LLM 的 last-token pooling 提取 embedding，而非 projection head。

    为什么图像索引不能与文本索引混用？
        两者维度不同（4096 vs 1024）且 embedding 语义空间不同，混用会导致相似度无意义。
    """

    # 推荐的 instruction 前缀（按 embedding 模型惯例）
    _IMAGE_INSTRUCTION = "Represent this image for retrieval."
    _TEXT_INSTRUCTION = "Represent the query for image retrieval:"
    _DEFAULT_IMAGE_BATCH_SIZE = 2

    def __init__(self, model_name: str | None = None) -> None:
        self._mode = CFG.embedder.image_mode.lower().strip()
        self._dim = CFG.embedder.image_embedding_dim
        self._api_model_name = model_name or CFG.embedder.image_model_name or "qwen3-vl-embedding"
        self._api_url = CFG.embedder.image_api_url
        self._api_key = os.environ.get(CFG.embedder.image_api_key_env) or os.environ.get("EMBEDDING_API_KEY") or os.environ.get("LLM_API_KEY")
        self._lock = threading.Lock()

        if self._mode == "api":
            if not self._api_url:
                raise EnvironmentError("请在 config.yml 中设置 embedder.image_api_url")
            if not self._api_key:
                raise EnvironmentError(
                    f"请在项目根目录 .env 文件中设置 {CFG.embedder.image_api_key_env}"
                )
            print(
                "  [ImageEmbedder-api] "
                f"model={self._api_model_name}, dim={self._dim}"
            )
            return

        if torch is None or Qwen3VLProcessor is None:
            raise ImportError(
                "本地 Qwen3-VL embedding 依赖未安装；"
                "若要调用 API，请设置 config.yml: embedder.image_mode: api"
            )
        name = model_name or CFG.embedder.image_model_name or "Qwen/Qwen3-VL-Embedding-8B"
        # 优先使用本地模型快照
        model_path = CFG.paths.models_dir / name
        source = str(model_path) if model_path.exists() else name
        print(f"  [ImageEmbedder] model={source}")

        self._model = Qwen3VLForEmbedding.from_pretrained(
            source,
            dtype=torch.float16,
            device_map={"": "cuda:0"} if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
        )
        self._processor = Qwen3VLProcessor.from_pretrained(source, padding_side="right")
        self._model.eval()
        # device_map="auto" 时取第一个参数所在设备，用于输入张量迁移
        self._device = next(self._model.parameters()).device
        self._validate_embedding_dim()

    @property
    def dim(self) -> int:
        return self._dim

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.size == 0:
            return embeddings.astype("float32")
        if embeddings.ndim != 2:
            raise ValueError(f"API 返回 embedding 不是 2D 矩阵：shape={embeddings.shape}")
        if embeddings.shape[1] != self._dim:
            raise ValueError(
                "image_embedding_dim 配置与实际多模态 API 返回维度不一致："
                f"config.yml={self._dim}, actual={embeddings.shape[1]}。"
            )
        faiss.normalize_L2(embeddings)
        return embeddings.astype("float32")

    def _call_embedding_api(self, content: list[dict[str, Any]]) -> np.ndarray:
        payload = {
            "model": self._api_model_name,
            "input": {"contents": content},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        response = None
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                with self._lock:
                    response = requests.post(
                        self._api_url,
                        headers=headers,
                        json=payload,
                        timeout=120,
                    )
                if response.status_code < 500:
                    break
            except requests.RequestException as exc:
                last_exc = exc
            time.sleep(2 * attempt)

        if response is None:
            raise RuntimeError(f"DashScope 多模态 embedding 请求失败：{last_exc}") from last_exc
        if response.status_code >= 400:
            raise RuntimeError(
                "DashScope 多模态 embedding 调用失败："
                f"HTTP {response.status_code} {response.text[:800]}"
            )
        data = response.json()
        embedding = self._extract_embedding(data)
        return self._normalize_embeddings(np.array([embedding], dtype="float32"))

    @staticmethod
    def _extract_embedding(data: dict[str, Any]) -> list[float]:
        """Extract an embedding vector from DashScope multimodal response variants."""
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

    def _validate_embedding_dim(self) -> None:
        """校验配置维度与模型实际输出维度一致，避免索引维度漂移。"""
        config = self._model.config
        actual_dim = getattr(config.text_config, "hidden_size", None)
        vision_dim = getattr(config.vision_config, "out_hidden_size", None)
        expected_dim = CFG.embedder.image_embedding_dim

        if actual_dim is None:
            raise ValueError("无法从 Qwen3-VL 配置中解析 image embedding 维度")
        if vision_dim is not None and vision_dim != actual_dim:
            raise ValueError(
                f"Qwen3-VL 配置异常：text hidden_size={actual_dim}, "
                f"vision out_hidden_size={vision_dim}"
            )
        if actual_dim != self._dim:
            raise ValueError(
                f"image_embedding_dim 配置错误：config.yml={self._dim}, "
                f"模型实际输出维度={actual_dim}"
            )

    @staticmethod
    def _last_token_pool(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """提取每条序列最后一个有效 token 的向量。"""
        flipped_mask = attention_mask.flip(dims=[1])
        last_one_pos = flipped_mask.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_pos - 1
        row = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
        return last_hidden_state[row, col]

    def _build_conversation(
        self,
        *,
        text: str | None = None,
        image: Any | None = None,
        instruction: str,
    ) -> list[dict[str, Any]]:
        """Build a single text/image conversation compatible with Qwen3VLProcessor."""
        content: list[dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": text or "NULL"})
        return [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content},
        ]

    def _prepare_batch_inputs(
        self,
        conversations: list[list[dict[str, Any]]],
        images: list[Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        texts = self._processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self._processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {k: v.to(self._device) for k, v in inputs.items()}

    def encode_images(
        self,
        images: list,
        captions: list[str] | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """批量编码 PIL 图像，返回 L2 归一化后的 (n, dim) float32 矩阵。

        当提供 captions 时，embedding 会同时融合图像视觉内容与文本描述，
        可以使用 caption_merged（原始 caption + QwenVL 生成描述）来丰富语义表达。

        Args:
            images:    PIL 图像列表。
            captions:  可选，与 images 等长的 caption 列表（建议使用 caption_merged）。
                       若为 None 或某条为空，则仅用视觉内容 embedding。
            batch_size: 每批编码数量，默认取 CFG.embedder.batch_size // 8（8B 模型较大）。

        Returns:
            (n, dim) float32 矩阵，L2 归一化，可直接用于 IndexFlatIP cosine 检索。
        """
        if self._mode == "api":
            if not images:
                return np.zeros((0, self._dim), dtype="float32")
            all_vecs: list[np.ndarray] = []
            total = len(images)
            for idx, (img, cap) in enumerate(zip(images, captions or [None] * len(images)), start=1):
                print(f"  [ImageEmbedder-api] image embedding {idx}/{total}")
                content: list[dict[str, Any]] = [
                    {"image": pil_image_to_data_url(img)},
                ]
                if cap:
                    content.append({"text": f"Caption: {cap}"})
                all_vecs.append(self._call_embedding_api(content))
            return np.vstack(all_vecs)

        # 8B 模型显存占用大，默认批次远小于文本 embedder
        bs = batch_size or self._DEFAULT_IMAGE_BATCH_SIZE
        all_vecs: list[np.ndarray] = []

        for i in range(0, len(images), bs):
            batch_imgs = images[i : i + bs]
            batch_caps = captions[i : i + bs] if captions else [None] * len(batch_imgs)

            conversations = []
            for img, cap in zip(batch_imgs, batch_caps):
                query_text = f"Caption: {cap}" if cap else None
                conversations.append(
                    self._build_conversation(
                        text=query_text,
                        image=img,
                        instruction=self._IMAGE_INSTRUCTION,
                    )
                )
            inputs = self._prepare_batch_inputs(conversations, images=batch_imgs)

            with torch.no_grad():
                outputs = self._model(**inputs)

            embs = self._last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            all_vecs.append(embs.float().cpu().numpy().astype("float32"))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result = np.vstack(all_vecs)
        faiss.normalize_L2(result)
        return result

    def encode_texts(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """批量编码文本列表（用于 caption 向量化），返回 L2 归一化后的 (n, dim) 矩阵。"""
        if self._mode == "api":
            if not texts:
                return np.zeros((0, self._dim), dtype="float32")
            total = len(texts)
            all_vecs = []
            for idx, text in enumerate(texts, start=1):
                print(f"  [ImageEmbedder-api] text embedding {idx}/{total}")
                all_vecs.append(self._call_embedding_api([{"text": text or "NULL"}]))
            return np.vstack(all_vecs)

        bs = batch_size or CFG.embedder.batch_size
        all_vecs: list[np.ndarray] = []

        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            conversations = [
                self._build_conversation(
                    text=t,
                    instruction=self._TEXT_INSTRUCTION,
                )
                for t in batch
            ]
            inputs = self._prepare_batch_inputs(conversations)

            with torch.no_grad():
                outputs = self._model(**inputs)

            embs = self._last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            all_vecs.append(embs.float().cpu().numpy().astype("float32"))

        result = np.vstack(all_vecs)
        if F is None:
            raise ImportError("本地 Qwen3-VL embedding 依赖 torch 未安装")
        return F.normalize(torch.from_numpy(result), p=2, dim=-1).cpu().numpy().astype("float32")

    def encode_text_query(self, query: str) -> np.ndarray:
        """将文本 query 映射到图像共享空间，返回 (1, dim) 向量。"""
        return self.encode_texts([query])

    def release(self) -> None:
        """释放模型，回收 GPU 显存。"""
        if self._mode == "api":
            print("[ImageEmbedder-api] released")
            return
        import gc

        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[ImageEmbedder] 模型已释放")
