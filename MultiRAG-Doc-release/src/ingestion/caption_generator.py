"""Caption 生成器：为论文 figure 生成检索导向的文字描述。

在 ingest 流程中生成 ``caption_generated``，并与原始 caption 合并为
``caption_merged``。合并后的文本用于 figure chunk 的文本向量化，也可作为
图像向量化时的补充文本条件。

接口约定：
    - generate(image, prompt)  →  str
    - generate_batch(images, prompts)  →  list[str]

语言范围：英文论文 figure。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image as PILImage


class CaptionGenerator(ABC):
    """Caption 推理接口，支持逐张及批量生成 figure caption。

    子类须实现 generate()；generate_batch() 提供默认串行实现。
    """

    @abstractmethod
    def generate(self, image: "PILImage.Image", prompt: str = "") -> str:
        """为单张 figure 生成描述。

        Args:
            image:  PIL.Image，原始图像（RGB）。
            prompt: 引导生成的文本提示（如原始 caption 或问题）。

        Returns:
            生成的描述字符串；推理失败时返回空字符串。
        """

    def generate_batch(
        self,
        images: list["PILImage.Image"],
        prompts: list[str] | None = None,
        batch_size: int = 4,
    ) -> list[str]:
        """批量生成 caption，返回与 images 等长的描述列表。

        Raises:
            ValueError: ``batch_size`` 非正数，或 ``prompts`` 与 ``images`` 数量不一致。
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be >= 1")
        if prompts is None:
            prompts = [""] * len(images)
        elif len(prompts) != len(images):
            raise ValueError("prompts and images must have the same length")
        results: list[str] = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            batch_prompts = prompts[i : i + batch_size]
            for img, prompt in zip(batch_imgs, batch_prompts, strict=True):
                results.append(self.generate(img, prompt))
        return results


_PROMPT_WITH_CAPTION = """Generate a compact retrieval caption for an academic figure.

Requirements:
- One line, max 80 words
- Preserve key technical terms from original caption
- Identify figure type
- Describe as indexed content (not explanation)
- Include components, relations, operations
- Include main model/method name if visible
- Add synonyms and related terms
- Add 1–2 short query-style phrases
- Avoid generic words
- Do not invent unseen details

Format:
<type>; <content>; <components/relations>; <keywords + synonyms + query phrases>

Original caption:
"{caption}"

"""

_PROMPT_WITHOUT_CAPTION = """Generate a retrieval-oriented caption for an academic figure.

Rules:
- One line, max 80 words
- If informative:
  <type>; <content>; <components>; <keywords + synonyms + query phrases>
- If clearly non-informative:
  irrelevant; non-informative; none; skip
- Prefer recall over filtering
- Include method/model name if visible
- No hallucination

Types:
architecture, pipeline, mechanism, comparison, curve, chart, table, qualitative result, quantitative result
"""


class QwenVLCaptionGenerator(CaptionGenerator):
    """基于 Qwen3-VL-8B-Instruct 的 caption 生成器。

    使用 Qwen3VLForConditionalGeneration。

    生成策略：beam search + repetition_penalty，抑制重复退化。
    模型延迟加载（首次调用 generate 时加载），支持多篇论文复用同一实例。

    prompt 策略：
        - 有原始 caption → _PROMPT_WITH_CAPTION（注入原始 caption）
        - 无原始 caption → _PROMPT_WITHOUT_CAPTION（纯视觉描述）
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        adapter_path: Path | None = None,
    ) -> None:
        """
        Args:
            model_dir:  本地模型快照目录（优先使用）；为 None 时从 HuggingFace 在线加载。
            model_name: HuggingFace repo id，model_dir 不存在时作为回退。
            adapter_path: LoRA / PEFT adapter 目录；存在时在 base model 上叠加加载。
        """
        self.model_dir = Path(model_dir) if model_dir else None
        self.model_name = model_name
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _model_path(self) -> str:
        """返回实际加载路径（本地目录优先）。"""
        if self.model_dir and self.model_dir.exists():
            return str(self.model_dir)
        return self.model_name

    def _load(self) -> None:
        """延迟加载模型与 processor（仅在首次 generate 时调用）。"""
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        path = self._model_path()
        print(f"[QwenVLCaptionGenerator] 加载模型：{path}")
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            path,
            dtype=torch.float16,
            device_map="auto",
        )
        if self.adapter_path and self.adapter_path.exists():
            print(f"[QwenVLCaptionGenerator] 加载 LoRA adapter：{self.adapter_path}")
            self._model = PeftModel.from_pretrained(base_model, str(self.adapter_path))
        else:
            self._model = base_model
        self._processor = AutoProcessor.from_pretrained(path)
        print("[QwenVLCaptionGenerator] 模型加载完成")

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def _build_prompt(self, original_caption: str) -> str:
        """根据是否有原始 caption 选择并渲染 prompt 模板。"""
        if original_caption and original_caption.strip():
            return _PROMPT_WITH_CAPTION.replace("{caption}", original_caption.strip())
        return _PROMPT_WITHOUT_CAPTION

    def generate(self, image: "PILImage.Image", prompt: str = "") -> str:
        """为单张 figure 生成检索描述。

        Args:
            image:  PIL.Image（RGB）。
            prompt: 原始 caption（来自 PDF 提取）；非空时注入 prompt 模板。

        Returns:
            生成的英文描述；失败时返回空字符串。
        """
        if image is None:
            return ""
        if self._model is None:
            self._load()

        try:
            import torch
            from qwen_vl_utils import process_vision_info

            final_prompt = self._build_prompt(prompt)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": final_prompt},
                ],
            }]

            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self._model.device)

            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                ids = self._model.generate(
                    **inputs,
                    max_new_tokens=300,
                    min_new_tokens=8,
                    do_sample=False,
                    num_beams=5,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=4,
                    length_penalty=1.0,
                    early_stopping=True,
                )
            generated_ids = ids[0][input_len:]
            return self._processor.decode(generated_ids, skip_special_tokens=True).strip()
        except Exception as exc:
            print(f"[QwenVLCaptionGenerator] generate() 失败: {exc}")
            return ""

    def release(self) -> None:
        """释放模型与 processor 占用的显存和内存。"""
        import gc

        import torch

        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[QwenVLCaptionGenerator] 模型已释放")


class DashScopeCaptionGenerator(CaptionGenerator):
    """API-backed figure caption generator using DashScope OpenAI-compatible VL models."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key_env: str = "LLM_API_KEY",
    ) -> None:
        from src.openai_compat import build_openai_client

        self.model_name = model_name
        self._client = build_openai_client(
            base_url=base_url,
            api_key_env=api_key_env,
            fallback_api_key_env="LLM_API_KEY",
        )
        print(f"[DashScopeCaptionGenerator] model={model_name}")

    def _build_prompt(self, original_caption: str) -> str:
        if original_caption and original_caption.strip():
            return _PROMPT_WITH_CAPTION.replace("{caption}", original_caption.strip())
        return _PROMPT_WITHOUT_CAPTION

    def generate(self, image: "PILImage.Image", prompt: str = "") -> str:
        if image is None:
            return ""

        try:
            from src.ingestion.image_api_utils import pil_image_to_data_url

            final_prompt = self._build_prompt(prompt)
            data_url = pil_image_to_data_url(image, max_side=1400, max_bytes=1_500_000)
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": final_prompt},
                        ],
                    }
                ],
                max_tokens=220,
                temperature=0,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            print(f"[DashScopeCaptionGenerator] generate() 失败: {exc}")
            return ""

    def release(self) -> None:
        self._client = None
        print("[DashScopeCaptionGenerator] released")


def build_default_generator(
    model_dir: Path | None = None,
    model_name: str | None = None,
    adapter_path: Path | None = None,
) -> CaptionGenerator:
    """构造默认 caption 生成器，供 ingest 流程复用。

    Args:
        model_dir: 本地模型目录；传 None 时从 CFG 读取。
        model_name: 覆盖默认模型名称；传 None 时从 CFG 读取。
        adapter_path: LoRA adapter 路径；传 None 时不加载 adapter。
    """
    from src.config import CFG

    if CFG.caption_model.mode.lower().strip() == "api":
        return DashScopeCaptionGenerator(
            model_name=model_name or CFG.caption_model.model_name,
            base_url=CFG.caption_model.base_url,
            api_key_env=CFG.caption_model.api_key_env,
        )

    resolved_dir = model_dir or CFG.caption_model.model_dir
    resolved_name = model_name or CFG.caption_model.model_name
    resolved_adapter = adapter_path if adapter_path is not None else CFG.caption_model.adapter_path
    return QwenVLCaptionGenerator(
        model_dir=resolved_dir,
        model_name=resolved_name,
        adapter_path=resolved_adapter,
    )


def _load_caption_figure_image(figure: dict[str, Any]) -> "PILImage.Image | None":
    """Load a figure image for caption generation, returning ``None`` on failure."""
    from src.ingestion.image_ingestor import load_figure_image

    try:
        return load_figure_image(figure)
    except ValueError:
        return None


def enrich_figures_with_captions(
    figures: list[dict[str, Any]],
    model_dir: "Path | None" = None,
    model_name: str | None = None,
    generator: "CaptionGenerator | None" = None,
) -> list[dict[str, Any]]:
    """用 caption 模型为 figures 列表中每张图片生成描述，返回新列表。

    为每个 figure dict 写入：
        - ``caption_generated``：模型生成的描述
        - ``caption_merged``：原始 caption 与生成 caption 合并后的文本

    image 来源优先级：``image_bytes`` > ``image_path``；两者均无时跳过。

    Args:
        figures:    parse_pdf_multimodal 返回的 figure dict 列表。
        model_dir:  本地模型快照目录，传给 QwenVLCaptionGenerator。
        model_name: 覆盖默认模型名称。
        generator:  已实例化的 CaptionGenerator，优先使用。

    Returns:
        enriched figure dict 列表（浅拷贝，不修改原列表）。
    """
    if generator is None:
        generator = build_default_generator(model_dir=model_dir, model_name=model_name)

    enriched = [dict(fig) for fig in figures]
    for fig in enriched:
        pil_img = _load_caption_figure_image(fig)
        cap_orig = fig.get("caption", "")
        cap_gen = generator.generate(pil_img, cap_orig) if pil_img else ""
        fig["caption_generated"] = cap_gen
        fig["caption_merged"] = merge_captions(cap_orig, cap_gen)
    return enriched


def merge_captions(caption_original: str, caption_generated: str) -> str:
    """拼接原始 caption 和模型生成 caption（换行符分隔）。

    原始 caption 保留论文作者意图，生成 caption 补充视觉细节。
    任一为空时只保留非空部分。
    """
    parts = [s.strip() for s in [caption_original, caption_generated] if s and s.strip()]
    return "\n".join(parts)
