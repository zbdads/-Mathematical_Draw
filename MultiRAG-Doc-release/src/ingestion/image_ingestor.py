"""图像 ingest 辅助：PIL 解码、磁盘落盘、Qwen3-VL 向量化。

职责：
    - 接收 parser 输出的 figure list（含 image_bytes 或 image_path）。
    - 将有效图像解码为 PIL.Image，保存到磁盘。
    - 调用 ImageEmbedder 进行批量向量化。
    - 返回 (figure_metadata_list, embeddings)，由 pipeline 层写入索引。

不负责：
    - FAISS 索引管理（由 src/index/ 或 ImageRetriever 负责）。
    - figure metadata 持久化（由 ImageRetriever.save() 负责）。
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.ingestion.image_embedder import ImageEmbedder
    from PIL import Image as PILImage


def load_figure_image(figure: dict[str, Any]) -> "PILImage.Image":
    """Load a figure image from ``image_bytes`` or ``image_path``.

    Raises:
        ValueError: The figure has no usable image source or decoding fails.
    """
    from PIL import Image

    if figure.get("image_bytes") is not None:
        try:
            return Image.open(io.BytesIO(figure["image_bytes"])).convert("RGB")
        except Exception as exc:
            raise ValueError(f"解码 image_bytes 失败: {exc}") from exc

    image_path = figure.get("image_path")
    if image_path:
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"读取 image_path 失败: {exc}") from exc

    raise ValueError("无图像数据（image_bytes 和 image_path 均缺失）")


def ingest_images(
    figures: list[dict[str, Any]],
    paper_id: str,
    images_dir: Path,
    embedder: "ImageEmbedder",
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """PIL decode + disk write + encode figures. Returns (figure_metas, embeddings).

    Args:
        figures:    parse_pdf_multimodal() 返回的 figure 列表，每条须含：
                    page（int）、caption（str），以及
                    image_bytes（bytes）或 image_path（str）之一。
                    若已预分配 figure_id，优先使用；否则按 {paper_id}_fig_{page}_{idx} 生成。
        paper_id:   论文 ID，用于生成兜底 figure_id。
        images_dir: 图像文件保存目录（不存在时自动创建）。
        embedder:   已加载的 ImageEmbedder 实例。

    Returns:
        (new_figures, embeddings)：
            new_figures — 可直接写入 image_index metadata 的 dict 列表。
            embeddings  — shape (n, dim) 的 float32 ndarray，与 new_figures 一一对齐。
        若所有图像均无效，返回 ([], empty_array)。
    """
    images_dir.mkdir(parents=True, exist_ok=True)

    page_counters: dict[int, int] = {}
    pil_images: list["PILImage.Image"] = []
    new_figures: list[dict[str, Any]] = []

    for fig in figures:
        page = int(fig.get("page", 0))
        idx = page_counters.get(page, 0)
        page_counters[page] = idx + 1
        # 优先使用预分配的 figure_id（保证与 text_index 中的 figure chunk ID 一致）
        figure_id = fig.get("figure_id") or f"{paper_id}_fig_{page}_{idx}"

        try:
            pil_img = load_figure_image(fig)
        except ValueError as exc:
            print(f"  [跳过] {figure_id}：{exc}")
            continue

        img_path = images_dir / f"{figure_id}.png"
        pil_img.save(img_path)

        pil_images.append(pil_img)
        new_figures.append(
            {
                "figure_id": figure_id,
                "paper_id": paper_id,
                "page": [page],
                "caption": fig.get("caption", ""),
                "caption_generated": fig.get("caption_generated", ""),
                "caption_merged": fig.get("caption_merged") or fig.get("caption", ""),
                "image_path": str(img_path),
                "modality": "figure",
            }
        )

    if not pil_images:
        print("  [警告] 无有效图像，图像索引未建立。")
        return [], np.empty((0, embedder.dim), dtype=np.float32)

    captions = [fig.get("caption_merged") or fig.get("caption", "") for fig in new_figures]
    print(f"  [image_ingestor] 编码 {len(pil_images)} 张图像（图像 + caption 向量）…")
    embeddings = embedder.encode_images(pil_images, captions=captions)
    return new_figures, embeddings
