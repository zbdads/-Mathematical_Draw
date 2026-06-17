"""Utilities for sending local images to hosted multimodal APIs."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage


def pil_image_to_data_url(
    image: "PILImage.Image",
    *,
    max_side: int = 1100,
    max_bytes: int = 900_000,
) -> str:
    """Convert a PIL image to a compressed JPEG data URL within API size limits."""
    img = image.convert("RGB")
    width, height = img.size
    longest = max(width, height)
    if longest > max_side:
        scale = max_side / longest
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        img = img.resize(new_size)

    quality = 88
    data = b""
    while quality >= 45:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            break
        quality -= 10

    if len(data) > max_bytes:
        raise ValueError(
            f"图片压缩后仍超过 API 限制：{len(data)} bytes > {max_bytes} bytes"
        )

    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
