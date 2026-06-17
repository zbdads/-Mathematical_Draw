"""共享工具函数：image_path_to_url、序列化辅助等。"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

from src.config import CFG

logger = logging.getLogger(__name__)

_FIG_ROOT = (CFG.paths.index_dir / "figures").resolve()


def image_path_to_url(image_path: str | None) -> str | None:
    """将 figures 根目录下的文件路径转换为 URL，不合法时返回 None。

    只取路径最后两级（paper_id/filename.png），不依赖绝对路径前缀，
    因此远端 ingest 写入的绝对路径在本地也能正确解析。
    """
    if not image_path:
        return None

    parts = Path(image_path).parts
    if len(parts) < 2:
        return None

    paper_id, filename = parts[-2], parts[-1]
    if not filename.endswith(".png"):
        return None

    return f"/figures/{quote(paper_id)}/{quote(filename)}"


def _coerce(v: Any) -> Any:
    """将 numpy scalar 等非 JSON 原生类型转为 Python 标准类型。"""
    if hasattr(v, "item"):
        return v.item()
    return v


def serialize_result(r: dict) -> dict:
    """将 chunk/evidence dict 转为 JSON-safe dict，剔除 embedding 字段。"""
    out: dict[str, Any] = {}
    for k, v in r.items():
        if k in ("embedding_text", "embedding"):
            continue
        if isinstance(v, dict):
            out[k] = {dk: _coerce(dv) for dk, dv in v.items()}
        elif isinstance(v, list):
            out[k] = [_coerce(x) if not isinstance(x, (str, int, float, bool, type(None))) else x for x in v]
        else:
            out[k] = _coerce(v)
    return out


def serialize_answer(answer: Any) -> dict | None:
    """将 FormattedAnswer dataclass 转为 JSON-safe dict。"""
    if answer is None:
        return None
    try:
        d = asdict(answer)
    except Exception:
        return None
    return d
