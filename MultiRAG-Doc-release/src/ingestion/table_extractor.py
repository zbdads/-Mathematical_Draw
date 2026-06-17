"""表格抽取与数值结构化模块。

在 parser.py 的 Docling 提取基础上，进一步将 DataFrame 转为结构化字典，
并做数值预处理（去除 %, ±, < 等符号后转 float）。

用法：
    from src.ingestion.table_extractor import extract_tables

    tables = extract_tables(Path("paper.pdf"))
    # 每条：{"page", "caption", "markdown", "headers", "rows"}
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from src.ingestion.parser import parse_pdf_tables

logger = logging.getLogger(__name__)

_STRIP_PATTERN = re.compile(r"[%±<>≤≥~\s]")


def _to_float(cell: str) -> float | None:
    """将单元格字符串转为 float，失败返回 None。"""
    cleaned = _STRIP_PATTERN.sub("", str(cell))
    # 保留正负号和小数点
    cleaned = re.sub(r"[^\d.\-]", "", cleaned)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        logger.debug("无法转换为数值：%r → %r", cell, cleaned)
        return None


def _dataframe_to_rows(df: Any) -> tuple[list[str], list[dict[str, Any]]]:
    """将 pandas DataFrame 转为 (headers, rows) 结构。

    headers: 列名列表（字符串）。
    rows:    每行是 {列名: {"raw": str, "value": float|None}} 字典。
    """
    headers = [str(col) for col in df.columns.tolist()]
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        row_dict: dict[str, Any] = {}
        for col in headers:
            raw = str(row[col]) if row[col] is not None else ""
            row_dict[col] = {"raw": raw, "value": _to_float(raw)}
        rows.append(row_dict)
    return headers, rows


def structure_tables(raw_tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """对已解析的 table dict 列表做数值结构化。

    Args:
        raw_tables: parse_pdf_tables() 返回的原始表格列表，
                    每条至少含 page / caption / markdown / dataframe。

    Returns:
        结构化表格列表，每条格式：
        {
            "page":     int,          # 页码（从 1 开始）
            "caption":  str,          # 表格标题（可能为空）
            "markdown": str,          # Markdown 格式原始表格
            "headers":  list[str],    # 表头列名
            "rows": [                 # 数据行
                {
                    "列名": {
                        "raw":   str,         # 原始字符串
                        "value": float|None,  # 数值（解析失败为 None）
                    },
                    ...
                },
                ...
            ],
        }
    """
    results: list[dict[str, Any]] = []

    for t in raw_tables:
        df = t.get("dataframe")
        if df is None or df.empty:
            headers: list[str] = []
            rows: list[dict[str, Any]] = []
        else:
            headers, rows = _dataframe_to_rows(df)

        results.append(
            {
                "page": t["page"],
                "caption": t["caption"],
                "markdown": t["markdown"],
                "headers": headers,
                "rows": rows,
            }
        )

    return results


def extract_tables(pdf_path: Path) -> list[dict[str, Any]]:
    """从 PDF 提取表格并做数值结构化（便捷包装）。

    等同于 ``structure_tables(parse_pdf_tables(pdf_path))``。
    """
    return structure_tables(parse_pdf_tables(pdf_path))
