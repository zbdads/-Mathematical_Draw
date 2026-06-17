"""PDF 解析模块：PDF → page-level 文本列表。"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - exercised in lightweight CPU/API repro envs
    torch = None

# 模块级 Docling converter 懒加载缓存（避免多次初始化模型）
# key: enable_formula_enrichment
_docling_converters: dict[bool, Any] = {}


def _build_accelerator_options():
    """优先让 Docling 使用 CUDA。"""
    from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions

    device = (
        AcceleratorDevice.CUDA
        if torch is not None and torch.cuda.is_available()
        else AcceleratorDevice.CPU
    )
    print(f"  [docling] accelerator={device.value}")
    return AcceleratorOptions(device=device, num_threads=4)


def _caption_to_text(caption_obj: Any, document: Any) -> str:
    """将 Docling caption 对象安全转换为文本，兼容 RefItem 和 TextItem。"""
    if caption_obj is None:
        return ""

    obj = caption_obj
    resolve = getattr(caption_obj, "resolve", None)
    if callable(resolve):
        try:
            obj = resolve(document)
        except Exception:
            obj = caption_obj

    for field in ("text", "orig", "label"):
        value = getattr(obj, field, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    if isinstance(obj, dict):
        for key in ("text", "orig", "label"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return str(obj)


def _get_docling_converter(enable_formula_enrichment: bool = False):
    """懒加载并缓存 Docling DocumentConverter。

    若 config.yml paths.docling_models_dir 不为空且目录存在，
    则通过 artifacts_path 使用本地模型，跳过网络下载。
    """
    if enable_formula_enrichment in _docling_converters:
        return _docling_converters[enable_formula_enrichment]

    if enable_formula_enrichment not in _docling_converters:
        from docling.document_converter import (
            DocumentConverter,
            InputFormat,
            PdfFormatOption,
        )
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        from src.config import CFG

        kw: dict[str, Any] = {"do_ocr": False, "do_table_structure": True}
        if enable_formula_enrichment:
            kw["do_formula_enrichment"] = True

        docling_models_dir = CFG.paths.docling_models_dir
        if docling_models_dir is not None and docling_models_dir.exists():
            print(f"  [docling] 使用本地模型：{docling_models_dir}")
            kw["artifacts_path"] = str(docling_models_dir)
            if enable_formula_enrichment:
                formula_model_dir = docling_models_dir / "docling-project--CodeFormulaV2"
                if not formula_model_dir.exists():
                    raise FileNotFoundError(
                        "本地 Docling 模型目录缺少 CodeFormulaV2："
                        f"{formula_model_dir}"
                    )

        # generate_picture_images=True 让 Docling 裁剪图片区域并保存为 PIL Image
        try:
            pipeline_options = PdfPipelineOptions(
                **kw,
                accelerator_options=_build_accelerator_options(),
                generate_picture_images=True,
                images_scale=2.0,
            )
        except TypeError:
            # 兼容旧版 Docling（仅去掉 generate_picture_images 参数）
            pipeline_options = PdfPipelineOptions(
                **kw,
                accelerator_options=_build_accelerator_options(),
            )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        _docling_converters[enable_formula_enrichment] = converter
    return _docling_converters[enable_formula_enrichment]


def _extract_figures(document) -> list[dict[str, Any]]:
    """从 Docling document 中提取图片元数据。复用 docling_demo.demo_pictures() 逻辑。"""
    figures: list[dict[str, Any]] = []
    for pic in document.pictures:
        prov = pic.prov[0]
        bbox = prov.bbox
        caption = _caption_to_text(pic.captions[0], document) if pic.captions else ""

        # 尝试获取图片字节（需要 generate_picture_images=True 且 Docling >= 2.x）
        image_bytes = None
        try:
            img_ref = getattr(pic, "image", None)
            if img_ref is not None:
                pil_img = getattr(img_ref, "pil_image", None)
                if pil_img is not None:
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    image_bytes = buf.getvalue()
        except Exception:
            pass

        figures.append(
            {
                "page": prov.page_no,
                "bbox": (bbox.l, bbox.t, bbox.r, bbox.b),
                "caption": caption,
                "image_bytes": image_bytes,
            }
        )
    return figures


def _extract_tables(document) -> list[dict[str, Any]]:
    """从 Docling document 中提取表格元数据。复用 docling_demo.demo_tables() 逻辑。"""
    tables: list[dict[str, Any]] = []
    for table in document.tables:
        prov = table.prov[0]
        caption = _caption_to_text(table.captions[0], document) if table.captions else ""
        try:
            md = table.export_to_markdown(document)
        except TypeError:
            # 兼容旧版 Docling（仅支持无参调用）
            md = table.export_to_markdown()

        df = None
        try:
            df = table.export_to_dataframe(document)
        except TypeError:
            df = table.export_to_dataframe()
        except Exception:
            pass

        tables.append(
            {
                "page": prov.page_no,
                "caption": caption,
                "markdown": md,
                "dataframe": df,
            }
        )
    return tables


def _extract_equations(document) -> list[dict[str, Any]]:
    """从 Docling document.texts 中提取公式元数据。"""
    equations: list[dict[str, Any]] = []

    formula_item_type = None
    try:
        from docling_core.types.doc import FormulaItem  # type: ignore

        formula_item_type = FormulaItem
    except Exception:
        formula_item_type = None

    for item in getattr(document, "texts", []):
        label = getattr(item, "label", None)
        label_str = str(getattr(label, "value", label)).strip().lower()

        is_formula = False
        if formula_item_type is not None and isinstance(item, formula_item_type):
            is_formula = True
        elif label_str == "formula" or label_str.endswith(".formula"):
            is_formula = True

        if not is_formula:
            continue

        prov_list = getattr(item, "prov", None) or []
        if not prov_list:
            continue

        prov = prov_list[0]
        bbox_obj = getattr(prov, "bbox", None)
        if bbox_obj is None:
            continue

        content = (getattr(item, "text", None) or getattr(item, "orig", None) or "").strip()
        if not content:
            continue

        bbox = (bbox_obj.l, bbox_obj.t, bbox_obj.r, bbox_obj.b)
        equations.append(
            {
                "page": prov.page_no,
                "bbox": bbox,
                "content": content,
            }
        )

    equations.sort(key=lambda x: (int(x["page"]), x["bbox"][1], x["bbox"][0]))
    return equations


def parse_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    """用 Docling 版面分析提取 PDF 文本，返回每页文本。

    Docling 具有阅读顺序修正和元素分类能力，对多栏布局的学术论文
    提取质量优于 pdfplumber。已缓存的 converter 会被复用。

    Args:
        pdf_path: PDF 文件路径。

    Returns:
        [{"page": int, "text": str}, ...]，页码从 1 开始。
    """
    try:
        document = _get_docling_converter(enable_formula_enrichment=False).convert(
            str(pdf_path)
        ).document
    except ImportError as exc:
        print(f"  [parser] Docling 不可用，回退到 PyMuPDF 文本解析：{exc}")
        return _parse_pdf_with_pymupdf(pdf_path)

    # 确定公式类型，用于过滤
    formula_item_type = None
    try:
        from docling_core.types.doc import FormulaItem  # type: ignore

        formula_item_type = FormulaItem
    except Exception:
        pass

    # 按页收集文本片段（document.texts 已按阅读顺序排列）
    page_texts: dict[int, list[str]] = {}
    for item in getattr(document, "texts", []):
        # 跳过公式（公式由 parse_pdf_multimodal 单独处理）
        if formula_item_type is not None and isinstance(item, formula_item_type):
            continue
        label = getattr(item, "label", None)
        label_str = str(getattr(label, "value", label)).strip().lower()
        if label_str == "formula" or label_str.endswith(".formula"):
            continue

        text = (getattr(item, "text", None) or getattr(item, "orig", None) or "").strip()
        if not text:
            continue

        # 标题项加 ## 前缀，供 build_chunks() 提取 section 信息
        if label_str in {"title", "section_header"}:
            text = f"## {text}"

        prov_list = getattr(item, "prov", None) or []
        page_no = prov_list[0].page_no if prov_list else 1
        page_texts.setdefault(page_no, []).append(text)

    return [
        {"page": page_no, "text": "\n\n".join(texts)}
        for page_no, texts in sorted(page_texts.items())
    ]


def _parse_pdf_with_pymupdf(pdf_path: Path) -> list[dict[str, Any]]:
    """Lightweight text-only PDF parser used when Docling is not installed."""
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "Docling 不可用，且未安装 PyMuPDF。请安装 docling 或 pymupdf 后重试。"
        ) from exc

    pages: list[dict[str, Any]] = []
    with fitz.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": i, "text": text})
    return pages


def _parse_pdf_pages_as_figures_with_pymupdf(
    pdf_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fallback multimodal parser: render each PDF page as one visual figure."""
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "Docling 不可用，且未安装 PyMuPDF。请安装 docling 或 pymupdf 后重试。"
        ) from exc

    figures: list[dict[str, Any]] = []
    with fitz.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image_bytes = pix.tobytes("png")
            rect = page.rect
            figures.append(
                {
                    "page": i,
                    "bbox": (0.0, 0.0, float(rect.width), float(rect.height)),
                    "caption": f"Rendered full page {i} from {pdf_path.stem}.",
                    "image_bytes": image_bytes,
                    "source": "pymupdf_page_render",
                }
            )
    return figures, [], []


def parse_pdf_multimodal(
    pdf_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """用 Docling 一次性解析 PDF，返回 (figures, tables, equations)。

    比分别调用 parse_pdf_figures / parse_pdf_tables 更高效，
    CLI --multimodal 模式应优先调用此函数。

    Returns:
        figures: [{"page", "bbox", "caption", "image_bytes"}, ...]
        tables:  [{"page", "caption", "markdown", "dataframe"}, ...]
        equations: [{"page", "bbox", "content"}, ...]
    """
    try:
        document = _get_docling_converter(enable_formula_enrichment=True).convert(
            str(pdf_path)
        ).document
    except ImportError as exc:
        print(f"  [parser] Docling 不可用，回退到 PyMuPDF 页面图解析：{exc}")
        return _parse_pdf_pages_as_figures_with_pymupdf(pdf_path)
    figures = _extract_figures(document)
    tables = _extract_tables(document)
    equations = _extract_equations(document)
    return figures, tables, equations


def release_converter() -> None:
    """释放 Docling converter 缓存，回收 GPU 显存。"""
    import gc

    _docling_converters.clear()
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[docling] converter 缓存已释放")


def parse_pdf_figures(pdf_path: Path) -> list[dict[str, Any]]:
    """提取 PDF 中的图片。

    使用 Docling DocumentConverter 解析，遍历 document.pictures，
    输出 {"page", "bbox", "caption", "image_bytes"}。

    若需同时获取表格或公式，建议改用 parse_pdf_multimodal() 避免重复解析。
    """
    document = _get_docling_converter(enable_formula_enrichment=False).convert(
        str(pdf_path)
    ).document
    return _extract_figures(document)


def parse_pdf_tables(pdf_path: Path) -> list[dict[str, Any]]:
    """提取 PDF 中的表格。

    使用 Docling DocumentConverter + TableFormer，遍历 document.tables，
    输出 {"page", "caption", "markdown", "dataframe"}。

    若需同时获取图片或公式，建议改用 parse_pdf_multimodal() 避免重复解析。
    """
    document = _get_docling_converter(enable_formula_enrichment=False).convert(
        str(pdf_path)
    ).document
    return _extract_tables(document)
