"""分阶段 ingest pipeline。

按资源类型拆分批量入库流程：
    1. 解析 PDF 并写入 staging 中间产物
    2. 为 figure 生成补充 caption
    3. 构建 chunk 并写入文本相关索引
    4. 构建图像向量索引

中间产物存放于 ``database/staging/{paper_id}/``，每个阶段完成后写入
``stage_N_done.json`` 标记文件，支持断点续跑。
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import CFG

_STAGING_DIR = CFG.paths.root / "database" / "staging"
_INDEX_PATH = CFG.paths.index_dir / "text_index.faiss"
_CAPTION_BGE_INDEX_PATH = CFG.paths.index_dir / "caption_index_bge.faiss"
_LEGACY_CAPTION_INDEX_PATH = CFG.paths.index_dir / "caption_index.faiss"
_FIGURES_DIR = CFG.paths.index_dir / "figures"


# ── 幂等性工具 ────────────────────────────────────────────────────────────


def _staging_dir(paper_id: str) -> Path:
    return _STAGING_DIR / paper_id


def _is_stage_done(paper_id: str, stage: int) -> bool:
    """检查指定论文的某阶段是否已完成。"""
    return (_staging_dir(paper_id) / f"stage_{stage}_done.json").exists()


def _mark_stage_done(paper_id: str, stage: int) -> None:
    """写入阶段完成标记文件。"""
    marker = _staging_dir(paper_id) / f"stage_{stage}_done.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    data = {"stage": stage, "paper_id": paper_id, "done_at": datetime.now(timezone.utc).isoformat()}
    with open(marker, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── 序列化工具 ────────────────────────────────────────────────────────────


def _save_staging(
    paper_id: str,
    pages: list[dict[str, Any]],
    figures: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    equations: list[dict[str, Any]],
) -> None:
    """将解析阶段产物写入 staging 目录。

    - image_bytes 保存为 PNG 文件，JSON 中存 image_path
    - tables 中的 dataframe 字段丢弃（下游只用 markdown）
    """
    sd = _staging_dir(paper_id)
    sd.mkdir(parents=True, exist_ok=True)
    figures_img_dir = sd / "figures"
    figures_img_dir.mkdir(exist_ok=True)

    # 页面文本
    with open(sd / "pages.json", "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    # 图片元数据：将 image_bytes 落盘为 PNG，JSON 中仅保留路径
    serializable_figures: list[dict[str, Any]] = []
    for fig in figures:
        fig_copy = {k: v for k, v in fig.items() if k != "image_bytes"}
        image_bytes = fig.get("image_bytes")
        if image_bytes is not None:
            figure_id = fig.get("figure_id", "unknown")
            img_path = figures_img_dir / f"{figure_id}.png"
            from PIL import Image

            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            pil_img.save(img_path)
            fig_copy["image_path"] = str(img_path)
        # 移除不可 JSON 序列化的 dataframe 字段
        fig_copy.pop("dataframe", None)
        serializable_figures.append(fig_copy)

    with open(sd / "figures.json", "w", encoding="utf-8") as f:
        json.dump(serializable_figures, f, ensure_ascii=False, indent=2)

    # 表格元数据：仅保留可序列化字段
    serializable_tables = []
    for table in tables:
        t_copy = {k: v for k, v in table.items() if k != "dataframe"}
        serializable_tables.append(t_copy)

    with open(sd / "tables.json", "w", encoding="utf-8") as f:
        json.dump(serializable_tables, f, ensure_ascii=False, indent=2)

    # 公式元数据
    with open(sd / "equations.json", "w", encoding="utf-8") as f:
        json.dump(equations, f, ensure_ascii=False, indent=2)


def _load_staging(paper_id: str) -> dict[str, Any]:
    """从 staging 目录读取中间产物。

    Returns:
        {"pages": [...], "figures": [...], "tables": [...], "equations": [...]}
    """
    sd = _staging_dir(paper_id)
    result: dict[str, Any] = {}
    for key in ("pages", "figures", "tables", "equations"):
        path = sd / f"{key}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                result[key] = json.load(f)
        else:
            result[key] = []
    return result


def _attach_figure_image_paths(
    chunks: list[dict[str, Any]],
    figures: list[dict[str, Any]],
) -> None:
    """Propagate persisted figure image paths into figure chunks."""
    figure_paths = {
        str(fig.get("figure_id", "")): fig.get("image_path")
        for fig in figures
        if fig.get("figure_id")
    }
    for chunk in chunks:
        if chunk.get("modality") != "figure":
            continue
        figure_id = str(chunk.get("id", ""))
        image_path = figure_paths.get(figure_id)
        if image_path:
            chunk["image_path"] = image_path


# ── 解析阶段 ──────────────────────────────────────────────────────────────


def stage_1_parse_all(
    pdf_paths: list[Path],
    paper_ids: list[str],
    force: bool = False,
) -> None:
    """解析全部论文并将中间产物写入 staging。

    完成后释放 Docling converter 缓存。
    """
    from src.ingestion.chunk_id import assign_figure_ids
    from src.ingestion.parser import parse_pdf, parse_pdf_multimodal, release_converter

    for i, (pdf_path, paper_id) in enumerate(zip(pdf_paths, paper_ids)):
        if not force and _is_stage_done(paper_id, 1):
            print(f"  [Stage 1] 跳过 {paper_id}（已完成）")
            continue

        print(f"\n  [Stage 1] [{i + 1}/{len(pdf_paths)}] 解析 {paper_id} …")
        pages = parse_pdf(pdf_path)
        figures, tables, equations = parse_pdf_multimodal(pdf_path)
        print(f"    {len(pages)} 页, {len(figures)} 图, {len(tables)} 表, {len(equations)} 公式")

        assign_figure_ids(figures, paper_id)
        _save_staging(paper_id, pages, figures, tables, equations)
        _mark_stage_done(paper_id, 1)

    release_converter()
    print("\n  [Stage 1] 完成，Docling 已释放")


# ── Caption 阶段 ─────────────────────────────────────────────────────────


def stage_2_caption_all(
    paper_ids: list[str],
    force: bool = False,
) -> None:
    """为全部论文的 figure 生成补充 caption。

    完成后释放 QwenVL 模型。
    """
    from src.ingestion.caption_generator import build_default_generator, enrich_figures_with_captions

    generator = build_default_generator()

    for i, paper_id in enumerate(paper_ids):
        if not force and _is_stage_done(paper_id, 2):
            print(f"  [Stage 2] 跳过 {paper_id}（已完成）")
            continue

        sd = _staging_dir(paper_id)
        figures_path = sd / "figures.json"
        if not figures_path.exists():
            print(f"  [Stage 2] 跳过 {paper_id}（无 figures.json）")
            _mark_stage_done(paper_id, 2)
            continue

        with open(figures_path, encoding="utf-8") as f:
            figures = json.load(f)

        if not figures:
            print(f"  [Stage 2] 跳过 {paper_id}（无 figure）")
            _mark_stage_done(paper_id, 2)
            continue

        print(f"\n  [Stage 2] [{i + 1}/{len(paper_ids)}] Caption {paper_id}（{len(figures)} 张）…")
        enriched = enrich_figures_with_captions(figures, generator=generator)

        # 保存带 caption 字段的 figure 元数据；图像文件已在 staging 中落盘
        with open(sd / "figures_captioned.json", "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)

        _mark_stage_done(paper_id, 2)

    generator.release()
    print("\n  [Stage 2] 完成，QwenVL 已释放")


# ── 文本索引阶段 ─────────────────────────────────────────────────────────


def stage_3_chunk_and_embed_all(
    paper_ids: list[str],
    clean: bool = False,
    force: bool = False,
    skip_caption_index: bool = False,
) -> None:
    """构建 chunk 并写入文本检索相关索引。

    完成后释放 TextEmbedder。
    """
    from src.index.metadata_store import MetadataStore
    from src.index.vector_store import VectorStore
    from src.ingestion.multimodal_chunker import build_chunks
    from src.ingestion.text_embedder import TextEmbedder

    pending_paper_ids = [paper_id for paper_id in paper_ids if force or not _is_stage_done(paper_id, 3)]
    if not pending_paper_ids:
        print("  [Stage 3] 全部已完成，跳过 TextEmbedder 加载")
        return

    print("  [Stage 3] 加载 TextEmbedder …")
    embedder = TextEmbedder()

    chunks_path = CFG.paths.chunks_dir / "all_chunks.json"
    if clean or not (_INDEX_PATH.exists() and chunks_path.exists()):
        vs = VectorStore(dim=embedder.dim)
        ms = MetadataStore()
    else:
        vs = VectorStore.load(_INDEX_PATH)
        ms = MetadataStore.load(chunks_path)
        # 追加模式下检查重复 paper_id，避免同一论文被重复写入全局索引
        existing_paper_ids = {str(c.get("paper_id", "")) for c in ms.get_all()}
        deduped: list[str] = []
        for pid in pending_paper_ids:
            if pid in existing_paper_ids:
                print(f"  [Stage 3] 跳过 {pid}（paper_id 已存在于索引，使用 --force 可强制重建）")
            else:
                deduped.append(pid)
        pending_paper_ids = deduped

    caption_vs = None
    if not skip_caption_index:
        if clean:
            caption_vs = VectorStore(dim=embedder.dim)
        elif _CAPTION_BGE_INDEX_PATH.exists():
            caption_vs = VectorStore.load(_CAPTION_BGE_INDEX_PATH)
        elif _LEGACY_CAPTION_INDEX_PATH.exists():
            caption_vs = VectorStore.load(_LEGACY_CAPTION_INDEX_PATH)
        else:
            caption_vs = VectorStore(dim=embedder.dim)
        if caption_vs._dim != embedder.dim:
            print(
                "  [Stage 3] caption 索引维度与当前 embedding 不一致，重建 caption 索引："
                f"{caption_vs._dim} -> {embedder.dim}"
            )
            caption_vs = VectorStore(dim=embedder.dim)

    for i, paper_id in enumerate(pending_paper_ids):
        print(f"\n  [Stage 3] [{i + 1}/{len(pending_paper_ids)}] Chunk+Embed {paper_id} …")
        staging = _load_staging(paper_id)

        # 优先使用已补充 caption 的 figure 元数据
        sd = _staging_dir(paper_id)
        captioned_path = sd / "figures_captioned.json"
        if captioned_path.exists():
            with open(captioned_path, encoding="utf-8") as f:
                figures = json.load(f)
        else:
            figures = staging["figures"]
            # 未经过 caption 阶段时补齐下游依赖字段
            for fig in figures:
                fig.setdefault("caption_generated", "")
                fig.setdefault("caption_merged", fig.get("caption", ""))

        chunks = build_chunks(
            staging["pages"],
            paper_id=paper_id,
            chunk_size=CFG.chunker.chunk_size,
            overlap_sentences=CFG.chunker.overlap_sentences,
            min_chunk_size=CFG.chunker.min_chunk_size,
            prepend_header=CFG.chunker.prepend_header,
            figures=figures,
            tables=staging["tables"],
            equations=staging["equations"],
        )
        _attach_figure_image_paths(chunks, figures)

        # 文本 chunk 与 figure chunk 分开编码，分别进入对应索引
        text_only_chunks = [c for c in chunks if c.get("modality") != "figure"]
        caption_chunks_list = [c for c in chunks if c.get("modality") == "figure"]

        text_embeddings = embedder.encode([c["content"] for c in text_only_chunks])
        for chunk, emb in zip(text_only_chunks, text_embeddings):
            chunk["embedding_text"] = emb.tolist()

        if caption_chunks_list and not skip_caption_index:
            cap_embeddings = embedder.encode([c["content"] for c in caption_chunks_list])
            for chunk, emb in zip(caption_chunks_list, cap_embeddings):
                chunk["embedding_text"] = emb.tolist()

        # 全量 chunk 元数据统一写入 metadata
        all_chunk_ids = ms.add_chunks(chunks)

        # 文本索引只写入非 figure chunk 向量
        text_chunk_ids = [cid for cid, c in zip(all_chunk_ids, chunks) if c.get("modality") != "figure"]
        vs.add(text_embeddings, text_chunk_ids)

        # caption 索引只写入 figure chunk 向量
        if caption_chunks_list and caption_vs is not None:
            caption_chunk_ids = [cid for cid, c in zip(all_chunk_ids, chunks) if c.get("modality") == "figure"]
            caption_vs.add(cap_embeddings, caption_chunk_ids)

        # 保存单篇论文的 chunk 元数据快照
        per_paper_ms = MetadataStore()
        per_paper_ms.add_chunks(chunks)
        per_paper_ms.save(CFG.paths.chunks_dir / f"{paper_id}_chunks.json")

        print(f"    {len(chunks)} chunks（text={len(text_only_chunks)}, caption={len(caption_chunks_list)}），"
              f"text embedding shape: {text_embeddings.shape}")
        _mark_stage_done(paper_id, 3)

    # 持久化全局文本索引与元数据
    vs.save(_INDEX_PATH)
    ms.save(chunks_path)
    if caption_vs is not None:
        caption_vs.save(_CAPTION_BGE_INDEX_PATH)
        caption_count = caption_vs.ntotal
    else:
        caption_count = 0
    print(f"\n  [Stage 3] 完成，text 索引总向量数：{vs.ntotal}，caption 索引总向量数：{caption_count}")

    embedder.release()
    print("  [Stage 3] TextEmbedder 已释放")


# ── 图像索引阶段 ─────────────────────────────────────────────────────────


def stage_4_image_embed_all(
    paper_ids: list[str],
    clean: bool = False,
    force: bool = False,
) -> None:
    """编码 figure 图像并写入图像检索索引。

    完成后释放 ImageEmbedder。
    """
    pending_paper_ids = [paper_id for paper_id in paper_ids if force or not _is_stage_done(paper_id, 4)]
    if not pending_paper_ids:
        print("  [Stage 4] 全部已完成，跳过 ImageEmbedder 加载")
        return

    from src.ingestion.image_embedder import ImageEmbedder
    from src.ingestion.image_ingestor import ingest_images
    from src.retrieval.image_retriever import ImageRetriever

    print("  [Stage 4] 加载 ImageEmbedder（Qwen3-VL-Embedding）…")
    image_embedder = ImageEmbedder()
    image_retriever = ImageRetriever()

    # 若磁盘上已有图像索引则先加载，以便继续追加
    image_retriever.ensure_loaded()

    for i, paper_id in enumerate(pending_paper_ids):

        sd = _staging_dir(paper_id)
        # 优先使用已补充 caption 的 figure 元数据
        captioned_path = sd / "figures_captioned.json"
        figures_path = sd / "figures.json"
        if captioned_path.exists():
            with open(captioned_path, encoding="utf-8") as f:
                figures = json.load(f)
        elif figures_path.exists():
            with open(figures_path, encoding="utf-8") as f:
                figures = json.load(f)
        else:
            print(f"  [Stage 4] 跳过 {paper_id}（无 figures）")
            _mark_stage_done(paper_id, 4)
            continue

        if not figures:
            print(f"  [Stage 4] 跳过 {paper_id}（无 figure）")
            _mark_stage_done(paper_id, 4)
            continue

        print(f"\n  [Stage 4] [{i + 1}/{len(pending_paper_ids)}] Image Embed {paper_id}（{len(figures)} 张）…")
        images_dir = _FIGURES_DIR / paper_id
        overwrite = clean and i == 0
        new_figures, vecs = ingest_images(figures, paper_id, images_dir, image_embedder)
        qwen_caption_texts = [f.get("caption_merged") or f.get("caption", "") for f in new_figures]
        qwen_caption_embeddings = (
            image_embedder.encode_texts(qwen_caption_texts)
            if qwen_caption_texts
            else None
        )
        count = image_retriever.add(
            new_figures,
            vecs,
            caption_embeddings_qwenvl=qwen_caption_embeddings,
            overwrite=overwrite,
        )
        print(f"    入库 {count} 张图像")
        _mark_stage_done(paper_id, 4)

    image_embedder.release()
    print(f"\n  [Stage 4] 完成，图像索引总向量数：{image_retriever.ntotal}")
    print("  [Stage 4] ImageEmbedder 已释放")


# ── 编排函数 ──────────────────────────────────────────────────────────────


def run_staged_ingest_all(
    pdf_paths: list[Path],
    clean: bool = False,
    clean_index: bool = False,
    force_stages: list[int] | None = None,
    skip_caption: bool = False,
    skip_caption_index: bool = False,
    skip_image: bool = False,
) -> None:
    """按阶段执行完整的批量 ingest 流程。

    Args:
        pdf_paths:     待处理的 PDF 文件列表。
        clean:         从头重建所有数据（删除 index/chunks/staging）。
        clean_index:   仅清空 index/chunks，保留 staging 中间产物。
        force_stages:  强制重跑的阶段编号列表（如 [1, 2]）。
        skip_caption:  跳过 caption 阶段。
        skip_caption_index: 跳过 figure caption 的文本向量索引。
        skip_image:    跳过图像向量索引阶段，仅构建文本/caption 文本索引。
    """
    import shutil

    force_stages = force_stages or []
    paper_ids = [p.stem for p in pdf_paths]

    print(f"\n{'=' * 60}")
    print(f"[staged-ingest] 分阶段 Pipeline")
    print(f"  论文数：{len(pdf_paths)}")
    print(
        f"  clean={clean}  clean_index={clean_index}  skip_caption={skip_caption}  "
        f"skip_caption_index={skip_caption_index}  skip_image={skip_image}  force_stages={force_stages}"
    )
    print(f"{'=' * 60}")

    if clean:
        for d in (CFG.paths.index_dir, CFG.paths.chunks_dir, _STAGING_DIR):
            if d.exists():
                shutil.rmtree(d)
                print(f"[clean] 已删除 {d}")
    elif clean_index:
        for d in (CFG.paths.index_dir, CFG.paths.chunks_dir):
            if d.exists():
                shutil.rmtree(d)
                print(f"[clean-index] 已删除 {d}")

    # 解析阶段
    print(f"\n{'─' * 60}")
    print("[Stage 1] Parse（Docling）")
    print(f"{'─' * 60}")
    stage_1_parse_all(pdf_paths, paper_ids, force=(1 in force_stages))

    # Caption 阶段（可选）
    if not skip_caption:
        print(f"\n{'─' * 60}")
        print("[Stage 2] Caption（QwenVL）")
        print(f"{'─' * 60}")
        stage_2_caption_all(paper_ids, force=(2 in force_stages))
    else:
        print(f"\n[Stage 2] 已跳过（--skip-caption）")

    # 文本索引阶段
    clean_for_embed = clean or clean_index
    force_stage_3 = (3 in force_stages) or clean_index
    force_stage_4 = (4 in force_stages) or clean_index
    print(f"\n{'─' * 60}")
    print("[Stage 3] Chunk + Text Embed（BGE-M3）")
    print(f"{'─' * 60}")
    stage_3_chunk_and_embed_all(
        paper_ids,
        clean=clean_for_embed,
        force=force_stage_3,
        skip_caption_index=skip_caption_index,
    )

    # 图像索引阶段
    if skip_image:
        print(f"\n[Stage 4] 已跳过（--skip-image）")
    else:
        print(f"\n{'─' * 60}")
        print("[Stage 4] Image Embed（Qwen3-VL-Embedding）")
        print(f"{'─' * 60}")
        stage_4_image_embed_all(paper_ids, clean=clean_for_embed, force=force_stage_4)

    print(f"\n{'=' * 60}")
    print("[staged-ingest] 全部完成")
    print(f"{'=' * 60}")
