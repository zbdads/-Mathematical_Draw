"""Ingest pipeline：PDF 解析 → 分块 → 向量化 → 建索引。

在多模态模式下，同步构建文本相关索引和图像检索索引。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import CFG
from src.index.metadata_store import MetadataStore
from src.index.vector_store import VectorStore
from src.ingestion.chunk_id import assign_figure_ids
from src.ingestion.multimodal_chunker import build_chunks
from src.ingestion.parser import parse_pdf, parse_pdf_multimodal
from src.ingestion.text_embedder import TextEmbedder

if TYPE_CHECKING:
    from src.ingestion.caption_generator import CaptionGenerator
    from src.ingestion.image_embedder import ImageEmbedder
    from src.retrieval.image_retriever import ImageRetriever

_INDEX_PATH = CFG.paths.index_dir / "text_index.faiss"
_CAPTION_BGE_INDEX_PATH = CFG.paths.index_dir / "caption_index_bge.faiss"
_LEGACY_CAPTION_INDEX_PATH = CFG.paths.index_dir / "caption_index.faiss"
_FIGURES_DIR = CFG.paths.index_dir / "figures"


def _chunks_path(paper_id: str) -> Path:
    return CFG.paths.chunks_dir / f"{paper_id}_chunks.json"


def _combined_chunks_path() -> Path:
    return CFG.paths.chunks_dir / "all_chunks.json"


def _attach_figure_image_paths(
    chunks: list[dict],
    figures: list[dict] | None,
) -> None:
    """Propagate persisted figure image paths into figure chunks."""
    if not figures:
        return

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


def run_ingest(
    pdf_path: Path,
    paper_id: str,
    multimodal: bool = False,
    overwrite: bool = False,
    use_caption_model: bool = False,
    embedder: TextEmbedder | None = None,
    image_embedder: "ImageEmbedder | None" = None,
    caption_generator: "CaptionGenerator | None" = None,
    vector_store: VectorStore | None = None,
    metadata_store: MetadataStore | None = None,
    image_retriever: "ImageRetriever | None" = None,
    caption_vector_store: VectorStore | None = None,
    on_progress: Callable[[str, str], None] | None = None,
) -> dict:
    """执行完整 ingest 流程。

    Returns:
        统计信息 dict，包含 pages / text_chunks / figure_chunks /
        text_dim / image_dim / text_index_total / figures_indexed。
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 不存在：{pdf_path}")

    combined_chunks_path = _combined_chunks_path()
    index_exists = _INDEX_PATH.exists()
    chunks_exists = combined_chunks_path.exists()
    if not overwrite and index_exists != chunks_exists:
        raise RuntimeError(
            "索引文件和合并 chunks 文件状态不一致，请检查并手动清理后重试，"
            "或使用 --overwrite 强制重建。"
        )
    if not overwrite and chunks_exists:
        existing_store = MetadataStore.load(combined_chunks_path)
        existing_papers = {str(c.get("paper_id", "")) for c in existing_store.get_all()}
        if paper_id in existing_papers:
            raise ValueError(
                f"paper_id={paper_id!r} 已存在于 {combined_chunks_path}。\n"
                "为避免重复入库，请改用 --overwrite 覆盖重建。"
            )

    print(f"\n[ingest] PDF: {pdf_path}  paper_id: {paper_id}")

    # 1. 解析 PDF
    print("[1/4] 解析 PDF …")
    if on_progress:
        on_progress("1/4", "解析 PDF…")
    pages = parse_pdf(pdf_path)
    print(f"  共 {len(pages)} 页")

    # 2. 分块
    figures, tables, equations = None, None, None
    if multimodal:
        print("[2/4] Docling 提取文档结构（首次运行会下载模型）…")
        if on_progress:
            on_progress("2/4", "Docling 提取文档结构…")
        figures, tables, equations = parse_pdf_multimodal(pdf_path)
        print(f"  共 {len(figures)} 张图片，{len(tables)} 个表格，{len(equations)} 个公式")

        # 预分配 figure_id（格式与 ImageRetriever 一致：{paper_id}_fig_{page}_{idx}）
        # 必须在 build_chunks 和 ingest_images 前完成，以保证两路 ID 一致
        assign_figure_ids(figures, paper_id)

        # 可选：caption 模型生成 caption_generated，合并为 caption_merged
        if use_caption_model:
            from src.ingestion.caption_generator import enrich_figures_with_captions

            print("[2.5/4] Caption 模型生成 figure caption …")
            if on_progress:
                on_progress("2.5/4", f"Caption 模型生成 figure caption（{len(figures)} 张）…")
            figures = enrich_figures_with_captions(
                figures, model_dir=CFG.caption_model.model_dir, generator=caption_generator
            )
            print(f"  Caption 生成完成（{len(figures)} 张）")
            if on_progress:
                on_progress("2.5/4", f"Caption 生成完成（{len(figures)} 张）")
        else:
            for fig in figures:
                fig.setdefault("caption_generated", "")
                fig.setdefault("caption_merged", fig.get("caption", ""))
    else:
        print("[2/4] chunk 分块 …")
        if on_progress:
            on_progress("2/4", "分块中…")

    chunks = build_chunks(
        pages,
        paper_id=paper_id,
        chunk_size=CFG.chunker.chunk_size,
        overlap_sentences=CFG.chunker.overlap_sentences,
        min_chunk_size=CFG.chunker.min_chunk_size,
        prepend_header=CFG.chunker.prepend_header,
        figures=figures,
        tables=tables,
        equations=equations,
    )
    print(f"  共 {len(chunks)} 个 chunk")

    # 3. 向量化（text chunk 与 figure caption chunk 分开编码）
    print("[3/4] 向量化 …")
    if on_progress:
        on_progress("3/4", "向量化…")
    embedder = embedder or TextEmbedder()

    text_only_chunks = [c for c in chunks if c.get("modality") != "figure"]
    caption_chunks_list = [c for c in chunks if c.get("modality") == "figure"]

    text_embeddings = embedder.encode([c["content"] for c in text_only_chunks])
    print(f"  text embedding shape: {text_embeddings.shape}")
    for chunk, emb in zip(text_only_chunks, text_embeddings):
        chunk["embedding_text"] = emb.tolist()

    if caption_chunks_list:
        caption_embeddings = embedder.encode([c["content"] for c in caption_chunks_list])
        print(f"  caption embedding shape: {caption_embeddings.shape}")
        for chunk, emb in zip(caption_chunks_list, caption_embeddings):
            chunk["embedding_text"] = emb.tolist()

    # 3.5 图像索引（仅 multimodal 且有图片时）
    image_ingest_count = 0
    if multimodal and figures:
        print("[3.5/4] 建立图像向量索引（Qwen3-VL-Embedding）…")
        from src.ingestion.image_ingestor import ingest_images
        if image_embedder is None:
            from src.ingestion.image_embedder import ImageEmbedder as _IE
            image_embedder = _IE()
        if image_retriever is None:
            from src.retrieval.image_retriever import ImageRetriever as _IR
            image_retriever = _IR()
        images_dir = _FIGURES_DIR / paper_id
        new_figures, vecs = ingest_images(figures, paper_id, images_dir, image_embedder)
        qwen_caption_texts = [f.get("caption_merged") or f.get("caption", "") for f in new_figures]
        qwen_caption_embeddings = (
            image_embedder.encode_texts(qwen_caption_texts)
            if qwen_caption_texts
            else None
        )
        image_ingest_count = image_retriever.add(
            new_figures,
            vecs,
            caption_embeddings_qwenvl=qwen_caption_embeddings,
            overwrite=overwrite,
        )
        _attach_figure_image_paths(chunks, new_figures)
        print(f"  图像索引入库：{image_ingest_count} 张")

    # 4. 建索引并持久化
    print("[4/4] 建立 FAISS 索引并保存 …")
    if on_progress:
        on_progress("4/4", "建立 FAISS 索引并保存…")

    if vector_store is not None and metadata_store is not None:
        # 共享实例：直接复用调用方传入的索引，无需磁盘加载
        vs = vector_store
        ms = metadata_store
    elif overwrite:
        print("  overwrite=True：重建索引与元数据（覆盖已有数据）")
        vs = VectorStore(dim=embedder.dim)
        ms = MetadataStore()
    elif index_exists and chunks_exists:
        vs = VectorStore.load(_INDEX_PATH)
        ms = MetadataStore.load(combined_chunks_path)
    else:
        vs = VectorStore(dim=embedder.dim)
        ms = MetadataStore()

    # ALL chunks → metadata（figure chunk 也需要用于 citation 输出）
    all_chunk_ids = ms.add_chunks(chunks)

    # text_index：仅写入 text chunk 向量
    text_chunk_ids = [cid for cid, c in zip(all_chunk_ids, chunks) if c.get("modality") != "figure"]
    vs.add(text_embeddings, text_chunk_ids)

    vs.save(_INDEX_PATH)
    ms.save(combined_chunks_path)

    # caption_index：仅写入 figure caption 向量
    if caption_chunks_list:
        if caption_vector_store is not None:
            caption_vs = caption_vector_store
        elif overwrite:
            caption_vs = VectorStore(dim=embedder.dim)
        elif _CAPTION_BGE_INDEX_PATH.exists():
            caption_vs = VectorStore.load(_CAPTION_BGE_INDEX_PATH)
        elif _LEGACY_CAPTION_INDEX_PATH.exists():
            caption_vs = VectorStore.load(_LEGACY_CAPTION_INDEX_PATH)
        else:
            caption_vs = VectorStore(dim=embedder.dim)

        caption_chunk_ids = [cid for cid, c in zip(all_chunk_ids, chunks) if c.get("modality") == "figure"]
        caption_vs.add(caption_embeddings, caption_chunk_ids)
        caption_vs.save(_CAPTION_BGE_INDEX_PATH)
        print(f"  Caption 索引已保存：{_CAPTION_BGE_INDEX_PATH}（{caption_vs.ntotal} 向量）")

    per_paper_ms = MetadataStore()
    per_paper_ms.add_chunks(chunks)
    per_paper_ms.save(_chunks_path(paper_id))

    print(f"\n  索引已保存：{_INDEX_PATH}")
    print(f"  Chunks 已保存：{_chunks_path(paper_id)}")
    print(f"  合并 Chunks：{combined_chunks_path}")
    print(f"  索引总向量数：{vs.ntotal}")

    return {
        "pages": len(pages),
        "text_chunks": len(text_only_chunks),
        "figure_chunks": len(caption_chunks_list),
        "text_dim": embedder.dim,
        "image_dim": image_embedder.dim if image_embedder is not None else None,
        "text_index_total": vs.ntotal,
        "figures_indexed": image_ingest_count,
    }
