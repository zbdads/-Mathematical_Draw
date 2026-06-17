"""批量 ingest pipeline：共享模型实例，依次入库多篇 PDF。

调用方（cli.py）只负责参数解析和结果打印，本模块负责：
- 按 clean 标志决定是否清理现有索引/chunks
- 共享 TextEmbedder / VectorStore / MetadataStore / ImageRetriever / CaptionGenerator 初始化
- 循环调用 run_ingest 并收集统计结果
- 释放 CaptionGenerator 资源
"""

from __future__ import annotations

import shutil
from pathlib import Path

from src.config import CFG
from src.index.metadata_store import MetadataStore
from src.index.vector_store import VectorStore
from src.ingestion.text_embedder import TextEmbedder
from src.pipeline.ingest import run_ingest

_INDEX_PATH = CFG.paths.index_dir / "text_index.faiss"
_CAPTION_BGE_INDEX_PATH = CFG.paths.index_dir / "caption_index_bge.faiss"
_LEGACY_CAPTION_INDEX_PATH = CFG.paths.index_dir / "caption_index.faiss"
_CHUNKS_PATH = CFG.paths.chunks_dir / "all_chunks.json"


def run_ingest_all(
    pdf_paths: list[Path],
    clean: bool = False,
    multimodal: bool = False,
    use_caption_model: bool = False,
) -> dict:
    """批量入库多篇 PDF，在所有论文间共享模型实例。

    Args:
        pdf_paths: 待入库的 PDF 文件列表（已排序）。
        clean: 是否从头重建索引（删除现有 index/chunks）。
        multimodal: 是否启用图像/表格/公式提取。
        use_caption_model: 是否启用 caption 模型为 figure 生成描述。

    Returns:
        dict 包含：
            results  -- 每篇成功入库的统计信息列表
            skipped  -- 被跳过的 paper_id 列表
    """
    if clean:
        for d in (CFG.paths.index_dir, CFG.paths.chunks_dir):
            if d.exists():
                shutil.rmtree(d)
                print(f"[clean] 已删除 {d}")

    print("[init] 加载 TextEmbedder …")
    shared_embedder = TextEmbedder()

    if clean or not (_INDEX_PATH.exists() and _CHUNKS_PATH.exists()):
        print("[init] 新建文本索引 …")
        shared_vs = VectorStore(dim=shared_embedder.dim)
        shared_ms = MetadataStore()
    else:
        print("[init] 从磁盘加载文本索引 …")
        shared_vs = VectorStore.load(_INDEX_PATH)
        shared_ms = MetadataStore.load(_CHUNKS_PATH)

    if clean:
        print("[init] 新建 caption 索引 …")
        shared_caption_vs = VectorStore(dim=shared_embedder.dim)
    elif _CAPTION_BGE_INDEX_PATH.exists():
        print("[init] 从磁盘加载 caption 索引 …")
        shared_caption_vs = VectorStore.load(_CAPTION_BGE_INDEX_PATH)
    elif _LEGACY_CAPTION_INDEX_PATH.exists():
        print("[init] 从旧路径加载 caption 索引 …")
        shared_caption_vs = VectorStore.load(_LEGACY_CAPTION_INDEX_PATH)
    else:
        print("[init] 新建 caption 索引 …")
        shared_caption_vs = VectorStore(dim=shared_embedder.dim)

    shared_image_retriever = None
    shared_image_embedder = None
    shared_caption_generator = None
    if multimodal:
        from src.ingestion.image_embedder import ImageEmbedder
        from src.retrieval.image_retriever import ImageRetriever

        print("[init] 加载 ImageEmbedder（Qwen3-VL-Embedding）…")
        shared_image_embedder = ImageEmbedder()
        shared_image_retriever = ImageRetriever(embedder=shared_image_embedder)
        if use_caption_model:
            from src.ingestion.caption_generator import build_default_generator

            print("[init] 加载 CaptionGenerator …")
            shared_caption_generator = build_default_generator()

    results: list[dict] = []
    skipped: list[str] = []
    for i, pdf_path in enumerate(pdf_paths):
        paper_id = pdf_path.stem
        overwrite = clean and i == 0
        try:
            stat = run_ingest(
                pdf_path,
                paper_id=paper_id,
                multimodal=multimodal,
                overwrite=overwrite,
                use_caption_model=use_caption_model,
                embedder=shared_embedder,
                image_embedder=shared_image_embedder,
                image_retriever=shared_image_retriever,
                caption_generator=shared_caption_generator,
                vector_store=shared_vs,
                metadata_store=shared_ms,
                caption_vector_store=shared_caption_vs,
            )
            results.append({"paper_id": paper_id, **stat})
        except ValueError:
            print(f"[skip] {paper_id}：已入库，跳过")
            skipped.append(paper_id)
        except Exception as e:
            print(f"\n[error] {paper_id}：{e}")
            raise

    if shared_caption_generator is not None:
        shared_caption_generator.release()
    if shared_image_embedder is not None:
        shared_image_embedder.release()

    return {"results": results, "skipped": skipped}
