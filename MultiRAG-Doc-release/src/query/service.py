"""Query service: shared resource management and core retrieval."""

from __future__ import annotations

from pathlib import Path

from src.config import CFG
from src.index.metadata_store import MetadataStore
from src.index.vector_store import VectorStore
from src.ingestion.text_embedder import TextEmbedder
from src.modeling.query_intent import expected_model_fields, is_math_modeling_query
from src.retrieval.image_retriever import ImageRetriever
from src.retrieval.multi_doc_ranker import rank_multi_doc
from src.retrieval.reranker import Reranker
from src.retrieval.text_retriever import TextRetriever

_INDEX_PATH = CFG.paths.index_dir / "text_index.faiss"
_CAPTION_BGE_INDEX_PATH = CFG.paths.index_dir / "caption_index_bge.faiss"
_CAPTION_QWENVL_INDEX_PATH = CFG.paths.index_dir / "caption_index_qwenvl.faiss"
_LEGACY_CAPTION_INDEX_PATH = CFG.paths.index_dir / "caption_index.faiss"
_COMBINED_CHUNKS_PATH: Path = CFG.paths.chunks_dir / "all_chunks.json"
_IMAGE_INDEX_PATH = CFG.paths.index_dir / "image_index.faiss"
_IMAGE_METADATA_PATH = CFG.paths.index_dir / "image_metadata.json"


def flatten_dual_results(retrieval: dict[str, list[dict]]) -> list[dict]:
    """按展示约定拍平双赛道结果：text 在前，figure 在后。"""
    return retrieval.get("text_results", []) + retrieval.get("figure_results", [])


def _boost_modeling_results(question: str, results: list[dict]) -> list[dict]:
    """Promote model-card chunks for modeling queries without hiding raw evidence."""
    if not results or not is_math_modeling_query(question):
        return results
    expected_fields = set(expected_model_fields(question))
    q = question.lower()
    hhc_query = any(
        keyword in q
        for keyword in (
            "home health care",
            "home healthcare",
            "hhc",
            "caregiver",
            "patient",
            "居家医疗",
            "护理员",
            "患者",
        )
    )
    operator_query_hints = _operator_query_hints(q)

    def _boosted_score(hit: dict) -> tuple[float, float]:
        score = float(hit.get("score", hit.get("rerank_score", 0.0)) or 0.0)
        boost = 0.0
        if hit.get("modality") == "math_model":
            boost += 0.20
        elif hit.get("modality") == "model_region":
            boost += 0.06
        chunk_type = str(hit.get("chunk_type", ""))
        if chunk_type in expected_fields:
            boost += 0.24
        elif chunk_type == "model_card":
            boost += 0.08
        elif chunk_type.startswith("model_region_"):
            region_type = str(hit.get("model_region_type", ""))
            if "objective" in expected_fields and "objective" in region_type:
                boost += 0.16
            if "constraints" in expected_fields and "constraint" in region_type:
                boost += 0.16
            if {"variables", "parameters", "sets"} & expected_fields and "notation" in region_type:
                boost += 0.14
            if region_type == "model_section":
                boost += 0.06
        model_elements = set(str(v) for v in hit.get("model_elements", []) or [])
        operator_hints = set(str(v) for v in hit.get("operator_hints", []) or [])
        domain_signals = set(str(v) for v in hit.get("domain_signals", []) or [])
        hhc_signals = set(str(v) for v in hit.get("hhc_signals", []) or [])
        if expected_fields & model_elements:
            boost += 0.08 * len(expected_fields & model_elements)
        if "formula" in model_elements:
            boost += 0.04
        if operator_query_hints & operator_hints:
            boost += 0.07 * len(operator_query_hints & operator_hints)
        if hhc_query and "home_health_care" in domain_signals:
            boost += 0.12
        if hhc_query and hhc_signals:
            boost += min(0.12, 0.025 * len(hhc_signals))
        try:
            evidence_score = float(hit.get("model_evidence_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            evidence_score = 0.0
        if evidence_score > 0:
            boost += min(0.16, evidence_score / 200.0)
        return (score + boost, score)

    for hit in results:
        boosted_score, _ = _boosted_score(hit)
        hit["modeling_boosted_score"] = boosted_score
    boosted = sorted(results, key=_boosted_score, reverse=True)
    for rank, hit in enumerate(boosted, 1):
        hit["rank"] = rank
    return boosted


def _operator_query_hints(question_lower: str) -> set[str]:
    hints: set[str] = set()
    keyword_map = {
        "assignment": ("assign", "assignment", "allocate", "allocation", "分配"),
        "routing_flow": ("route", "routing", "travel", "path", "arc", "路径", "路线"),
        "time_window": ("time window", "earliest", "latest", "时间窗"),
        "time_propagation": ("arrival", "departure", "start time", "service duration", "到达", "离开"),
        "waiting_time": ("waiting", "delay", "tardiness", "等待", "延迟"),
        "capacity": ("capacity", "workload", "working time", "容量", "工作量"),
        "skill_matching": ("skill", "qualification", "技能", "资质"),
        "outsourcing": ("outsourcing", "outsource", "external", "外包"),
        "priority_class": ("vip", "priority", "ordinary patient", "优先"),
        "multi_objective": ("multi-objective", "weighted", "trade-off", "多目标"),
    }
    for hint, keywords in keyword_map.items():
        if any(keyword in question_lower for keyword in keywords):
            hints.add(hint)
    return hints


def _modeling_metadata_score(question: str, hit: dict) -> float:
    """Score model chunks by structural metadata, independent of vector score."""
    if not is_math_modeling_query(question):
        return 0.0
    q = question.lower()
    expected_fields = set(expected_model_fields(question))
    operator_hints = _operator_query_hints(q)
    model_elements = set(str(v) for v in hit.get("model_elements", []) or [])
    hit_operator_hints = set(str(v) for v in hit.get("operator_hints", []) or [])
    domain_signals = set(str(v) for v in hit.get("domain_signals", []) or [])
    hhc_signals = set(str(v) for v in hit.get("hhc_signals", []) or [])

    hhc_query = any(
        keyword in q
        for keyword in (
            "home health care",
            "home healthcare",
            "hhc",
            "caregiver",
            "patient",
            "居家医疗",
            "护理员",
            "患者",
        )
    )
    production_query = any(
        keyword in q
        for keyword in (
            "job shop",
            "flow shop",
            "production",
            "machine",
            "workshop",
            "车间",
            "机器",
        )
    )

    score = 0.0
    if hit.get("modality") == "model_region":
        score += 1.0
    elif hit.get("modality") == "math_model":
        score += 0.8

    chunk_type = str(hit.get("chunk_type", ""))
    region_type = str(hit.get("model_region_type", ""))
    if chunk_type in expected_fields:
        score += 2.0
    if expected_fields & model_elements:
        score += 1.0 * len(expected_fields & model_elements)
    if "formula" in model_elements:
        score += 0.6
    if "objective" in expected_fields and "objective" in region_type:
        score += 1.4
    if "constraints" in expected_fields and "constraint" in region_type:
        score += 1.4
    if {"variables", "parameters", "sets"} & expected_fields and "notation" in region_type:
        score += 1.0
    if operator_hints & hit_operator_hints:
        score += 0.7 * len(operator_hints & hit_operator_hints)
    if hhc_query:
        if "home_health_care" in domain_signals:
            score += 2.5
        score += min(2.0, 0.35 * len(hhc_signals))
        if "production_scheduling" in domain_signals and "home_health_care" not in domain_signals:
            score -= 2.0
    if production_query and "production_scheduling" in domain_signals:
        score += 2.0
    try:
        score += min(2.0, float(hit.get("model_evidence_score", 0.0) or 0.0) / 40.0)
    except (TypeError, ValueError):
        pass
    return score


def _load_compatible_caption_index(embedder: TextEmbedder) -> VectorStore | None:
    """Load a caption text index only when its vector dimension matches.

    Text and caption retrieval share the active TextEmbedder. After changing
    the text embedding model, stale caption indexes can keep an old FAISS
    dimension and would otherwise crash search with an AssertionError.
    """
    candidates = (
        _CAPTION_QWENVL_INDEX_PATH,
        _CAPTION_BGE_INDEX_PATH,
        _LEGACY_CAPTION_INDEX_PATH,
    )
    for path in candidates:
        if not path.exists():
            continue
        try:
            vs = VectorStore.load(path)
        except Exception as exc:
            print(f"  [caption-index] skip unreadable {path.name}: {exc}")
            continue
        if vs._dim == embedder.dim:
            print(f"  [caption-index] loaded {path.name}, dim={vs._dim}")
            return vs
        print(
            "  [caption-index] skip dimension mismatch "
            f"{path.name}: index_dim={vs._dim}, embedder_dim={embedder.dim}"
        )
    return None


class QueryService:
    """持有 query 运行所需资源，并暴露核心检索能力。"""

    def __init__(
        self,
        vs: VectorStore,
        ms: MetadataStore,
        embedder: TextEmbedder,
        reranker: Reranker | None = None,
        img_retriever: ImageRetriever | None = None,
        caption_vs: VectorStore | None = None,
    ) -> None:
        self._embedder = embedder
        self._vs = vs
        self._caption_vs = caption_vs
        self.text_retriever = TextRetriever(vs, ms, embedder)
        self.reranker = reranker
        self.img_retriever = img_retriever
        self.caption_retriever = TextRetriever(caption_vs, ms, embedder) if caption_vs else None
        self._ms = ms

    @property
    def embedder(self) -> TextEmbedder:
        return self._embedder

    @staticmethod
    def from_disk() -> "QueryService":
        """从磁盘加载所有资源，构造 service 实例。"""
        if not _INDEX_PATH.exists():
            raise FileNotFoundError(f"索引不存在：{_INDEX_PATH}\n请先运行 ingest 命令。")
        if not _COMBINED_CHUNKS_PATH.exists():
            raise FileNotFoundError(
                f"Chunk 文件不存在：{_COMBINED_CHUNKS_PATH}\n请先运行 ingest 命令。"
            )

        vs = VectorStore.load(_INDEX_PATH)
        ms = MetadataStore.load(_COMBINED_CHUNKS_PATH)
        embedder = TextEmbedder()

        reranker = None
        rerank_cfg = CFG.reranker
        if rerank_cfg.enabled:
            reranker_local = CFG.paths.models_dir / rerank_cfg.model_name
            reranker_source = str(reranker_local) if reranker_local.exists() else rerank_cfg.model_name
            reranker = Reranker(model_name=reranker_source)

        img_retriever = ImageRetriever.load_if_available(
            index_path=_IMAGE_INDEX_PATH,
            metadata_path=_IMAGE_METADATA_PATH,
        )

        caption_vs = _load_compatible_caption_index(embedder)

        return QueryService(
            vs,
            ms,
            embedder,
            reranker=reranker,
            img_retriever=img_retriever,
            caption_vs=caption_vs,
        )

    def retrieve_core(
        self,
        question: str,
        top_k: int,
        paper_id: str | None,
        skip_rerank: bool = False,
        figure_top_k: int | None = None,
    ) -> dict[str, list[dict]]:
        """检索主干：text 赛道 + figure 赛道（分离返回）。"""
        rerank_cfg = CFG.reranker
        top_k_fig = figure_top_k if figure_top_k is not None else CFG.retriever.top_k_fig
        fetch_k_text = rerank_cfg.candidate_k if (rerank_cfg.enabled and not skip_rerank) else top_k
        if is_math_modeling_query(question):
            fetch_k_text = max(fetch_k_text, top_k + 16)

        query_vec = None
        if paper_id is None and CFG.retriever.multi_paper_balanced:
            query_vec = self._embedder.encode_query(question)
            per_doc_k = max(1, CFG.retriever.multi_paper_per_doc_k)
            if is_math_modeling_query(question):
                # Modeling queries need enough slots for notation/objective/constraint
                # chunks from the same relevant paper before global boosted ranking.
                per_doc_k = max(per_doc_k, min(5, max(3, top_k // 2)))
            per_doc_fetch_k = max(per_doc_k, fetch_k_text)
            if is_math_modeling_query(question):
                per_doc_fetch_k = max(per_doc_fetch_k, per_doc_k + 8)
            text_results = []
            for pid in self._ms.paper_ids():
                per_doc_results = self.text_retriever.retrieve(
                    question,
                    top_k=per_doc_fetch_k,
                    paper_id=pid,
                    query_vec=query_vec,
                )
                if is_math_modeling_query(question):
                    per_doc_results = _boost_modeling_results(question, per_doc_results)
                text_results.extend(per_doc_results[:per_doc_k])
            text_results = sorted(
                text_results,
                key=lambda r: float(r.get("score", 0.0)),
                reverse=True,
            )
        else:
            text_results = self.text_retriever.retrieve(
                question,
                top_k=fetch_k_text,
                paper_id=paper_id,
            )
        text_results = [r for r in text_results if r.get("modality") != "figure"]
        if paper_id is None and text_results and not CFG.retriever.multi_paper_balanced:
            text_results = rank_multi_doc(text_results, strategy="max")

        if (
            not skip_rerank
            and rerank_cfg.enabled
            and self.reranker is not None
            and text_results
        ):
            rerank_top_k = fetch_k_text if is_math_modeling_query(question) else top_k
            text_results = self.reranker.rerank(question, text_results, top_k=rerank_top_k)
        else:
            text_results = text_results[:fetch_k_text]

        text_results = _boost_modeling_results(question, text_results)[:top_k]

        figure_results: list[dict] = []
        if self.img_retriever is not None:
            caption_hits: list[dict] = []
            if self.caption_retriever is not None:
                caption_hits = self.caption_retriever.retrieve(
                    question,
                    top_k=top_k_fig,
                    paper_id=paper_id,
                )

            figure_results = self.img_retriever.retrieve_with_caption_hits(
                question,
                caption_hits=caption_hits,
                top_k=top_k_fig,
                paper_id=paper_id,
                alpha=CFG.retriever.figure_alpha,
                beta=CFG.retriever.figure_beta,
            )
            figure_results = figure_results[:top_k_fig]

        for i, r in enumerate(text_results, 1):
            r["rank"] = i
        for i, r in enumerate(figure_results, 1):
            r["rank"] = i

        return {
            "text_results": text_results,
            "figure_results": figure_results,
        }

    def retrieve_model_metadata(
        self,
        question: str,
        *,
        top_k: int,
        paper_id: str | None = None,
    ) -> list[dict]:
        """Retrieve model evidence by structured chunk metadata.

        This complements vector search for mathematical modeling queries where
        objective/constraint/notation chunks must not be missed.
        """
        candidates: list[tuple[float, dict]] = []
        for chunk in self._ms.get_all():
            if chunk.get("modality") not in {"model_region", "math_model"}:
                continue
            if paper_id and chunk.get("paper_id") != paper_id:
                continue
            hit = {
                "rank": 0,
                "score": 0.0,
                "chunk_id": chunk.get("id", ""),
                "paper_id": chunk.get("paper_id", ""),
                "modality": chunk.get("modality", "text"),
                "page": chunk.get("page", -1),
                "content": chunk.get("content", ""),
                "section": chunk.get("section", ""),
            }
            for key in (
                "chunk_type",
                "source_chunk_ids",
                "evidence_refs",
                "model_card",
                "model_region_type",
                "model_region_score",
                "model_elements",
                "operator_hints",
                "domain_signals",
                "hhc_signals",
                "formula_signal_count",
                "model_evidence_score",
            ):
                if key in chunk:
                    hit[key] = chunk[key]
            score = _modeling_metadata_score(question, hit)
            if score <= 0:
                continue
            hit["metadata_score"] = round(score, 4)
            candidates.append((score, hit))

        candidates.sort(key=lambda item: item[0], reverse=True)
        results = [hit for _, hit in candidates[:top_k]]
        for rank, hit in enumerate(results, 1):
            hit["rank"] = rank
        return results

    def close(self) -> None:
        """释放当前 service 持有的可释放资源。"""
        if self._embedder is not None:
            try:
                self._embedder.release()
            except Exception:
                pass
        self.text_retriever = None
        self.caption_retriever = None
        self.reranker = None
        self.img_retriever = None
        self._ms = None
        self._embedder = None
        self._vs = None
        self._caption_vs = None
