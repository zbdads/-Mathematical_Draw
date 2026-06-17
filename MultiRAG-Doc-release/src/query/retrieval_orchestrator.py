"""Retrieval Orchestrator：并行执行多个 sub-query，汇总候选证据池。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.config import CFG

if TYPE_CHECKING:
    from src.query.planner import QueryPlan
    from src.query.service import QueryService

_DEFAULT_PER_QUERY_TOP_K = 8


@dataclass
class EvidenceCandidate:
    chunk_id: str
    paper_id: str
    page: int | list[int]
    modality: str
    content: str
    section: str
    source_query: str
    retrieval_path: str
    retrieval_score: float
    figure_id: str | None = None
    image_path: str | None = None
    hit_paths: list[str] | None = None


def hit_to_candidate(hit: dict[str, Any], source_query: str) -> EvidenceCandidate:
    modality = hit.get("modality", "text")
    is_figure = modality == "figure"

    chunk_id = hit.get("chunk_id", "") or hit.get("figure_id", "")
    content = hit.get("content", "") or hit.get("caption", "")
    retrieval_path = "figure" if is_figure else "text"

    return EvidenceCandidate(
        chunk_id=chunk_id,
        paper_id=hit.get("paper_id", ""),
        page=hit.get("page", -1),
        modality=modality,
        content=content,
        section=hit.get("section", ""),
        source_query=source_query,
        retrieval_path=retrieval_path,
        retrieval_score=float(hit.get("score", 0.0)),
        figure_id=hit.get("figure_id"),
        image_path=hit.get("image_path"),
        hit_paths=None,
    )


class RetrievalOrchestrator:
    """并行执行 QueryPlan.sub_queries，汇总候选证据池。"""

    def retrieve(
        self,
        plan: "QueryPlan",
        service: "QueryService",
        paper_id: str | None,
        per_query_top_k: int = _DEFAULT_PER_QUERY_TOP_K,
    ) -> list[EvidenceCandidate]:
        sub_queries = plan.sub_queries
        candidates: list[EvidenceCandidate] = []

        def _flatten_hits(hits_by_track: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
            return hits_by_track.get("text_results", []) + hits_by_track.get("figure_results", [])

        if len(sub_queries) == 1:
            hits_by_track = service.retrieve_core(
                sub_queries[0],
                per_query_top_k,
                paper_id,
                skip_rerank=True,
                figure_top_k=CFG.retriever.top_k_fig,
            )
            hits = _flatten_hits(hits_by_track)
            candidates = [hit_to_candidate(h, sub_queries[0]) for h in hits]
        else:
            futures: dict = {}
            with ThreadPoolExecutor(max_workers=len(sub_queries)) as executor:
                for sq in sub_queries:
                    fut = executor.submit(
                        service.retrieve_core,
                        sq,
                        per_query_top_k,
                        paper_id,
                        True,
                        CFG.retriever.top_k_fig,
                    )
                    futures[fut] = sq

                sq_results: dict[str, list[EvidenceCandidate]] = {}
                for fut in as_completed(futures):
                    sq = futures[fut]
                    hits_by_track = fut.result()
                    hits = _flatten_hits(hits_by_track)
                    sq_results[sq] = [hit_to_candidate(h, sq) for h in hits]

            for sq in sub_queries:
                candidates.extend(sq_results.get(sq, []))

        return candidates

