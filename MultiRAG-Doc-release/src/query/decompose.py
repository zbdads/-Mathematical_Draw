"""Decompose query mode."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict

from src.config import CFG
from src.generator.answer_prompt_selector import parse_question_type_from_rationale
from src.query.answer_synthesizer import AnswerSynthesizer
from src.query.planner import QueryPlan, QueryPlanner
from src.query.retrieval_orchestrator import EvidenceCandidate, RetrievalOrchestrator
from src.query.service import QueryService
from src.retrieval.reranker import Reranker
from src.agent.evidence_scorer import ScoredEvidence


def _candidate_to_scored(c: EvidenceCandidate, final_score: float) -> ScoredEvidence:
    return ScoredEvidence(
        chunk_id=c.chunk_id,
        paper_id=c.paper_id,
        page=c.page,
        modality=c.modality,
        content=c.content,
        section=c.section,
        source_query=c.source_query,
        retrieval_path=c.retrieval_path,
        retrieval_score=c.retrieval_score,
        llm_relevance_score=-1,
        focused_summary="",
        final_score=final_score,
        figure_id=c.figure_id,
        image_path=c.image_path,
        hit_paths=c.hit_paths or [c.retrieval_path],
        matched_sub_queries=[c.source_query],
    )


def _bge_rerank_candidates(
    original_question: str,
    candidates: list[EvidenceCandidate],
    reranker: Reranker | None,
    decompose_top_k: int,
) -> list[ScoredEvidence]:
    text_cands = [c for c in candidates if c.modality != "figure"]
    fig_cands = [c for c in candidates if c.modality == "figure"]

    best_text: dict[str, EvidenceCandidate] = {}
    for c in text_cands:
        existing = best_text.get(c.chunk_id)
        if existing is None or c.retrieval_score > existing.retrieval_score:
            best_text[c.chunk_id] = c

    deduped_text = list(best_text.values())
    text_scored: list[ScoredEvidence] = []
    if deduped_text:
        if reranker is not None:
            chunk_dicts = [{"content": c.content, "chunk_id": c.chunk_id} for c in deduped_text]
            reranked = reranker.rerank(original_question, chunk_dicts, top_k=decompose_top_k)
            for r in reranked:
                c = best_text[r["chunk_id"]]
                text_scored.append(_candidate_to_scored(c, final_score=r["rerank_score"]))
        else:
            for c in sorted(deduped_text, key=lambda x: x.retrieval_score, reverse=True)[:decompose_top_k]:
                text_scored.append(_candidate_to_scored(c, final_score=c.retrieval_score))

    fig_scored = [_candidate_to_scored(c, final_score=c.retrieval_score) for c in fig_cands]
    return text_scored + fig_scored


def run_decompose_query(
    question: str,
    top_k: int | None,
    paper_id: str | None,
    generate_answer: bool,
    debug_decompose: bool = False,
    stream_callback: Callable[[str], None] | None = None,
    plan_callback: Callable[[dict], None] | None = None,
    retrieval_done_callback: Callable[[list[dict], list[dict]], None] | None = None,
    service: QueryService | None = None,
) -> dict:
    """执行 decompose query 流程。"""
    owns_service = service is None
    service = service or QueryService.from_disk()
    try:
        return _execute(
            question,
            top_k,
            paper_id,
            generate_answer,
            debug_decompose,
            service,
            stream_callback,
            plan_callback,
            retrieval_done_callback,
        )
    finally:
        if owns_service:
            service.close()


def _execute(
    question: str,
    top_k: int | None,
    paper_id: str | None,
    generate_answer: bool,
    debug_decompose: bool,
    service: QueryService,
    stream_callback: Callable[[str], None] | None,
    plan_callback: Callable[[dict], None] | None,
    retrieval_done_callback: Callable[[list[dict], list[dict]], None] | None,
) -> dict:
    rerank_cfg = CFG.reranker
    decompose_top_k = top_k if top_k is not None else rerank_cfg.decompose_top_k
    per_query_top_k = rerank_cfg.candidate_k

    planner = QueryPlanner()
    plan: QueryPlan = planner.plan(question)

    if plan_callback is not None:
        plan_callback(_plan_to_dict(plan))

    orchestrator = RetrievalOrchestrator()
    candidates = orchestrator.retrieve(plan, service, paper_id, per_query_top_k)

    scored_evidence = _bge_rerank_candidates(
        original_question=plan.original_question,
        candidates=candidates,
        reranker=service.reranker,
        decompose_top_k=decompose_top_k,
    )

    if retrieval_done_callback is not None:
        text_ev = [asdict(se) for se in scored_evidence if se.modality != "figure"]
        fig_ev = [asdict(se) for se in scored_evidence if se.modality == "figure"]
        retrieval_done_callback(text_ev, fig_ev)

    question_type = _extract_question_type(plan)

    synthesizer = AnswerSynthesizer()
    outcome = synthesizer.answer(
        question=plan.original_question,
        evidence=scored_evidence,
        generate_answer=generate_answer,
        question_type=question_type,
        stream_callback=stream_callback,
        answer_language=CFG.generator.answer_language,
    )

    outcome["plan"] = plan
    if debug_decompose:
        answer_messages = synthesizer.build_messages(
            question=plan.original_question,
            evidence=scored_evidence,
            question_type=question_type,
            answer_language=CFG.generator.answer_language,
        )
        outcome["debug"] = _build_debug_info(
            plan,
            candidates,
            scored_evidence,
            answer_messages,
        )
    else:
        outcome["debug"] = None
    return outcome


def _plan_to_dict(plan: QueryPlan) -> dict:
    try:
        return asdict(plan)
    except Exception:
        return {}


def _extract_question_type(plan: QueryPlan) -> str:
    return parse_question_type_from_rationale(plan.planner_rationale)


def _build_debug_info(plan: QueryPlan, candidates, scored_evidence, answer_messages) -> dict:
    sub_query_overviews = {}
    for sq in plan.sub_queries:
        sq_candidates = [c for c in candidates if c.source_query == sq]
        sub_query_overviews[sq] = {
            "candidate_count": len(sq_candidates),
            "top_hits": [
                {
                    "chunk_id": c.chunk_id,
                    "modality": c.modality,
                    "retrieval_score": c.retrieval_score,
                    "preview": c.content[:120].replace("\n", " "),
                }
                for c in sq_candidates[:5]
            ],
        }

    sub_query_to_chunks: dict[str, list[str]] = {}
    for se in scored_evidence:
        for sq in se.matched_sub_queries or [se.source_query]:
            sub_query_to_chunks.setdefault(sq, []).append(se.chunk_id)

    return {
        "plan": _plan_to_dict(plan),
        "sub_query_overviews": sub_query_overviews,
        "sub_query_to_chunks": sub_query_to_chunks,
        "answer_messages": answer_messages,
    }
