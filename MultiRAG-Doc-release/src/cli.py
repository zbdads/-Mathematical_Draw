"""CLI 入口：ingest、ingest-all、query 三个子命令。

用法：
    python -m src.cli ingest --pdf database/pdf/RAG_2020.pdf --paper-id RAG_2020
    python -m src.cli ingest --pdf database/pdf/RAG_2020.pdf --paper-id RAG_2020 --overwrite
    python -m src.cli ingest-all --pdf-dir database/pdf
    python -m src.cli ingest-all --pdf-dir database/pdf --clean --multimodal
    python -m src.cli ingest-all --pdf-dir database/pdf --multimodal --caption-model
    python -m src.cli query --question "What is RAG?" --top-k 5
    python -m src.cli query --question "..." --paper-id RAG_2020
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from src.config import CFG
from src.generator.answer_formatter import render_answer
from src.pipeline.ingest import run_ingest
from src.pipeline.ingest_all import run_ingest_all
from src.pipeline.query import run_query
from src.evaluation.hhc_modeling_eval import run_hhc_modeling_eval
from src.evaluation.run_retrieval_eval import run_retrieval_eval

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ── ingest 子命令 ─────────────────────────────────────────────────────────
def cmd_ingest(args: argparse.Namespace) -> None:
    paper_id: str = args.paper_id or Path(args.pdf).stem
    run_ingest(
        Path(args.pdf),
        paper_id=paper_id,
        multimodal=getattr(args, "multimodal", False),
        overwrite=getattr(args, "overwrite", False),
        use_caption_model=getattr(args, "caption_model", False),
    )


# ── ingest-all 子命令 ─────────────────────────────────────────────────────
def cmd_ingest_all(args: argparse.Namespace) -> None:
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.is_dir():
        print(f"[error] 目录不存在：{pdf_dir}")
        return

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[warn] 未找到任何 PDF：{pdf_dir}")
        return

    print(f"[ingest-all] 发现 {len(pdfs)} 篇论文：")
    for p in pdfs:
        print(f"  {p.stem}")

    # --staged 模式：分阶段 pipeline
    if getattr(args, "staged", False):
        from src.pipeline.staged_ingest import run_staged_ingest_all

        force_stages = getattr(args, "force_stage", None) or []
        skip_caption = getattr(args, "skip_caption", False)
        clean_index = getattr(args, "clean_index", False)
        run_staged_ingest_all(
            pdf_paths=pdfs,
            clean=args.clean,
            clean_index=clean_index,
            force_stages=force_stages,
            skip_caption=skip_caption,
            skip_caption_index=getattr(args, "skip_caption_index", False),
            skip_image=getattr(args, "skip_image", False),
        )
        return

    outcome = run_ingest_all(
        pdf_paths=pdfs,
        clean=args.clean,
        multimodal=getattr(args, "multimodal", False),
        use_caption_model=getattr(args, "caption_model", False),
    )

    results = outcome["results"]
    skipped = outcome["skipped"]
    print("\n" + "=" * 60)
    print(f"{'paper_id':<25} {'pages':>6} {'text':>7} {'figure':>7} {'ntotal':>8}")
    print("-" * 60)
    for r in results:
        text_chunks = r.get("text_chunks", r.get("chunks", 0))
        figure_chunks = r.get("figure_chunks", 0)
        ntotal = r.get("text_index_total", r.get("ntotal", 0))
        print(
            f"{r['paper_id']:<25} {r['pages']:>6} "
            f"{text_chunks:>7} {figure_chunks:>7} {ntotal:>8}"
        )
    print("=" * 60)
    ntotal = results[-1].get("text_index_total", results[-1].get("ntotal", 0)) if results else 0
    print(f"[done] 入库 {len(results)} 篇，跳过 {len(skipped)} 篇，索引总向量数：{ntotal}")


# ── query 子命令 ──────────────────────────────────────────────────────────
def _resolve_top_k(raw_top_k: int | None) -> int:
    """Resolve CLI top-k with the config default and validate user input."""
    if raw_top_k is None:
        return CFG.retriever.top_k
    if raw_top_k < 1:
        raise ValueError("--top-k must be >= 1")
    return raw_top_k


def _print_results(results: list[dict], title: str = "Results") -> None:
    """打印检索结果列表（普通 query 与 agent query 共用）。"""
    print("\n" + "=" * 70)
    print(f"[{title}] {len(results)} hits")
    print("-" * 70)
    for r in results:
        modality = r.get("modality", "text")
        raw_page = r.get("page", -1)
        pages = raw_page if isinstance(raw_page, list) else [raw_page]
        page_str = "-".join(str(p) for p in pages)
        chunk_id = r.get("chunk_id", r.get("figure_id", ""))
        score = r.get("score", r.get("final_score", 0.0))
        score_parts = [f"score={score:.4f}"]
        if r.get("faiss_score") is not None:
            score_parts.append(f"faiss={r['faiss_score']:.4f}")
        if r.get("rerank_score") is not None:
            score_parts.append(f"rerank={r['rerank_score']:.4f}")
        if r.get("llm_relevance_score") is not None:
            score_parts.append(f"llm={r['llm_relevance_score']}/10")
        rank = r.get("rank", "?")
        print(
            f"Rank {rank!s:>2}  [{modality}]  {'  '.join(score_parts)}  "
            f"[page {page_str}]  {chunk_id}"
        )
        detail_parts: list[str] = []
        if r.get("chunk_type"):
            detail_parts.append(f"type={r['chunk_type']}")
        if r.get("model_elements"):
            detail_parts.append("elements=" + ",".join(str(v) for v in r["model_elements"]))
        if r.get("operator_hints"):
            detail_parts.append("operators=" + ",".join(str(v) for v in r["operator_hints"]))
        if r.get("hhc_signals"):
            detail_parts.append("hhc=" + ",".join(str(v) for v in r["hhc_signals"]))
        if r.get("model_evidence_score") is not None:
            detail_parts.append(f"model_score={r['model_evidence_score']}")
        if detail_parts:
            print("  " + "  ".join(detail_parts))
        preview = r.get("content", "")[:200].replace("\n", " ")
        print(f"  {preview} …")
        if modality == "figure" and r.get("image_path"):
            print(f"  image: {r['image_path']}")
        print()


def _print_dual_results(text_results: list[dict], figure_results: list[dict]) -> None:
    _print_results(text_results, title="Text Results")
    _print_results(figure_results, title="Figure Results")


def _make_stream_callback() -> tuple[Callable[[str], None], Callable[[], bool]]:
    started = False

    def _on_token(chunk: str) -> None:
        nonlocal started
        if not started:
            print("=" * 70)
            print("[answer-stream]")
            print()
            started = True
        print(chunk, end="", flush=True)

    return _on_token, lambda: started


def cmd_query(args: argparse.Namespace) -> None:
    paper_id: str | None = args.paper_id or None
    mode: str = getattr(args, "mode", "standard")
    debug: bool = getattr(args, "debug", False)

    if mode == "decompose":
        raw_top_k: int | None = args.top_k
        if raw_top_k is not None and raw_top_k < 1:
            print("[error] --top-k must be >= 1")
            return
        decompose_top_k_display = raw_top_k if raw_top_k is not None else CFG.reranker.decompose_top_k
        print(f"\n[query-decompose] 问题：{args.question!r}  top_k={decompose_top_k_display}  paper_id={paper_id or '全库'}")
        on_token, stream_started = _make_stream_callback()

        outcome = run_query(
            question=args.question,
            top_k=raw_top_k,
            paper_id=paper_id,
            generate_answer=args.generate,
            mode=mode,
            debug=debug,
            stream_callback=on_token if args.generate else None,
        )
        if stream_started():
            print()
        plan = outcome.get("plan")
        if plan is not None:
            print(f"\n[planner] intent: {plan.intent}")
            print(f"[planner] sub_queries: {plan.sub_queries}")

        if debug and outcome.get("debug"):
            _print_debug_info(outcome["debug"])
        else:
            _print_results(outcome["results"])
    elif mode == "agent":
        try:
            top_k = _resolve_top_k(args.top_k)
        except ValueError as exc:
            print(f"[error] {exc}")
            return
        print(f"\n[query-agent] 问题：{args.question!r}  top_k={top_k}  paper_id={paper_id or '全库'}")
        on_token, stream_started = _make_stream_callback()

        outcome = run_query(
            question=args.question,
            top_k=top_k,
            paper_id=paper_id,
            generate_answer=args.generate,
            mode=mode,
            debug=debug,
            stream_callback=on_token if args.generate else None,
        )
        if stream_started():
            print()
        _print_agent_outcome(outcome, debug_agent=debug)
        return

    else:
        try:
            top_k = _resolve_top_k(args.top_k)
        except ValueError as exc:
            print(f"[error] {exc}")
            return
        print(f"\n[query] 问题：{args.question!r}  top_k={top_k}  paper_id={paper_id or '全库'}")
        outcome = run_query(
            question=args.question,
            top_k=top_k,
            paper_id=paper_id,
            generate_answer=args.generate,
            mode=mode,
        )
        text_results = outcome.get("text_results")
        figure_results = outcome.get("figure_results")
        if isinstance(text_results, list) and isinstance(figure_results, list):
            _print_dual_results(text_results, figure_results)
        else:
            _print_results(outcome["results"])

    answer = outcome.get("answer")
    if answer is not None and not (mode == "decompose" and args.generate):
        print("=" * 70)
        guardrail_reason = outcome.get("guardrail_reason", "")
        if guardrail_reason:
            print("[guardrail] 触发拒答：" + guardrail_reason)
        print("\n" + render_answer(answer))
    elif mode == "decompose" and args.generate:
        guardrail_reason = outcome.get("guardrail_reason", "")
        if guardrail_reason:
            print(f"[guardrail] 触发拒答：{guardrail_reason}")


def _print_agent_outcome(outcome: dict, debug_agent: bool = False) -> None:
    """打印 agent loop 的结构化结果。"""
    from src.generator.answer_formatter import render_answer

    print("\n" + "=" * 70)
    print(f"[agent] terminate_reason: {outcome.get('terminate_reason', '')}")
    print(f"[agent] selected_evidence_count: {outcome.get('selected_evidence_count', 0)}")

    warnings = outcome.get("warnings", [])
    if warnings:
        print("[agent] warnings:")
        for w in warnings:
            print(f"  [{w['code']}] (step {w['step']}) {w['message']}")

    results = outcome.get("results", [])
    if results:
        _print_results(results, title="Agent Selected Evidence")

    answer = outcome.get("answer")
    if answer is not None:
        guardrail_reason = outcome.get("guardrail_reason", "")
        if guardrail_reason:
            print(f"[guardrail] 触发拒答：{guardrail_reason}")
        else:
            print("\n" + render_answer(answer))

    if debug_agent:
        trace = outcome.get("agent_trace", [])
        if trace:
            print("\n[agent-trace]")
            for t in trace:
                step = t.get("step", "?")
                action = t.get("action", t.get("step", ""))
                obs = t.get("observation", "")
                note = t.get("note", "")
                new_ev = t.get("new_evidence", "")
                parts = [f"  step={step}  action={action}"]
                if new_ev != "":
                    parts.append(f"  new_evidence={new_ev}")
                if obs:
                    parts.append(f"\n    obs: {obs}")
                if note:
                    parts.append(f"\n    note: {note}")
                print("".join(parts))


def _print_debug_info(debug: dict) -> None:
    """打印 --debug-decompose 模式的调试信息。"""
    import json

    plan_info = debug.get("plan", {})
    print("\n" + "=" * 70)
    print("[debug-decompose] QueryPlan")
    print(f"  normalized_question : {plan_info.get('normalized_question', '')}")
    print(f"  sub_queries         : {plan_info.get('sub_queries', [])}")
    print(f"  answer_mode         : {plan_info.get('answer_mode', '')}")
    rationale = plan_info.get("planner_rationale", {})
    print(f"  planner_rationale   : {json.dumps(rationale, ensure_ascii=False, indent=4)}")

    print("\n[debug-decompose] Sub-query 召回概览")
    for sq, overview in debug.get("sub_query_overviews", {}).items():
        print(f"  [{sq}]  candidates={overview['candidate_count']}")
        for hit in overview.get("top_hits", []):
            print(
                f"    chunk={hit['chunk_id']}  [{hit['modality']}]  "
                f"score={hit['retrieval_score']}  {hit['preview'][:80]}…"
            )

    print("\n[debug-decompose] Sub-query -> Selected Chunk IDs")
    for sq, chunk_ids in debug.get("sub_query_to_chunks", {}).items():
        chunk_str = ", ".join(f"[{cid}]" for cid in chunk_ids) if chunk_ids else "(none)"
        print(f"  [{sq}] -> {chunk_str}")

    print(f"\n[debug-decompose] Selected Evidence ({debug.get('selected_evidence_count', 0)} 条)")
    print("=" * 70)
    for se in debug.get("selected_evidence", []):
        pages = se["page"] if isinstance(se["page"], list) else [se["page"]]
        page_str = "-".join(str(p) for p in pages)
        print(
            f"  {se['chunk_id']}  [{se['modality']}]  page={page_str}  "
            f"llm={se['llm_relevance_score']}/10  final={se['final_score']:.2f}"
        )
        print(f"    source_query : {se['source_query']}")
        print(f"    matched_sq   : {se.get('matched_sub_queries', [])}")
        print()

    answer_prompt = debug.get("answer_prompt", [])
    if answer_prompt:
        print("[debug-decompose] Answer Prompt")
        print("=" * 70)
        for msg in answer_prompt:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            print(f"[{role}]")
            print(content)
            print()


# ── eval-retrieval 子命令 ─────────────────────────────────────────────────
def cmd_eval_retrieval(args: argparse.Namespace) -> None:
    run_retrieval_eval(
        peerqa_testset_path=Path(args.peerqa_testset) if args.peerqa_testset else None,
        index_dir=Path(args.index_dir) if args.index_dir else None,
        chunks_path=Path(args.chunks) if args.chunks else None,
        top_k=args.top_k,
        config_tag=args.config_tag,
        output_dir=Path(args.output) if args.output else None,
        intra_paper=args.intra_paper,
    )


def cmd_eval_hhc_modeling(args: argparse.Namespace) -> None:
    run_hhc_modeling_eval(
        testset_path=Path(args.testset) if args.testset else None,
        output_dir=Path(args.output) if args.output else None,
        top_k=args.top_k,
        paper_id=args.paper_id or None,
        full_llm=args.full_llm,
        max_tokens=args.max_tokens,
        limit_cases=args.limit_cases,
        case_ids=args.case_id,
        fast_llm_eval=args.fast_llm_eval,
        config_tag=args.config_tag,
    )


def cmd_build_model_cards(args: argparse.Namespace) -> None:
    from src.modeling.pipeline import build_model_cards_from_chunks

    result = build_model_cards_from_chunks(
        paper_id=args.paper_id or None,
        max_chunks=args.max_chunks,
        field_level=not args.legacy,
        rebuild_index=not args.no_index,
    )
    print("\n" + "=" * 70)
    print(f"[model-cards] generated: {len(result['cards'])}")
    for path in result["paths"]:
        print(f"  {path}")
    if result.get("index_stats"):
        print(f"[model-cards] index: {result['index_stats']}")


def cmd_build_model_regions(args: argparse.Namespace) -> None:
    from src.ingestion.model_region_pipeline import rebuild_model_region_chunks

    result = rebuild_model_region_chunks(paper_id=args.paper_id or None)
    print("\n" + "=" * 70)
    print(f"[model-regions] generated: {result['region_chunks']}")
    print(f"[model-regions] total model_region chunks: {result['model_region_chunks']}")
    print(f"[model-regions] total math_model chunks: {result['math_model_chunks']}")
    print(f"[model-regions] chunks_total: {result['chunks_total']}")
    print(f"[model-regions] text_index_total: {result['text_index_total']}")
    for pid, counts in result.get("per_paper_counts", {}).items():
        print(f"  {pid}: {counts}")


def cmd_generate_model(args: argparse.Namespace) -> None:
    from src.modeling.model_builder import generate_harness_draft, generate_math_model
    from src.modeling.platemo_codegen import generate_platemo_code as build_platemo_code

    if args.harness_only or args.harness_formulas:
        result = generate_harness_draft(
            args.problem,
            top_k=args.top_k,
            paper_id=args.paper_id or None,
            render_formulas=args.harness_formulas,
        )
        if args.platemo_code:
            code_generation = build_platemo_code(
                problem=args.problem,
                model=result.get("model"),
                harness_draft=result.get("harness_draft"),
                problem_spec=result.get("problem_spec"),
                platemo_root=args.platemo_root or None,
                class_name=args.platemo_class_name or None,
                write_file=not args.no_platemo_write,
            )
            result["code_generation"] = code_generation
            result["warnings"] = list(result.get("warnings") or []) + list(code_generation.get("warnings") or [])
    else:
        result = generate_math_model(
            args.problem,
            top_k=args.top_k,
            paper_id=args.paper_id or None,
            max_tokens=args.max_tokens,
            use_blueprint=args.blueprint,
            generate_platemo_code=args.platemo_code,
            platemo_root=args.platemo_root or None,
            platemo_class_name=args.platemo_class_name or None,
            write_platemo_file=args.platemo_code and not args.no_platemo_write,
        )
    print("\n" + "=" * 70)
    print(f"[generate-model] problem: {args.problem!r}")
    print(f"[generate-model] paper_id: {args.paper_id or '全库'}")
    print(f"[generate-model] skill: {result.get('skill', '')}")
    print(f"[generate-model] generation_mode: {result.get('generation_mode', '')}")
    print(f"[generate-model] evidence: text={len(result['text_results'])}, figure={len(result['figure_results'])}")
    if result.get("revision_note"):
        print(f"[generate-model] revision: {result['revision_note']}")
    if result.get("warnings"):
        print("[generate-model] warnings:")
        for warning in result["warnings"]:
            print(f"  - {warning}")
    if result.get("parse_error"):
        print(f"[generate-model] parse_error: {result['parse_error']}")

    if result.get("code_generation"):
        code = result["code_generation"]
        print("\n[platemo code]")
        print(f"  platform: {code.get('platform', '')} / {code.get('language', '')}")
        print(f"  class: {code.get('class_name', '')}")
        print(f"  written: {code.get('written', False)}")
        if code.get("target_path"):
            print(f"  target: {code.get('target_path')}")
        component_map = code.get("component_map") or {}
        implemented = component_map.get("implemented") or []
        unsupported = component_map.get("unsupported") or []
        if implemented:
            print("  implemented: " + ", ".join(str(item) for item in implemented))
        if unsupported:
            print("  unsupported: " + ", ".join(str(item) for item in unsupported))
        if args.print_platemo_code:
            print("\n[platemo matlab]")
            print(code.get("matlab_code", ""))

    if result.get("modeling_plan"):
        plan = result["modeling_plan"]
        print("\n[modeling plan]")
        print(f"  type: {plan.get('problem_type', '')}")
        print(f"  scope: {plan.get('modeling_scope', '')}")
        print(f"  size: {plan.get('expected_model_size', '')}")
        decisions = plan.get("component_decisions") or []
        if decisions:
            print("  decisions:")
            for item in decisions:
                print(
                    "    - {decision}: {component}".format(
                        decision=item.get("decision", ""),
                        component=item.get("component", ""),
                    )
                )
    elif result.get("plan_error"):
        print(f"\n[modeling plan] failed: {result['plan_error']}")

    if result.get("harness_draft"):
        draft = result["harness_draft"]
        spec = draft.get("model_spec") or {}
        validation = draft.get("validation") or {}
        print("\n[harness draft]")
        print(f"  mode: {draft.get('mode', '')}")
        print(f"  problem_type: {draft.get('problem_type', '')}")
        print(f"  operators: {', '.join(op.get('name', '') for op in spec.get('operators', []) if isinstance(op, dict))}")
        print(f"  sets: {len(spec.get('sets', []) or [])}")
        print(f"  parameters: {len(spec.get('parameters', []) or [])}")
        print(f"  variables: {len(spec.get('variables', []) or [])}")
        print(f"  constraint_groups: {len(spec.get('constraint_groups', []) or [])}")
        print(f"  validation: {validation.get('status', '')}")
        warnings = validation.get("warnings") or []
        if warnings:
            for warning in warnings:
                print(f"    - {warning}")

    if result.get("modeling_blueprint"):
        blueprint = result["modeling_blueprint"]
        print("\n[modeling blueprint]")
        print(f"  formulation_type: {blueprint.get('formulation_type', '')}")
        groups = blueprint.get("constraint_groups") or []
        if groups:
            print("  constraint_groups:")
            for item in groups:
                flag = "use" if item.get("include") else "omit"
                print(f"    - {flag}: {item.get('name', '')}")
        omitted = blueprint.get("omitted_components") or []
        if omitted:
            print("  omitted: " + "; ".join(str(item) for item in omitted))
    elif result.get("blueprint_error"):
        print(f"\n[modeling blueprint] failed: {result['blueprint_error']}")

    print("\n[retrieved evidence]")
    for item in result["text_results"] + result["figure_results"]:
        chunk_id = item.get("chunk_id") or item.get("figure_id") or ""
        modality = item.get("modality", "text")
        paper = item.get("paper_id", "")
        page = item.get("page", "")
        score = item.get("score", 0.0)
        try:
            score_str = f"{float(score):.4f}"
        except Exception:
            score_str = str(score)
        print(f"  [{chunk_id}] paper={paper} page={page} modality={modality} score={score_str}")

    print("\n[generated model json]")
    if result.get("model") is not None:
        print(json.dumps(result["model"], ensure_ascii=False, indent=2))
    else:
        print(result.get("raw_output", ""))


# ── 主入口 ────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="src.cli",
        description="MultiRAG-Doc CLI：PDF 入库与语义检索",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="解析 PDF 并建立向量索引")
    p_ingest.add_argument("--pdf", required=True, help="PDF 文件路径")
    p_ingest.add_argument("--paper-id", default="", help="论文标识符（默认取文件名）")
    p_ingest.add_argument(
        "--multimodal",
        action="store_true",
        help="同时用 Docling 提取图片、表格、行间公式",
    )
    p_ingest.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖重建索引与 chunks，避免重复入库",
    )
    p_ingest.add_argument(
        "--caption-model",
        action="store_true",
        help="启用 caption 模型为每张 figure 生成丰富描述（需 GPU）",
    )

    # ingest-all
    p_ingest_all = sub.add_parser("ingest-all", help="批量入库一个目录下所有 PDF")
    p_ingest_all.add_argument("--pdf-dir", required=True, help="PDF 所在目录")
    p_ingest_all.add_argument(
        "--clean",
        action="store_true",
        help="从头重建索引（删除现有 index/chunks）",
    )
    p_ingest_all.add_argument(
        "--multimodal",
        action="store_true",
        help="同时用 Docling 提取图片、表格、行间公式",
    )
    p_ingest_all.add_argument(
        "--caption-model",
        action="store_true",
        help="启用 caption 模型为每张 figure 生成丰富描述（需 GPU）",
    )
    p_ingest_all.add_argument(
        "--staged",
        action="store_true",
        help="启用分阶段 pipeline（按模型拆分阶段，降低 GPU 显存峰值）",
    )
    p_ingest_all.add_argument(
        "--force-stage",
        type=int,
        action="append",
        help="强制重跑指定阶段（可多选，如 --force-stage 1 --force-stage 2）",
    )
    p_ingest_all.add_argument(
        "--clean-index",
        action="store_true",
        help="仅清空 index/chunks，保留 staging 中间产物（配合 --staged）",
    )
    p_ingest_all.add_argument(
        "--skip-caption",
        action="store_true",
        help="跳过 caption 生成环节",
    )
    p_ingest_all.add_argument(
        "--skip-caption-index",
        action="store_true",
        help="跳过 figure caption 文本向量索引，节省 embedding API 额度",
    )
    p_ingest_all.add_argument(
        "--skip-image",
        action="store_true",
        help="跳过图像向量索引阶段（配合 --staged）",
    )

    # query
    p_query = sub.add_parser("query", help="语义检索")
    p_query.add_argument("--question", required=True, help="查询问题")
    p_query.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"返回结果数（默认 {CFG.retriever.top_k}）",
    )
    p_query.add_argument("--paper-id", default="", help="限定检索范围（可选）")
    p_query.add_argument("--generate", action="store_true", help="检索后调用 LLM 生成答案")
    p_query.add_argument(
        "--mode",
        choices=["standard", "decompose", "agent"],
        default="standard",
        help=(
            "执行策略：standard（直接检索）、"
            "decompose（QueryPlanner → 多 sub-query → BGE rerank）、"
            "agent（Agentic RAG Loop，迭代检索 + citation 校验）"
        ),
    )
    p_query.add_argument(
        "--debug",
        action="store_true",
        help="输出当前 mode 对应的调试信息；standard 模式下忽略",
    )
    # eval-retrieval
    p_eval = sub.add_parser("eval-retrieval", help="PeerQA 召回评测（Recall@k / MRR）")
    p_eval.add_argument(
        "--peerqa-testset",
        default="",
        dest="peerqa_testset",
        help="peerqa_eval_testset.json 路径（默认 database/testset/peerqa_eval_testset.json）",
    )
    p_eval.add_argument(
        "--index-dir",
        default="",
        dest="index_dir",
        help="PeerQA FAISS 索引目录（默认 database/peerqa_index/）",
    )
    p_eval.add_argument(
        "--chunks",
        default="",
        help="PeerQA all_chunks.json 路径（默认 database/peerqa_chunks/all_chunks.json）",
    )
    p_eval.add_argument("--top-k", type=int, default=20, help="召回数量（默认 20）")
    p_eval.add_argument("--config-tag", default="peerqa_baseline", help="结果标识（默认 peerqa_baseline）")
    p_eval.add_argument("--output", default="", help="结果输出目录（默认 database/eval_results/）")
    p_eval.add_argument(
        "--intra-paper",
        action="store_true",
        dest="intra_paper",
        help="论文内部召回：每条 query 只在来源论文的 chunks 内检索",
    )

    p_eval_hhc = sub.add_parser(
        "eval-hhc-modeling",
        help="HHC 自动建模回归评测：组件选择、公式骨架与 Verifier",
    )
    p_eval_hhc.add_argument(
        "--testset",
        default="",
        help="HHC 建模测试集路径（默认 database/testset/hhc_modeling_testset.json）",
    )
    p_eval_hhc.add_argument(
        "--output",
        default="",
        help="结果输出目录（默认 database/eval_results/）",
    )
    p_eval_hhc.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="每个 case 检索建模证据数量（默认 6）",
    )
    p_eval_hhc.add_argument("--paper-id", default="", help="限定参考论文（默认全库）")
    p_eval_hhc.add_argument(
        "--config-tag",
        default="hhc_modeling",
        help="结果文件标识（默认 hhc_modeling）",
    )
    p_eval_hhc.add_argument(
        "--full-llm",
        action="store_true",
        help="使用完整大模型生成再评测；默认只评测 Harness 公式以节省时间和额度",
    )
    p_eval_hhc.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="完整大模型生成的最大输出 token（仅 --full-llm 时使用）",
    )
    p_eval_hhc.add_argument(
        "--limit-cases",
        type=int,
        default=None,
        help="只评测前 N 个 case；调试完整生成时建议先设为 1",
    )
    p_eval_hhc.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="只评测指定 case id，可重复传入",
    )
    p_eval_hhc.add_argument(
        "--fast-llm-eval",
        action="store_true",
        help="完整生成评测快检模式：较短超时、较少 token、单次尝试",
    )

    p_model_cards = sub.add_parser(
        "build-model-cards",
        help="从已入库 chunks 抽取 LLMOPT 风格五元素模型卡并写入检索索引",
    )
    p_model_cards.add_argument("--paper-id", default="", help="仅处理指定论文（默认全库）")
    p_model_cards.add_argument(
        "--max-chunks",
        type=int,
        default=18,
        help="每篇论文送入模型卡抽取的候选 chunk 数（默认 18）",
    )
    p_model_cards.add_argument(
        "--no-index",
        action="store_true",
        help="只生成 database/model_cards/*.json，不追加到文本索引",
    )
    p_model_cards.add_argument(
        "--legacy",
        action="store_true",
        help="使用旧版一次性模型卡抽取，不执行字段级五元素抽取",
    )

    p_model_regions = sub.add_parser(
        "build-model-regions",
        help="从已入库 chunks 中补建数学模型区域 chunk，并重建文本索引",
    )
    p_model_regions.add_argument("--paper-id", default="", help="仅处理指定论文（默认全库）")

    p_generate_model = sub.add_parser(
        "generate-model",
        help="基于已入库模型知识库，为用户问题生成结构化数学建模草稿",
    )
    p_generate_model.add_argument("--problem", required=True, help="待建模的实际问题描述")
    p_generate_model.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="检索建模证据数量（默认 8）",
    )
    p_generate_model.add_argument("--paper-id", default="", help="限定参考论文（默认全库）")
    p_generate_model.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="覆盖本次模型生成最大输出 token（调试用）",
    )
    p_generate_model.add_argument(
        "--blueprint",
        action="store_true",
        help="启用额外 blueprint 规划阶段（更慢，适合深度调试）",
    )
    p_generate_model.add_argument(
        "--harness-only",
        action="store_true",
        help="只生成 Harness 草稿，不调用完整 LLM 生成",
    )
    p_generate_model.add_argument(
        "--harness-formulas",
        action="store_true",
        help="使用 Harness 草稿渲染一版可控公式模型，不调用完整 LLM 生成",
    )
    p_generate_model.add_argument(
        "--platemo-code",
        action="store_true",
        help="为生成的数学模型同步生成 PlatEMO MATLAB 问题类",
    )
    p_generate_model.add_argument(
        "--platemo-root",
        default="PlatEMO",
        help="PlatEMO 根目录（默认 ./PlatEMO，可按本机路径覆盖）",
    )
    p_generate_model.add_argument(
        "--platemo-class-name",
        default="",
        help="自定义生成的 MATLAB class 名称（默认按问题内容生成）",
    )
    p_generate_model.add_argument(
        "--no-platemo-write",
        action="store_true",
        help="只返回 MATLAB 代码，不写入 PlatEMO Problems 目录",
    )
    p_generate_model.add_argument(
        "--print-platemo-code",
        action="store_true",
        help="在 CLI 输出中打印完整 MATLAB 代码",
    )

    args = parser.parse_args()
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "ingest-all":
        cmd_ingest_all(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "eval-retrieval":
        cmd_eval_retrieval(args)
    elif args.command == "eval-hhc-modeling":
        cmd_eval_hhc_modeling(args)
    elif args.command == "build-model-cards":
        cmd_build_model_cards(args)
    elif args.command == "build-model-regions":
        cmd_build_model_regions(args)
    elif args.command == "generate-model":
        cmd_generate_model(args)


if __name__ == "__main__":
    main()
