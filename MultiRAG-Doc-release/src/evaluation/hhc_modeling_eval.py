"""Batch evaluation for HHC mathematical-model generation.

This is a lightweight regression test for the modeling pipeline. It evaluates
component selection, explicit omissions, rendered formula structure, and the
model verifier. By default it uses Harness formulas so the test stays fast and
does not spend full-generation tokens.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.modeling.model_builder import generate_harness_draft, generate_math_model
from src.query.service import QueryService

_DEFAULT_TESTSET = Path("database/testset/hhc_modeling_testset.json")
_DEFAULT_OUTPUT_DIR = Path("database/eval_results")


def run_hhc_modeling_eval(
    *,
    testset_path: Path | None = None,
    output_dir: Path | None = None,
    top_k: int = 6,
    paper_id: str | None = None,
    service: QueryService | None = None,
    full_llm: bool = False,
    max_tokens: int | None = None,
    limit_cases: int | None = None,
    case_ids: list[str] | None = None,
    fast_llm_eval: bool = False,
    config_tag: str = "hhc_modeling",
    save_report: bool = True,
) -> dict[str, Any]:
    """Run the HHC modeling regression set and save a JSON report."""
    testset_path = testset_path or _DEFAULT_TESTSET
    output_dir = output_dir or _DEFAULT_OUTPUT_DIR

    if not testset_path.exists():
        raise FileNotFoundError(f"HHC modeling testset not found: {testset_path}")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    with open(testset_path, encoding="utf-8") as f:
        testset: list[dict[str, Any]] = json.load(f)
    testset = _filter_testset(testset, limit_cases=limit_cases, case_ids=case_ids)
    if not testset:
        raise ValueError("HHC modeling testset is empty after filtering")

    if save_report:
        output_dir.mkdir(parents=True, exist_ok=True)
    mode = "full_llm" if full_llm else "harness_formula"
    print(f"[eval-hhc-modeling] cases={len(testset)} mode={mode} top_k={top_k} paper_id={paper_id or 'all'}")

    owns_service = service is None
    service = service or QueryService.from_disk()
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for idx, case in enumerate(testset, 1):
            record = _evaluate_case(
                case,
                service=service,
                top_k=top_k,
                paper_id=paper_id,
                full_llm=full_llm,
                max_tokens=max_tokens,
                fast_llm_eval=fast_llm_eval,
            )
            records.append(record)
            print(
                "[{idx:02d}/{total:02d}] {status:<4} "
                "{case_id:<36} selector={selector_score:.2f} verifier={verifier_status}"
                .format(
                    idx=idx,
                    total=len(testset),
                    status=record["status"],
                    case_id=str(record["id"])[:36],
                    selector_score=record["selector_score"],
                    verifier_status=record["verifier_status"],
                )
            )
            if record["status"] != "PASS":
                reasons = "; ".join(record.get("failure_reasons") or [])
                if reasons:
                    print(f"      {reasons}")
    finally:
        if owns_service:
            service.close()

    elapsed_seconds = round(time.perf_counter() - started, 2)
    summary = _summarize(records)
    summary["elapsed_seconds"] = elapsed_seconds

    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_tag": config_tag,
        "dataset": "hhc_modeling",
        "mode": mode,
        "top_k": top_k,
        "paper_id": paper_id,
        "limit_cases": limit_cases,
        "case_ids": case_ids or [],
        "fast_llm_eval": fast_llm_eval,
        "testset_path": str(testset_path),
        "summary": summary,
        "records": records,
    }

    output_path: Path | None = None
    if save_report:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{config_tag}_{mode}_{ts}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        report["output_path"] = str(output_path)

    print(
        "[eval-hhc-modeling] done "
        f"pass={summary['pass']} warn={summary['warn']} fail={summary['fail']} "
        f"selector_acc={summary['selector_accuracy']:.3f} "
        f"verifier_pass_rate={summary['verifier_pass_rate']:.3f} "
        f"elapsed={elapsed_seconds}s"
    )
    if output_path is not None:
        print(f"[eval-hhc-modeling] saved: {output_path}")
    return report


def _evaluate_case(
    case: dict[str, Any],
    *,
    service: QueryService,
    top_k: int,
    paper_id: str | None,
    full_llm: bool,
    max_tokens: int | None,
    fast_llm_eval: bool,
) -> dict[str, Any]:
    problem = str(case.get("problem") or "")
    started = time.perf_counter()
    error = ""
    result: dict[str, Any] | None = None
    try:
        if full_llm:
            effective_max_tokens = max_tokens
            generation_kwargs: dict[str, Any] = {}
            if fast_llm_eval:
                effective_max_tokens = max_tokens or 900
                generation_kwargs.update(
                    {
                        "generation_attempts": 1,
                        "reasoning_effort": "medium",
                        "timeout": 120.0,
                        "client_max_retries": 0,
                        "skip_llm_plan": True,
                        "skip_revision": True,
                    }
                )
            result = generate_math_model(
                problem,
                top_k=top_k,
                paper_id=paper_id,
                service=service,
                max_tokens=effective_max_tokens,
                **generation_kwargs,
            )
        else:
            result = generate_harness_draft(
                problem,
                top_k=top_k,
                paper_id=paper_id,
                service=service,
                render_formulas=True,
            )
    except Exception as exc:
        error = str(exc)

    elapsed_seconds = round(time.perf_counter() - started, 2)
    if result is None:
        error_status = "ERROR" if _is_external_generation_error(error) else "FAIL"
        return {
            "id": case.get("id", ""),
            "name": case.get("name", ""),
            "status": error_status,
            "failure_reasons": [f"generation_error: {error}"],
            "selector_score": 0.0,
            "verifier_status": "error",
            "verifier_score": 0.0,
            "error_type": "external_generation_error" if error_status == "ERROR" else "generation_error",
            "elapsed_seconds": elapsed_seconds,
        }

    harness = result.get("harness_draft") or {}
    selected = _components_by_status(harness, "selected")
    omitted = _components_by_status(harness, "omitted")
    selector_eval = _evaluate_selector(
        selected=selected,
        omitted=omitted,
        expected_selected=set(case.get("expected_selected") or []),
        expected_omitted=set(case.get("expected_omitted") or []),
        expected_not_selected=set(case.get("expected_not_selected") or []),
    )
    verifier = result.get("model_verification") or {}
    verifier_status = str(verifier.get("status") or "missing")
    verifier_score = float(verifier.get("score") or 0.0)
    quality = result.get("model_quality") or {}
    quality_scores = quality.get("scores") if isinstance(quality, dict) else {}
    warnings = _evaluation_warnings(result, fast_llm_eval=fast_llm_eval)

    failure_reasons: list[str] = []
    failure_reasons.extend(selector_eval["failure_reasons"])
    if verifier_status == "fail":
        failure_reasons.append("model_verifier failed")
    if verifier_status == "missing":
        failure_reasons.append("model_verifier missing")
    if result.get("parse_error"):
        failure_reasons.append(f"parse_error: {result['parse_error']}")

    status = "PASS"
    if failure_reasons:
        status = "FAIL"
    elif verifier_status == "warn" or warnings:
        status = "WARN"

    model = result.get("model") or {}
    constraints = model.get("constraints") if isinstance(model, dict) else []
    evidence_ids = _evidence_ids(result)
    required_eval = _evaluate_required_in_model(
        model=model,
        expected_selected=set(case.get("expected_selected") or []),
    )
    omitted_model_eval = _evaluate_omitted_in_model(
        model=model,
        expected_omitted=set(case.get("expected_omitted") or []),
        expected_not_selected=set(case.get("expected_not_selected") or []),
    )
    if full_llm:
        failure_reasons.extend(required_eval["failure_reasons"])
        failure_reasons.extend(omitted_model_eval["failure_reasons"])
        if not isinstance(model, dict) or not model:
            failure_reasons.append("model JSON missing")

    status = "PASS"
    if failure_reasons:
        status = "FAIL"
    elif verifier_status == "warn" or warnings:
        status = "WARN"

    return {
        "id": case.get("id", ""),
        "name": case.get("name", ""),
        "status": status,
        "failure_reasons": failure_reasons,
        "problem": problem,
        "selected_components": sorted(selected),
        "omitted_components": sorted(omitted),
        "expected_selected": sorted(set(case.get("expected_selected") or [])),
        "expected_omitted": sorted(set(case.get("expected_omitted") or [])),
        "expected_not_selected": sorted(set(case.get("expected_not_selected") or [])),
        "missing_selected": selector_eval["missing_selected"],
        "missing_omitted": selector_eval["missing_omitted"],
        "unexpected_selected": selector_eval["unexpected_selected"],
        "missing_model_components": required_eval["missing_model_components"],
        "unexpected_model_components": omitted_model_eval["unexpected_model_components"],
        "selector_score": selector_eval["selector_score"],
        "model_component_score": _component_model_score(
            required_eval=required_eval,
            omitted_model_eval=omitted_model_eval,
        ),
        "verifier_status": verifier_status,
        "verifier_score": verifier_score,
        "verifier_summary": verifier.get("summary", ""),
        "verifier_failed_checks": _checks_by_status(verifier, "fail"),
        "verifier_warn_checks": _checks_by_status(verifier, "warn"),
        "quality_status": quality.get("status", "") if isinstance(quality, dict) else "",
        "quality_overall_score": float(quality.get("overall_score") or 0.0) if isinstance(quality, dict) else 0.0,
        "quality_summary": quality.get("summary", "") if isinstance(quality, dict) else "",
        "quality_scores": quality_scores if isinstance(quality_scores, dict) else {},
        "quality_issues": list(quality.get("issues") or []) if isinstance(quality, dict) else [],
        "quality_strengths": list(quality.get("strengths") or []) if isinstance(quality, dict) else [],
        "generation_mode": result.get("generation_mode", ""),
        "skill": result.get("skill", ""),
        "warnings": warnings,
        "parse_error": result.get("parse_error", ""),
        "revision_note": result.get("revision_note", ""),
        "objective": (model.get("objective") or {}).get("formula", "") if isinstance(model, dict) else "",
        "constraint_count": len(constraints) if isinstance(constraints, list) else 0,
        "model_snapshot": _model_snapshot(model),
        "evidence_count": len(evidence_ids),
        "evidence_ids": evidence_ids,
        "elapsed_seconds": elapsed_seconds,
    }


def _filter_testset(
    testset: list[dict[str, Any]],
    *,
    limit_cases: int | None,
    case_ids: list[str] | None,
) -> list[dict[str, Any]]:
    filtered = testset
    requested_ids = {str(case_id) for case_id in (case_ids or []) if str(case_id).strip()}
    if requested_ids:
        filtered = [case for case in filtered if str(case.get("id", "")) in requested_ids]
    if limit_cases is not None and limit_cases > 0:
        filtered = filtered[:limit_cases]
    return filtered


def _evaluation_warnings(result: dict[str, Any], *, fast_llm_eval: bool) -> list[str]:
    warnings = list(result.get("warnings") or [])
    if fast_llm_eval:
        warnings = [
            warning
            for warning in warnings
            if str(warning).strip().lower() != "modeling_plan failed: skipped"
        ]
    return warnings


def _is_external_generation_error(error: str) -> bool:
    text = error.lower()
    return any(
        marker in text
        for marker in (
            "connection error",
            "timeout",
            "timed out",
            "rate limit",
            "quota",
            "server disconnected",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
        )
    )


def _components_by_status(harness_draft: dict[str, Any], status: str) -> set[str]:
    selector = harness_draft.get("component_selector") or []
    return {
        str(item.get("component", ""))
        for item in selector
        if isinstance(item, dict) and item.get("status") == status
    }


def _evaluate_required_in_model(
    *,
    model: dict[str, Any],
    expected_selected: set[str],
) -> dict[str, Any]:
    model_text = _model_formulation_text(model)
    missing = sorted(
        component
        for component in expected_selected
        if not _model_contains_component(model_text, component)
    )
    return {
        "missing_model_components": missing,
        "failure_reasons": (
            ["missing_model_components=" + ",".join(missing)] if missing else []
        ),
    }


def _evaluate_omitted_in_model(
    *,
    model: dict[str, Any],
    expected_omitted: set[str],
    expected_not_selected: set[str],
) -> dict[str, Any]:
    forbidden = expected_omitted | expected_not_selected
    model_text = _model_formulation_text(model)
    unexpected = sorted(
        component
        for component in forbidden
        if _model_contains_component(model_text, component)
    )
    return {
        "unexpected_model_components": unexpected,
        "failure_reasons": (
            ["unexpected_model_components=" + ",".join(unexpected)] if unexpected else []
        ),
    }


def _component_model_score(
    *,
    required_eval: dict[str, Any],
    omitted_model_eval: dict[str, Any],
) -> float:
    misses = len(required_eval.get("missing_model_components") or [])
    unexpected = len(omitted_model_eval.get("unexpected_model_components") or [])
    # The denominator is intentionally soft because this metric supplements
    # the Verifier; it is a quick drift signal, not a formal proof.
    return round(max(0.0, 1.0 - (misses + unexpected) / 8.0), 3)


def _model_formulation_text(model: dict[str, Any]) -> str:
    if not isinstance(model, dict):
        return ""
    parts: list[str] = []
    for field in ("sets", "parameters", "decision_variables"):
        values = model.get(field) or []
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, dict):
                parts.append(str(item.get("symbol", "")))
                parts.append(str(item.get("definition", "")))
                parts.append(str(item.get("domain", "")))
            else:
                parts.append(str(item))
    objective = model.get("objective")
    if isinstance(objective, dict):
        parts.append(str(objective.get("formula", "")))
        parts.append(str(objective.get("description", "")))
    constraints = model.get("constraints") or []
    if isinstance(constraints, list):
        for item in constraints:
            if isinstance(item, dict):
                parts.append(str(item.get("name", "")))
                parts.append(str(item.get("formula", "")))
                parts.append(str(item.get("description", "")))
            else:
                parts.append(str(item))
    return "\n".join(parts).lower()


_MODEL_COMPONENT_MARKERS: dict[str, tuple[str, ...]] = {
    "assignment": ("assign", "assignment", "coverage", "y_", "y_{"),
    "routing_flow": ("route", "routing", "flow", "arc", "x_", "x_{", "travel"),
    "time_propagation": ("time propagation", "start time", "service start", "big-m", "arrival", "departure", "t_"),
    "waiting_time": ("waiting", "delay", "tardiness", "w_", "w_{"),
    "time_window": ("time window", "latest", "b_", "b_{"),
    "capacity": ("capacity", "workload", "route duration", "l_", "l_{", "u_", "u_{"),
    "balance": ("workload balance", "fairness", "imbalance", "\\delta b", "max workload"),
    "skill_matching": ("skill", "qualification", "q_", "q_{", "r_", "r_{"),
    "outsourcing": ("outsourcing", "outsource", "external", "rejection", "unserved", "o_", "o_{", "\\pi"),
    "priority_class": ("vip", "priority", "ordinary patient", "\\omega", "priority-weighted"),
    "open_route": ("open route", "do not return", "no return", "without returning", "terminate"),
    "multi_objective": ("multi-objective", "weighted", "trade-off", "pareto"),
}


def _model_contains_component(model_text: str, component: str) -> bool:
    markers = _MODEL_COMPONENT_MARKERS.get(component, ())
    return any(marker in model_text for marker in markers)


def _model_snapshot(model: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(model, dict):
        return {}
    return {
        "sets": _symbol_snapshot(model.get("sets")),
        "parameters": _symbol_snapshot(model.get("parameters")),
        "decision_variables": _symbol_snapshot(model.get("decision_variables")),
        "component_applicability": [
            {
                "component": item.get("component", ""),
                "applicable": item.get("applicable"),
                "reason": item.get("reason", ""),
            }
            for item in model.get("component_applicability") or []
            if isinstance(item, dict)
        ],
        "constraints": [
            {
                "name": item.get("name", ""),
                "formula": item.get("formula", ""),
                "description": item.get("description", ""),
            }
            for item in model.get("constraints") or []
            if isinstance(item, dict)
        ][:12],
    }


def _symbol_snapshot(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            out.append(
                {
                    "symbol": str(item.get("symbol", "")),
                    "definition": str(item.get("definition", "")),
                }
            )
    return out


def _evaluate_selector(
    *,
    selected: set[str],
    omitted: set[str],
    expected_selected: set[str],
    expected_omitted: set[str],
    expected_not_selected: set[str],
) -> dict[str, Any]:
    missing_selected = sorted(expected_selected - selected)
    missing_omitted = sorted(expected_omitted - omitted)
    unexpected_selected = sorted(expected_not_selected & selected)
    total = len(expected_selected) + len(expected_omitted) + len(expected_not_selected)
    misses = len(missing_selected) + len(missing_omitted) + len(unexpected_selected)
    selector_score = 1.0 if total == 0 else max(0.0, 1.0 - misses / total)

    failure_reasons: list[str] = []
    if missing_selected:
        failure_reasons.append("missing_selected=" + ",".join(missing_selected))
    if missing_omitted:
        failure_reasons.append("missing_omitted=" + ",".join(missing_omitted))
    if unexpected_selected:
        failure_reasons.append("unexpected_selected=" + ",".join(unexpected_selected))

    return {
        "missing_selected": missing_selected,
        "missing_omitted": missing_omitted,
        "unexpected_selected": unexpected_selected,
        "selector_score": round(selector_score, 3),
        "failure_reasons": failure_reasons,
    }


def _checks_by_status(verifier: dict[str, Any], status: str) -> list[dict[str, Any]]:
    checks = verifier.get("checks") or []
    return [
        {
            "name": check.get("name", ""),
            "component": check.get("component", ""),
            "message": check.get("message", ""),
        }
        for check in checks
        if isinstance(check, dict) and check.get("status") == status
    ]


def _evidence_ids(result: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for item in (result.get("text_results") or []) + (result.get("figure_results") or []):
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("chunk_id") or item.get("figure_id") or "")
        if evidence_id:
            ids.append(evidence_id)
    return ids


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    pass_count = sum(1 for r in records if r.get("status") == "PASS")
    warn_count = sum(1 for r in records if r.get("status") == "WARN")
    fail_count = sum(1 for r in records if r.get("status") == "FAIL")
    error_count = sum(1 for r in records if r.get("status") == "ERROR")
    selector_accuracy = _average(float(r.get("selector_score") or 0.0) for r in records)
    verifier_pass_rate = (
        sum(1 for r in records if r.get("verifier_status") == "pass") / total
        if total
        else 0.0
    )
    verifier_avg_score = _average(float(r.get("verifier_score") or 0.0) for r in records)
    model_component_avg_score = _average(
        float(r.get("model_component_score") or 0.0) for r in records
    )
    quality_avg_score = _average(
        float(r.get("quality_overall_score") or 0.0) for r in records
    )
    return {
        "total": total,
        "pass": pass_count,
        "warn": warn_count,
        "fail": fail_count,
        "error": error_count,
        "selector_accuracy": round(selector_accuracy, 3),
        "verifier_pass_rate": round(verifier_pass_rate, 3),
        "verifier_avg_score": round(verifier_avg_score, 3),
        "model_component_avg_score": round(model_component_avg_score, 3),
        "quality_avg_score": round(quality_avg_score, 3),
    }


def _average(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)
