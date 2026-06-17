"""Rule-based quality rubric for generated optimization models.

The rubric is not a proof of mathematical correctness. It gives a stable,
explainable score that helps compare prompts, providers, and generation modes.
"""

from __future__ import annotations

import re
from typing import Any


_COMPONENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "assignment": ("assign", "assignment", "coverage", "y_", "y_{", "分配"),
    "routing_flow": ("route", "routing", "arc", "x_", "x_{", "incoming", "outgoing", "depot", "路径"),
    "time_propagation": ("start time", "arrival", "departure", "big-m", "service start", "t_", "时间"),
    "waiting_time": ("waiting", "wait", "delay", "w_", "w_{", "等待"),
    "time_window": ("time window", "earliest", "latest", "a_", "b_", "时间窗"),
    "capacity": ("capacity", "workload", "route duration", "l_", "u_", "工作量"),
    "skill_matching": ("skill", "qualification", "q_", "r_", "技能"),
    "outsourcing": ("outsourcing", "outsource", "external", "o_", "\\pi", "外包"),
    "priority_class": ("vip", "priority", "ordinary patient", "\\omega", "优先"),
    "synchronization": ("synchronized", "synchronization", "simultaneous", "\\Delta", "同步"),
    "break_scheduling": ("lunch", "break", "rest", "z_", "午休", "休息"),
    "overtime": ("overtime", "O_", "H_", "加班"),
    "multi_center": ("center", "depot", "g_", "多中心", "护理站"),
    "preference_matching": ("preference", "compatibility", "\\rho", "偏好"),
    "balance": ("balance", "fairness", "imbalance", "workload balance", "均衡", "公平"),
    "open_route": ("open route", "no return", "terminate", "without returning", "开放路径"),
    "multi_objective": ("multi-objective", "multi objective", "three-objective", "tri-objective", "pareto", "epsilon", "weighted", "三目标", "多目标"),
}

_CORE_HHC_COMPONENTS = {"assignment", "routing_flow"}


def evaluate_model_quality(
    model: dict[str, Any] | None,
    *,
    problem_spec: dict[str, Any] | None = None,
    harness_draft: dict[str, Any] | None = None,
    verifier: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a generated model with an HHC-oriented quality rubric."""
    if not isinstance(model, dict):
        return {
            "overall_score": 0.0,
            "status": "fail",
            "summary": "No structured model is available for quality scoring.",
            "scores": {},
            "issues": ["model JSON missing"],
            "strengths": [],
        }

    spec = problem_spec or {}
    selected = _selected_components(spec, harness_draft)
    forbidden = _forbidden_components(spec, harness_draft)
    model_text = _model_text(model)
    formulation_text = _formulation_text(model)
    constraints = _constraints(model)

    scores: dict[str, float] = {}
    issues: list[str] = []
    strengths: list[str] = []

    scores["structure_score"] = _score_structure(model, issues, strengths)
    scores["objective_score"] = _score_objective(model, selected, issues, strengths)
    scores["constraint_score"] = _score_constraints(constraints, selected, issues, strengths)
    scores["component_coverage_score"] = _score_component_coverage(
        model_text,
        selected,
        issues,
        strengths,
    )
    scores["boundary_control_score"] = _score_boundary_control(
        formulation_text,
        forbidden,
        issues,
        strengths,
    )
    scores["symbol_score"] = _score_symbols(model, verifier, issues, strengths)
    scores["academic_depth_score"] = _score_academic_depth(
        model,
        selected,
        spec,
        issues,
        strengths,
    )

    weights = {
        "structure_score": 0.15,
        "objective_score": 0.14,
        "constraint_score": 0.20,
        "component_coverage_score": 0.18,
        "boundary_control_score": 0.14,
        "symbol_score": 0.09,
        "academic_depth_score": 0.10,
    }
    overall = sum(scores[key] * weight for key, weight in weights.items())
    overall = round(max(0.0, min(1.0, overall)), 3)
    return {
        "overall_score": overall,
        "status": _status(overall),
        "summary": _summary(overall, issues),
        "scores": {key: round(value, 3) for key, value in scores.items()},
        "issues": _dedupe(issues)[:12],
        "strengths": _dedupe(strengths)[:12],
        "selected_components": sorted(selected),
        "forbidden_components": sorted(forbidden),
    }


def _score_structure(model: dict[str, Any], issues: list[str], strengths: list[str]) -> float:
    sets = _list_field(model, "sets")
    params = _list_field(model, "parameters")
    variables = _list_field(model, "decision_variables")
    constraints = _constraints(model)
    objective = model.get("objective") if isinstance(model.get("objective"), dict) else {}
    score = 0.0
    if len(sets) >= 3:
        score += 0.2
        strengths.append("defines core entity sets")
    else:
        issues.append("too few sets for a routing/scheduling model")
    if len(params) >= 4:
        score += 0.2
        strengths.append("defines several input parameters")
    else:
        issues.append("parameter list is thin")
    if len(variables) >= 3:
        score += 0.2
        strengths.append("defines assignment/routing/timing variables")
    else:
        issues.append("decision variable list is thin")
    if objective and objective.get("formula"):
        score += 0.2
    else:
        issues.append("objective formula missing")
    if len(constraints) >= 5:
        score += 0.2
    else:
        issues.append("too few constraints for a full HHC model")
    return score


def _score_objective(
    model: dict[str, Any],
    selected: set[str],
    issues: list[str],
    strengths: list[str],
) -> float:
    objective = model.get("objective") if isinstance(model.get("objective"), dict) else {}
    text = f"{objective.get('formula', '')}\n{objective.get('description', '')}".lower()
    if not text.strip():
        issues.append("objective is empty")
        return 0.0
    expected = []
    if "routing_flow" in selected:
        expected.append(("travel/routing cost", ("travel", "t_", "t_{", "x_", "x_{")))
    if "waiting_time" in selected:
        expected.append(("waiting time", ("waiting", "w_", "w_{", "delay")))
    if "outsourcing" in selected:
        expected.append(("outsourcing penalty", ("outsourcing", "outsource", "o_", "\\pi")))
    if "balance" in selected:
        expected.append(("workload balance", ("balance", "fairness", "imbalance", "workload")))
    if "priority_class" in selected:
        expected.append(("priority service", ("priority", "vip", "\\omega")))
    if "overtime" in selected:
        expected.append(("overtime penalty", ("overtime", "o_", "h_", "加班")))
    if "preference_matching" in selected:
        expected.append(("preference satisfaction", ("preference", "compatibility", "\\rho", "偏好")))
    if "multi_objective" in selected:
        expected.append(("multi-objective structure", ("multi-objective", "three-objective", "tri-objective", "pareto", "epsilon", "weighted", "z_1", "z_2", "z_3", "min \\left")))
    if not expected:
        return 0.85
    hits = 0
    for label, markers in expected:
        if any(marker in text for marker in markers):
            hits += 1
            strengths.append(f"objective covers {label}")
        else:
            issues.append(f"objective may miss {label}")
    return hits / len(expected)


def _score_constraints(
    constraints: list[dict[str, Any]],
    selected: set[str],
    issues: list[str],
    strengths: list[str],
) -> float:
    if not constraints:
        issues.append("constraints are missing")
        return 0.0
    count_score = min(1.0, len(constraints) / _target_constraint_count(selected))
    text = _constraints_text(constraints)
    logic_checks = [
        ("assignment coverage", ("assign", "coverage", "exactly one", "sum_{c", "y_")),
        ("routing degree/flow", ("incoming", "outgoing", "flow", "depot", "x_")),
        ("timing propagation", ("big-m", "start", "arrival", "service", "t_")),
    ]
    if "waiting_time" in selected:
        logic_checks.append(("waiting definition", ("waiting", "w_", "delay")))
    if "time_window" in selected:
        logic_checks.append(("time window feasibility", ("time window", "earliest", "latest", "a_", "b_")))
    if "capacity" in selected:
        logic_checks.append(("workload/capacity bound", ("capacity", "workload", "limit", "l_", "u_")))
    if "skill_matching" in selected:
        logic_checks.append(("skill compatibility", ("skill", "qualification", "q_", "r_")))
    if "outsourcing" in selected:
        logic_checks.append(("outsourcing logic", ("outsourcing", "outsource", "o_")))
    if "synchronization" in selected:
        logic_checks.append(("synchronization logic", ("synchronized", "simultaneous", "\\delta", "\\Delta", "同步")))
    if "break_scheduling" in selected:
        logic_checks.append(("break scheduling", ("lunch", "break", "rest", "z_", "午休")))
    if "overtime" in selected:
        logic_checks.append(("overtime definition", ("overtime", "o_", "h_", "加班")))
    if "multi_center" in selected:
        logic_checks.append(("multi-center assignment", ("center", "depot", "g_", "护理站")))
    if "preference_matching" in selected:
        logic_checks.append(("preference matching", ("preference", "compatibility", "\\rho", "偏好")))
    if "multi_objective" in selected:
        logic_checks.append(("multi-objective handling", ("multi-objective", "three-objective", "tri-objective", "pareto", "epsilon", "weighted", "lexicographic", "z_1", "z_2", "z_3")))
    hits = 0
    for label, markers in logic_checks:
        if any(marker in text for marker in markers):
            hits += 1
            strengths.append(f"constraints include {label}")
        else:
            issues.append(f"constraints may miss {label}")
    logic_score = hits / len(logic_checks)
    return 0.45 * count_score + 0.55 * logic_score


def _score_component_coverage(
    model_text: str,
    selected: set[str],
    issues: list[str],
    strengths: list[str],
) -> float:
    required = selected or _CORE_HHC_COMPONENTS
    hits = 0
    for component in sorted(required):
        markers = _COMPONENT_KEYWORDS.get(component, ())
        if not markers:
            continue
        if any(marker in model_text for marker in markers):
            hits += 1
            strengths.append(f"selected component present: {component}")
        else:
            issues.append(f"selected component not clearly present: {component}")
    denominator = len([c for c in required if c in _COMPONENT_KEYWORDS])
    return 1.0 if denominator == 0 else hits / denominator


def _score_boundary_control(
    model_text: str,
    forbidden: set[str],
    issues: list[str],
    strengths: list[str],
) -> float:
    if not forbidden:
        return 1.0
    violations = 0
    for component in sorted(forbidden):
        if _forbidden_component_present(component, model_text):
            violations += 1
            issues.append(f"forbidden component appears: {component}")
    if violations == 0:
        strengths.append("does not introduce forbidden components")
    return max(0.0, 1.0 - violations / max(1, len(forbidden)))


def _forbidden_component_present(component: str, formulation_text: str) -> bool:
    if component == "balance":
        return any(marker in formulation_text for marker in ("workload balance", "fairness", "imbalance", "\\delta", " b "))
    if component == "time_window":
        return any(
            marker in formulation_text
            for marker in ("time window", "latest", "due time", "b_", "b_{", "l_p", "l_{")
        )
    if component == "capacity":
        return any(
            marker in formulation_text
            for marker in ("capacity", "workload limit", "route duration limit", "l_c", "u_c", "u_{c}")
        )
    markers = _COMPONENT_KEYWORDS.get(component, ())
    return any(marker in formulation_text for marker in markers)


def _score_symbols(
    model: dict[str, Any],
    verifier: dict[str, Any] | None,
    issues: list[str],
    strengths: list[str],
) -> float:
    if verifier and verifier.get("status") == "pass":
        strengths.append("verifier found no obvious symbol issues")
        return 1.0
    warn_symbols = []
    for check in (verifier or {}).get("checks") or []:
        if isinstance(check, dict) and check.get("name") == "symbol_definitions":
            warn_symbols = list(check.get("symbols") or [])
            break
    if warn_symbols:
        issues.append("potential undefined symbols: " + ", ".join(str(s) for s in warn_symbols[:6]))
        return 0.65
    variables = _list_field(model, "decision_variables")
    if all(isinstance(item, dict) and item.get("symbol") for item in variables):
        return 0.85
    issues.append("some variables lack explicit symbols")
    return 0.6


def _score_academic_depth(
    model: dict[str, Any],
    selected: set[str],
    spec: dict[str, Any],
    issues: list[str],
    strengths: list[str],
) -> float:
    constraints = _constraints(model)
    variables = _list_field(model, "decision_variables")
    params = _list_field(model, "parameters")
    score = 0.0
    if len(constraints) >= 8:
        score += 0.3
        strengths.append("constraint count is closer to paper-level depth")
    elif spec.get("generation_depth") == "paper_level":
        issues.append("paper-level mode produced relatively few constraints")
    if len(variables) >= 4:
        score += 0.2
    else:
        issues.append("paper-level model could use richer decision variables")
    if len(params) >= 6:
        score += 0.2
    else:
        issues.append("parameterization is still sparse")
    optional_selected = selected - _CORE_HHC_COMPONENTS - {"time_propagation", "waiting_time"}
    if optional_selected:
        score += 0.15
    elif spec.get("generation_depth") == "paper_level" and "multi_objective" not in selected:
        issues.append("no advanced HHC components selected; prompt may be underspecified")
    if _has_constraint_names(constraints):
        score += 0.15
    else:
        issues.append("constraints should have clearer semantic names")
    return min(1.0, score)


def _selected_components(
    spec: dict[str, Any],
    harness_draft: dict[str, Any] | None,
) -> set[str]:
    values = spec.get("selected_components") or []
    selected = {str(value) for value in values if value}
    if selected:
        return selected
    selector = (harness_draft or {}).get("component_selector") or []
    return {
        str(item.get("component", ""))
        for item in selector
        if isinstance(item, dict) and item.get("status") == "selected"
    }


def _forbidden_components(
    spec: dict[str, Any],
    harness_draft: dict[str, Any] | None,
) -> set[str]:
    values = spec.get("forbidden_components") or []
    forbidden = {str(value) for value in values if value}
    if forbidden:
        return forbidden
    selector = (harness_draft or {}).get("component_selector") or []
    return {
        str(item.get("component", ""))
        for item in selector
        if isinstance(item, dict) and item.get("status") == "omitted"
    }


def _target_constraint_count(selected: set[str]) -> int:
    base = 6
    advanced = selected & {
        "time_window",
        "capacity",
        "skill_matching",
        "outsourcing",
        "priority_class",
        "balance",
        "synchronization",
        "break_scheduling",
        "overtime",
        "multi_center",
        "preference_matching",
        "open_route",
        "multi_objective",
    }
    return base + len(advanced)


def _list_field(model: dict[str, Any], field: str) -> list[Any]:
    value = model.get(field) or []
    return value if isinstance(value, list) else []


def _constraints(model: dict[str, Any]) -> list[dict[str, Any]]:
    values = model.get("constraints") or []
    return [item for item in values if isinstance(item, dict)] if isinstance(values, list) else []


def _constraints_text(constraints: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in constraints:
        parts.append(str(item.get("name", "")))
        parts.append(str(item.get("formula", "")))
        parts.append(str(item.get("description", "")))
    return "\n".join(parts).lower()


def _model_text(model: dict[str, Any]) -> str:
    parts: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif value is not None:
            parts.append(str(value))

    visit(model)
    return "\n".join(parts).lower()


def _formulation_text(model: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in ("sets", "parameters", "decision_variables"):
        for item in _list_field(model, field):
            if isinstance(item, dict):
                parts.append(str(item.get("symbol", "")))
                parts.append(str(item.get("definition", "")))
                parts.append(str(item.get("domain", "")))
            else:
                parts.append(str(item))
    objective = model.get("objective") if isinstance(model.get("objective"), dict) else {}
    parts.append(str(objective.get("formula", "")))
    parts.append(str(objective.get("description", "")))
    for item in _constraints(model):
        parts.append(str(item.get("name", "")))
        parts.append(str(item.get("formula", "")))
        parts.append(str(item.get("description", "")))
    return "\n".join(parts).lower()


def _has_constraint_names(constraints: list[dict[str, Any]]) -> bool:
    names = [str(item.get("name", "")).strip() for item in constraints]
    meaningful = [name for name in names if len(re.sub(r"[^A-Za-z]", "", name)) >= 4]
    return len(meaningful) >= max(1, len(constraints) // 2)


def _status(score: float) -> str:
    if score >= 0.8:
        return "good"
    if score >= 0.6:
        return "needs_review"
    return "weak"


def _summary(score: float, issues: list[str]) -> str:
    if score >= 0.8:
        return "The model is structurally solid under the rule-based rubric."
    if score >= 0.6:
        return "The model is usable but has quality gaps that should be reviewed."
    if issues:
        return "The model is weak under the rubric: " + issues[0]
    return "The model is weak under the rubric."


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = " ".join(str(value).strip().split())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out
