"""Lightweight verifier for generated optimization models.

The verifier is intentionally conservative: it catches common structural
mistakes and copy/template drift, but it does not claim to prove mathematical
correctness.
"""

from __future__ import annotations

import re
from typing import Any


_COMPONENT_MARKERS: dict[str, tuple[str, ...]] = {
    "skill_matching": ("skill", "qualification", "q_c", "r_p", "技能", "资质"),
    "outsourcing": ("outsourcing", "outsource", "external", "rejection", "unserved", "o_p", "\\pi_p", "外包"),
    "priority_class": ("vip", "priority", "ordinary patient", "p^v", "\\omega_p", "优先"),
    "synchronization": ("synchronized", "synchronization", "simultaneous", "\\Delta_s", "同步"),
    "break_scheduling": ("lunch", "break", "rest", "z_{pc}", "午休", "休息"),
    "overtime": ("overtime", "O_c", "H_c", "加班"),
    "multi_center": ("multi-center", "healthcare center", "depot assignment", "g_{cd}", "多中心", "护理站"),
    "preference_matching": ("preference", "compatibility", "\\rho_{pc}", "偏好"),
    "time_window": ("time window", "b_p", "latest", "时间窗"),
    "capacity": ("capacity", "workload", "l_c", "u_c", "working time", "工作量"),
    "balance": ("workload balance", "fairness", "\\delta b", "imbalance", "均衡", "公平"),
    "open_route": ("open route", "do not return", "no return", "without returning", "开放路径", "不返回"),
}

_OPERATOR_REQUIRED_HINTS: dict[str, tuple[str, ...]] = {
    "assignment": ("assignment", "assign", "coverage", "y_", "分配"),
    "routing_flow": ("flow", "route", "arc", "x_", "inflow", "outflow", "路径"),
    "time_propagation": ("time propagation", "start time", "service start", "big-m", "arrival", "departure", "时间"),
    "waiting_time": ("waiting", "w_", "delay", "tardiness", "等待"),
    "time_window": ("time window", "earliest", "latest", "a_", "b_", "时间窗"),
    "capacity": ("capacity", "workload", "limit", "l_", "u_", "工作量"),
    "skill_matching": ("skill", "qualification", "q_", "r_", "技能"),
    "outsourcing": ("outsourcing", "external", "o_", "penalty", "外包"),
    "synchronization": ("synchronized", "synchronization", "simultaneous", "\\Delta", "同步"),
    "break_scheduling": ("lunch", "break", "rest", "z_", "午休", "休息"),
    "overtime": ("overtime", "O_", "H_", "加班"),
    "multi_center": ("center", "depot", "g_", "多中心", "护理站"),
    "preference_matching": ("preference", "compatibility", "\\rho", "偏好"),
    "open_route": ("open route", "terminate", "return", "end", "开放路径"),
    "balance": ("balance", "fairness", "imbalance", " b", "均衡"),
}


def verify_model(
    model: dict[str, Any] | None,
    *,
    harness_draft: dict[str, Any] | None = None,
    problem: str = "",
) -> dict[str, Any]:
    """Verify a generated or rendered model against Harness choices."""
    if not isinstance(model, dict):
        return {
            "status": "needs_review",
            "score": 0.0,
            "checks": [
                {
                    "name": "model_present",
                    "status": "fail",
                    "message": "No structured model is available to verify.",
                }
            ],
            "summary": "No structured model is available.",
        }

    checks: list[dict[str, Any]] = []
    selected = _selected_components(harness_draft)
    omitted = _omitted_components(harness_draft)
    model_text = _model_text(model)
    formulation_text = _formulation_text(model)

    checks.extend(_check_omitted_components(formulation_text, omitted))
    checks.extend(_check_required_components(model, selected))
    checks.extend(_check_objective_alignment(model, selected))
    checks.extend(_check_symbol_definitions(model))
    checks.extend(_check_hhc_minimum_structure(model, selected, problem))

    status = _aggregate_status(checks)
    score = _score(checks)
    return {
        "status": status,
        "score": score,
        "checks": checks,
        "summary": _summary(status, checks),
    }


def _selected_components(harness_draft: dict[str, Any] | None) -> set[str]:
    selector = (harness_draft or {}).get("component_selector") or []
    selected = {
        str(item.get("component", ""))
        for item in selector
        if isinstance(item, dict) and item.get("status") == "selected"
    }
    if selected:
        return selected
    spec = (harness_draft or {}).get("model_spec") or {}
    return {
        str(op.get("name", ""))
        for op in spec.get("operators", []) or []
        if isinstance(op, dict)
    }


def _omitted_components(harness_draft: dict[str, Any] | None) -> set[str]:
    selector = (harness_draft or {}).get("component_selector") or []
    omitted = {
        str(item.get("component", ""))
        for item in selector
        if isinstance(item, dict) and item.get("status") == "omitted"
    }
    spec = (harness_draft or {}).get("model_spec") or {}
    for item in spec.get("omitted_components", []) or []:
        if isinstance(item, dict) and item.get("component"):
            omitted.add(str(item["component"]))
    return omitted


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
    """Collect only actual formulation content, excluding explanatory omissions."""
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
    for item in model.get("constraints") or []:
        if isinstance(item, dict):
            parts.append(str(item.get("name", "")))
            parts.append(str(item.get("formula", "")))
            parts.append(str(item.get("description", "")))
        else:
            parts.append(str(item))
    return "\n".join(parts).lower()


def _check_omitted_components(model_text: str, omitted: set[str]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for component in sorted(omitted):
        markers = _COMPONENT_MARKERS.get(component, ())
        hits = [marker for marker in markers if marker.lower() in model_text]
        checks.append(
            {
                "name": f"omitted_{component}",
                "status": "fail" if hits else "pass",
                "message": (
                    f"Component '{component}' was omitted but appears in the model: {', '.join(hits[:5])}"
                    if hits
                    else f"Component '{component}' is not present in the model."
                ),
                "component": component,
                "hits": hits[:8],
            }
        )
    return checks


def _check_required_components(model: dict[str, Any], selected: set[str]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    constraints_text = _constraints_text(model)
    objective_text = str((model.get("objective") or {}).get("formula", "")).lower()
    all_text = f"{objective_text}\n{constraints_text}"
    for component in sorted(selected):
        hints = _OPERATOR_REQUIRED_HINTS.get(component)
        if not hints:
            continue
        hits = [hint for hint in hints if hint.lower() in all_text]
        required = component in {
            "assignment",
            "routing_flow",
            "time_propagation",
            "waiting_time",
            "time_window",
            "capacity",
            "skill_matching",
            "outsourcing",
        }
        status = "pass" if hits else ("fail" if required else "warn")
        checks.append(
            {
                "name": f"required_{component}",
                "status": status,
                "message": (
                    f"Selected component '{component}' is reflected by: {', '.join(hits[:5])}"
                    if hits
                    else f"Selected component '{component}' is not clearly reflected in objective or constraints."
                ),
                "component": component,
                "hits": hits[:8],
            }
        )
    return checks


def _check_objective_alignment(model: dict[str, Any], selected: set[str]) -> list[dict[str, Any]]:
    objective = model.get("objective")
    formula = str(objective.get("formula", "") if isinstance(objective, dict) else "").lower()
    checks: list[dict[str, Any]] = []
    objective_expectations = {
        "routing_flow": ("x_", "x_{", "travel", "t_", "t_{"),
        "waiting_time": ("w_", "w_{", "waiting"),
        "outsourcing": ("o_", "o_{", "\\pi", "outsourcing"),
        "balance": (" b", " b ", "\\delta b", "balance"),
    }
    for component, hints in objective_expectations.items():
        if component not in selected:
            continue
        hits = [hint for hint in hints if hint in formula]
        checks.append(
            {
                "name": f"objective_{component}",
                "status": "pass" if hits else "warn",
                "message": (
                    f"Objective contains evidence for '{component}': {', '.join(hits[:4])}"
                    if hits
                    else f"Objective may be missing a term for selected component '{component}'."
                ),
                "component": component,
                "hits": hits[:6],
            }
        )
    if not formula:
        checks.append(
            {
                "name": "objective_present",
                "status": "fail",
                "message": "Objective formula is missing.",
            }
        )
    return checks


def _check_symbol_definitions(model: dict[str, Any]) -> list[dict[str, Any]]:
    defined = _defined_symbols(model)
    used = _used_symbols(model)
    ignored = {
        "min",
        "max",
        "sum",
        "\\min",
        "\\max",
        "\\sum",
        "forall",
        "\\forall",
        "in",
        "\\in",
        "le",
        "\\le",
        "ge",
        "\\ge",
        "quad",
        "\\quad",
        "mathbb",
        "\\mathbb",
        "text",
        "\\text",
        "ne",
        "\\ne",
        "left",
        "\\left",
        "right",
        "\\right",
        "cap",
        "\\cap",
        "cup",
        "\\cup",
        "where",
        "optional",
        "scalarization",
        "alpha",
        "beta",
        "gamma",
    }
    undefined = sorted(symbol for symbol in used - defined if symbol.lower() not in ignored)
    # Single-letter indexes and common Greek weights are often bound by sums.
    undefined = [
        symbol
        for symbol in undefined
        if symbol not in {"i", "j", "p", "c", "k", "P", "C", "N", "A", "M"}
        and not re.fullmatch(r"Z_?\{?[0-9mM]+\}?", symbol)
        and not re.fullmatch(r"[a-zA-Z]", symbol)
        and _base_symbol(symbol) not in defined
        and _base_symbol(symbol).lower() not in ignored
    ]
    return [
        {
            "name": "symbol_definitions",
            "status": "pass" if not undefined else "warn",
            "message": (
                "No obvious undefined formula symbols were detected."
                if not undefined
                else "Potentially undefined symbols: " + ", ".join(undefined[:12])
            ),
            "symbols": undefined[:20],
        }
    ]


def _check_hhc_minimum_structure(
    model: dict[str, Any],
    selected: set[str],
    problem: str,
) -> list[dict[str, Any]]:
    text = f"{problem}\n{_model_text(model)}".lower()
    is_hhc = any(keyword in text for keyword in ("home health", "hhc", "caregiver", "patient", "护理员", "患者"))
    if not is_hhc:
        return []
    constraints = _constraints_text(model)
    checks: list[dict[str, Any]] = []
    if "routing_flow" in selected:
        has_route_binary = any(marker in text for marker in ("x_", "x_{", "arc", "route"))
        has_flow = any(
            marker in constraints
            for marker in (
                "flow",
                "inflow",
                "outflow",
                "incoming",
                "outgoing",
                "inbound",
                "outbound",
                "enter",
                "leave",
                "departure",
                "return",
                "route link",
            )
        )
        has_degree_link = (
            any(marker in constraints for marker in ("x_", "x_{"))
            and any(marker in constraints for marker in ("y_", "y_{"))
            and "=" in constraints
        )
        checks.append(
            {
                "name": "hhc_routing_structure",
                "status": "pass" if has_route_binary and (has_flow or has_degree_link) else "fail",
                "message": (
                    "HHC routing structure includes route variables and flow logic."
                    if has_route_binary and (has_flow or has_degree_link)
                    else "HHC routing is selected but route variables or flow-balance logic are not clear."
                ),
            }
        )
    if "assignment" in selected:
        has_assignment = any(marker in constraints for marker in ("assign", "assigned", "coverage", "exactly one", "y_"))
        checks.append(
            {
                "name": "hhc_assignment_structure",
                "status": "pass" if has_assignment else "fail",
                "message": (
                    "HHC assignment coverage is visible."
                    if has_assignment
                    else "HHC assignment is selected but assignment coverage is not clear."
                ),
            }
        )
    return checks


def _constraints_text(model: dict[str, Any]) -> str:
    constraints = model.get("constraints") or []
    if not isinstance(constraints, list):
        return ""
    parts: list[str] = []
    for item in constraints:
        if isinstance(item, dict):
            parts.append(str(item.get("name", "")))
            parts.append(str(item.get("formula", "")))
            parts.append(str(item.get("description", "")))
        else:
            parts.append(str(item))
    return "\n".join(parts).lower()


def _defined_symbols(model: dict[str, Any]) -> set[str]:
    defined: set[str] = set()
    for field in ("sets", "parameters", "decision_variables"):
        values = model.get(field) or []
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, dict):
                defined.update(_symbol_aliases(str(item.get("symbol", ""))))
    return defined


def _used_symbols(model: dict[str, Any]) -> set[str]:
    formulas: list[str] = []
    objective = model.get("objective")
    if isinstance(objective, dict):
        formulas.append(str(objective.get("formula", "")))
    for constraint in model.get("constraints") or []:
        if isinstance(constraint, dict):
            formulas.append(str(constraint.get("formula", "")))
    used: set[str] = set()
    for formula in formulas:
        formula = re.sub(r"\\text\s*\{[^{}]*\}", " ", formula)
        scrubbed = re.sub(
            r"\\(?:min|max|sum|forall|in|le|ge|ne|quad|mathbb|text|left|right|cdot|times|cap|cup)\b",
            " ",
            formula,
        )
        for match in re.findall(r"\\?[A-Za-z]+(?:_\{?[A-Za-z0-9']+\}?|\^\{?[A-Za-z0-9']+\}?)*", scrubbed):
            used.update(_symbol_aliases(match))
    return used


def _symbol_aliases(symbol: str) -> set[str]:
    cleaned = symbol.strip()
    if not cleaned:
        return set()
    aliases = {cleaned}
    aliases.add(cleaned.replace("\\", ""))
    base = re.split(r"[_^]", cleaned.replace("\\", ""), maxsplit=1)[0]
    if base:
        aliases.add(base)
    return {alias for alias in aliases if alias}


def _base_symbol(symbol: str) -> str:
    return re.split(r"[_^]", symbol.replace("\\", ""), maxsplit=1)[0].strip()


def _aggregate_status(checks: list[dict[str, Any]]) -> str:
    statuses = {str(check.get("status", "")) for check in checks}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def _score(checks: list[dict[str, Any]]) -> float:
    if not checks:
        return 1.0
    weights = {"pass": 1.0, "warn": 0.55, "fail": 0.0}
    total = sum(weights.get(str(check.get("status")), 0.35) for check in checks)
    return round(total / len(checks), 3)


def _summary(status: str, checks: list[dict[str, Any]]) -> str:
    counts = {"pass": 0, "warn": 0, "fail": 0}
    for check in checks:
        key = str(check.get("status"))
        if key in counts:
            counts[key] += 1
    if status == "pass":
        return f"Verifier passed {counts['pass']} checks."
    if status == "warn":
        return f"Verifier found {counts['warn']} warning-level issue(s)."
    return f"Verifier found {counts['fail']} failing issue(s) and {counts['warn']} warning(s)."
