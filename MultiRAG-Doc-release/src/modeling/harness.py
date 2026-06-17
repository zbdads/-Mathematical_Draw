"""Lightweight modeling harness for structured optimization drafts.

The harness builds an intermediate SymbolPlan / ModelSpec from the user
problem, selected skill, and retrieved model-aware evidence. It is deliberately
not a fixed HHC template: components are selected only when the problem text or
evidence indicates that the structure is relevant.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.modeling.skills import ModelingSkill


@dataclass
class SymbolItem:
    symbol: str
    role: str
    description: str
    source: str = "harness"


@dataclass
class OperatorItem:
    name: str
    purpose: str
    evidence_chunk_ids: list[str] = field(default_factory=list)
    status: str = "selected"


@dataclass
class SymbolPlan:
    sets: list[SymbolItem] = field(default_factory=list)
    parameters: list[SymbolItem] = field(default_factory=list)
    variables: list[SymbolItem] = field(default_factory=list)


@dataclass
class ModelSpec:
    problem_type: str
    operators: list[OperatorItem] = field(default_factory=list)
    sets: list[SymbolItem] = field(default_factory=list)
    parameters: list[SymbolItem] = field(default_factory=list)
    variables: list[SymbolItem] = field(default_factory=list)
    objective_terms: list[dict[str, Any]] = field(default_factory=list)
    constraint_groups: list[dict[str, Any]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    omitted_components: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class HarnessDraft:
    mode: str
    problem_type: str
    component_selector: list[dict[str, Any]]
    symbol_plan: SymbolPlan
    model_spec: ModelSpec
    validation: dict[str, Any]
    evidence_summary: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_HHC_KEYWORDS = (
    "home health care",
    "home healthcare",
    "hhc",
    "caregiver",
    "caregivers",
    "patient",
    "patients",
    "nurse",
    "nurses",
    "visit",
    "visits",
    "居家医疗",
    "家庭医疗",
    "护理员",
    "患者",
)

_COMPONENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "assignment": ("assign", "assignment", "allocate", "allocation", "分配"),
    "routing_flow": ("route", "routing", "travel", "path", "arc", "depot", "路径", "路线"),
    "time_window": ("time window", "earliest", "latest", "ready time", "due time", "时间窗"),
    "time_propagation": ("arrival", "departure", "start time", "service duration", "到达", "离开"),
    "waiting_time": ("waiting", "waiting time", "delay", "tardiness", "等待", "延迟"),
    "capacity": ("capacity", "workload", "working time", "route duration", "容量", "工作量"),
    "skill_matching": ("skill", "qualification", "requirement", "技能", "资质"),
    "outsourcing": ("outsourcing", "outsource", "external service", "rejection", "unserved", "外包"),
    "priority_class": ("vip", "priority", "ordinary patient", "优先级"),
    "synchronization": ("synchronized", "synchronization", "simultaneous", "paired visit", "同步"),
    "break_scheduling": ("lunch", "break", "rest", "午休", "休息"),
    "overtime": ("overtime", "extra working", "late work", "加班"),
    "multi_center": ("multi-center", "multi center", "multiple centers", "healthcare center", "depot assignment", "多中心", "护理站"),
    "preference_matching": ("preference", "gender preference", "patient preference", "caregiver preference", "偏好"),
    "multi_objective": (
        "multi-objective",
        "multi objective",
        "three-objective",
        "tri-objective",
        "3-objective",
        "weighted",
        "trade-off",
        "pareto",
        "多目标",
        "三目标",
        "3目标",
        "三个目标",
    ),
    "open_route": ("open route", "do not return", "no return", "without returning", "开放路径", "不返回"),
    "balance": ("balance", "balanced", "fairness", "workload balance", "均衡", "公平"),
}

_NEGATION_PATTERNS: dict[str, tuple[str, ...]] = {
    "skill_matching": (
        "do not include skill",
        "no skill",
        "without skill",
        "do not consider skill",
        "do not include caregiver skill",
        "不要技能",
        "不考虑技能",
    ),
    "outsourcing": (
        "do not include outsourcing",
        "no outsourcing",
        "without outsourcing",
        "do not consider outsourcing",
        "不要外包",
        "不考虑外包",
    ),
    "priority_class": (
        "do not include vip",
        "no vip",
        "without vip",
        "do not consider vip",
        "do not include priority",
        "不要vip",
        "不考虑vip",
    ),
    "synchronization": ("do not include synchronization", "no synchronized visits", "不考虑同步"),
    "break_scheduling": ("do not include lunch", "no lunch break", "without break", "不考虑午休", "不考虑休息"),
    "overtime": ("do not include overtime", "no overtime", "without overtime", "不考虑加班"),
    "multi_center": ("single depot", "single center", "one center", "单中心", "单护理站"),
    "preference_matching": ("do not include preference", "no preference", "without preference", "不考虑偏好"),
    "time_window": ("do not include time window", "no time window", "without time window", "不考虑时间窗"),
    "capacity": ("do not include workload", "no workload", "without workload", "不考虑工作量"),
    "open_route": ("closed route", "must return", "闭合路径", "返回仓库"),
    "balance": ("do not include balance", "no balance", "without balance", "不考虑均衡", "不考虑公平"),
}


def build_harness_draft(
    problem: str,
    evidence: list[dict[str, Any]],
    *,
    skill: ModelingSkill,
    modeling_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured modeling draft without long-form LLM generation."""
    problem_type = _infer_problem_type(problem, skill)
    selected_ops = _select_operators(problem, evidence, modeling_plan=modeling_plan)
    omitted_components = _omitted_component_details(problem.lower())
    component_selector = _build_component_selector(
        problem,
        evidence,
        selected_ops=selected_ops,
        omitted_components=omitted_components,
    )
    symbol_plan = _build_symbol_plan(problem, problem_type, selected_ops, evidence)
    model_spec = _build_model_spec(
        problem,
        problem_type,
        selected_ops,
        symbol_plan,
        omitted_components=omitted_components,
    )
    validation = _validate_model_spec(model_spec)
    evidence_summary = _summarize_evidence(evidence)
    notes = [
        "Harness draft is a structured intermediate representation, not the final paper-level formulation.",
        "Symbols are recommendations and can be changed by a later Symbol Planner or by the user.",
        "No HHC-specific optional component is forced unless selected by the problem text or evidence.",
    ]
    if modeling_plan is None:
        notes.append("LLM modeling plan was unavailable; the harness used deterministic problem/evidence signals.")
    draft = HarnessDraft(
        mode="deterministic_harness_draft",
        problem_type=problem_type,
        component_selector=component_selector,
        symbol_plan=symbol_plan,
        model_spec=model_spec,
        validation=validation,
        evidence_summary=evidence_summary,
        notes=notes,
    )
    return draft.to_dict()


def _infer_problem_type(problem: str, skill: ModelingSkill) -> str:
    text = problem.lower()
    if skill.name == "home_health_care_routing_scheduling" or any(k in text for k in _HHC_KEYWORDS):
        return "home_health_care_routing_scheduling"
    if skill.name == "production_scheduling":
        return "production_scheduling"
    return "generic_optimization"


def _select_operators(
    problem: str,
    evidence: list[dict[str, Any]],
    *,
    modeling_plan: dict[str, Any] | None,
) -> list[OperatorItem]:
    text = problem.lower()
    operators: dict[str, OperatorItem] = {}
    omitted = _explicitly_omitted_components(text)

    def add_operator(name: str, purpose: str, chunk_id: str = "") -> None:
        if name in omitted:
            return
        if name not in operators:
            operators[name] = OperatorItem(name=name, purpose=purpose)
        if chunk_id and chunk_id not in operators[name].evidence_chunk_ids:
            operators[name].evidence_chunk_ids.append(chunk_id)

    for name, keywords in _COMPONENT_KEYWORDS.items():
        if _component_has_positive_signal(name, text, keywords):
            add_operator(name, _operator_purpose(name))

    # HHC defaults stay generic, but the harness may now import advanced
    # structures as evidence-supported assumptions when retrieved model regions
    # repeatedly support them.
    if _mentions_any(text, ("patient", "patients", "患者")) and _mentions_any(
        text, ("caregiver", "caregivers", "nurse", "nurses", "护理员", "护士")
    ):
        add_operator("assignment", _operator_purpose("assignment"))
    if _mentions_any(text, ("route", "routing", "travel", "visit", "visits", "路径", "路线")):
        add_operator("routing_flow", _operator_purpose("routing_flow"))
    if _mentions_any(text, ("waiting", "waiting time", "等待")):
        add_operator("waiting_time", _operator_purpose("waiting_time"))
        # Waiting time requires a service-start reference. This is a generic
        # timing link, not a full time-window assumption.
        add_operator("time_propagation", _operator_purpose("time_propagation"))
    if _has_multi_objective_signal(text):
        add_operator("multi_objective", _operator_purpose("multi_objective"))
    if _mentions_any(text, ("time window", "earliest", "latest", "ready time", "due time", "时间窗")):
        add_operator("time_window", _operator_purpose("time_window"))
        add_operator("time_propagation", _operator_purpose("time_propagation"))
    if _has_capacity_positive_signal(text):
        add_operator("capacity", _operator_purpose("capacity"))
    if _mentions_any(text, ("balance", "balanced", "fairness", "均衡", "公平")):
        add_operator("balance", _operator_purpose("balance"))
        if "balance" not in omitted:
            add_operator("capacity", _operator_purpose("capacity"))
    if _mentions_any(
        text,
        ("open route", "do not return", "don't return", "no return", "without returning", "开放路径", "不返回"),
    ):
        add_operator("routing_flow", _operator_purpose("routing_flow"))
        add_operator("open_route", _operator_purpose("open_route"))

    if modeling_plan:
        for item in modeling_plan.get("component_decisions") or []:
            if not isinstance(item, dict) or item.get("decision") not in {"use", "assumption"}:
                continue
            component = str(item.get("component", "")).lower()
            for name, keywords in _COMPONENT_KEYWORDS.items():
                if name in omitted:
                    continue
                if name.replace("_", " ") in component or any(k in component for k in keywords):
                    add_operator(name, str(item.get("reason") or _operator_purpose(name)))

    evidence_counts: dict[str, list[str]] = {}
    for item in evidence:
        chunk_id = str(item.get("chunk_id") or item.get("figure_id") or "")
        hints = item.get("operator_hints") or []
        for hint in hints:
            hint = str(hint)
            if hint in _COMPONENT_KEYWORDS and hint not in omitted:
                evidence_counts.setdefault(hint, [])
                if chunk_id and chunk_id not in evidence_counts[hint]:
                    evidence_counts[hint].append(chunk_id)
                if hint not in {"assignment", "routing_flow", "time_propagation"}:
                    continue
                add_operator(hint, _operator_purpose(hint), chunk_id)

    imported_optional = 0
    max_optional_imports = 4
    for hint, chunk_ids in sorted(
        evidence_counts.items(),
        key=lambda item: (_component_import_priority(item[0]), -len(item[1]), item[0]),
    ):
        if hint in operators or hint in omitted:
            continue
        if hint not in {"assignment", "routing_flow", "time_propagation"} and imported_optional >= max_optional_imports:
            continue
        if _evidence_can_import_component(hint, chunk_ids, text):
            add_operator(
                hint,
                f"evidence-supported HHC assumption: {_operator_purpose(hint)}",
                chunk_ids[0] if chunk_ids else "",
            )
            for chunk_id in chunk_ids[1:4]:
                add_operator(hint, _operator_purpose(hint), chunk_id)
            if hint not in {"assignment", "routing_flow", "time_propagation"}:
                imported_optional += 1

    return list(operators.values())


def _build_symbol_plan(
    problem: str,
    problem_type: str,
    operators: list[OperatorItem],
    evidence: list[dict[str, Any]],
) -> SymbolPlan:
    op_names = {op.name for op in operators}
    plan = SymbolPlan()

    if problem_type == "home_health_care_routing_scheduling":
        plan.sets.extend(
            [
                SymbolItem("P", "set", "patients or service requests"),
                SymbolItem("C", "set", "caregivers or mobile care resources"),
            ]
        )
        if "routing_flow" in op_names:
            plan.sets.append(SymbolItem("N", "set", "routing nodes, including depot and patients"))
            plan.sets.append(SymbolItem("A", "set", "feasible travel arcs between nodes"))
    else:
        plan.sets.append(SymbolItem("I", "set", "items, requests, or jobs"))
        plan.sets.append(SymbolItem("R", "set", "resources"))

    if "assignment" in op_names:
        plan.variables.append(SymbolItem("y_{pc}", "binary variable", "1 if patient p is assigned to caregiver c"))
    if "routing_flow" in op_names:
        plan.parameters.append(SymbolItem("t_{ij}", "parameter", "travel time or cost from node i to node j"))
        plan.variables.append(SymbolItem("x_{ijc}", "binary variable", "1 if caregiver c travels from node i to node j"))
        if "multi_objective" in op_names:
            plan.parameters.append(SymbolItem("f_c", "parameter", "fixed dispatch cost or activation penalty for caregiver c"))
            plan.variables.append(SymbolItem("u_c", "binary variable", "1 if caregiver c is dispatched"))
    if "routing_flow" in op_names:
        plan.parameters.append(SymbolItem("\\alpha", "parameter", "weight of the travel term in the objective"))
    if "waiting_time" in op_names:
        plan.parameters.append(SymbolItem("\\beta", "parameter", "weight of the waiting-time term in the objective"))
    if "multi_objective" in op_names:
        plan.parameters.append(SymbolItem("\\lambda_m", "parameter", "optional scalarization weight for objective component m"))
        plan.sets.append(SymbolItem("M^O", "set", "objective-component index set for multi-objective evaluation"))
    if "time_propagation" in op_names or "time_window" in op_names:
        plan.parameters.append(SymbolItem("s_p", "parameter", "service duration of patient p"))
        plan.variables.append(SymbolItem("T_p", "continuous variable", "service start time of patient p"))
    if "time_propagation" in op_names:
        plan.parameters.append(SymbolItem("M", "parameter", "sufficiently large constant for conditional timing constraints"))
    if "time_window" in op_names:
        plan.parameters.append(SymbolItem("a_p", "parameter", "earliest or requested service time of patient p"))
        plan.parameters.append(SymbolItem("b_p", "parameter", "latest service time of patient p"))
    if "waiting_time" in op_names:
        plan.parameters.append(SymbolItem("a_p", "parameter", "requested or earliest service time of patient p"))
        plan.variables.append(SymbolItem("T_p", "continuous variable", "service start time of patient p"))
        plan.variables.append(SymbolItem("W_p", "continuous variable", "waiting time of patient p"))
    if "capacity" in op_names:
        plan.parameters.append(SymbolItem("L_c", "parameter", "maximum workload or route duration of caregiver c"))
        plan.variables.append(SymbolItem("U_c", "continuous variable", "total workload assigned to caregiver c"))
    if "skill_matching" in op_names:
        plan.parameters.append(SymbolItem("q_c", "parameter", "caregiver skill or qualification level"))
        plan.parameters.append(SymbolItem("r_p", "parameter", "patient service skill requirement"))
    if "outsourcing" in op_names:
        plan.parameters.append(SymbolItem("\\gamma", "parameter", "weight of the outsourcing penalty term"))
        plan.parameters.append(SymbolItem("\\pi_p", "parameter", "outsourcing or rejection penalty for patient p"))
        plan.variables.append(SymbolItem("o_p", "binary variable", "1 if patient p is outsourced or externally served"))
    if "priority_class" in op_names:
        plan.sets.append(SymbolItem("P^V", "set", "priority or VIP patients, if defined by the problem"))
        plan.parameters.append(SymbolItem("\\omega_p", "parameter", "priority weight of patient p"))
    if "synchronization" in op_names:
        plan.sets.append(SymbolItem("S", "set", "pairs or groups of synchronized patient visits"))
        plan.parameters.append(SymbolItem("\\Delta_s", "parameter", "maximum allowed start-time gap for synchronized visit group s"))
    if "break_scheduling" in op_names:
        plan.parameters.append(SymbolItem("\\ell_c", "parameter", "earliest break start time for caregiver c"))
        plan.parameters.append(SymbolItem("u^B_c", "parameter", "latest break start time for caregiver c"))
        plan.parameters.append(SymbolItem("d^B_c", "parameter", "break duration for caregiver c"))
        plan.variables.append(SymbolItem("z_{pc}", "binary variable", "1 if caregiver c takes a break after serving patient p"))
    if "overtime" in op_names:
        plan.parameters.append(SymbolItem("\\eta", "parameter", "weight of the overtime term in the objective"))
        plan.parameters.append(SymbolItem("H_c", "parameter", "regular working-time limit of caregiver c"))
        plan.variables.append(SymbolItem("O_c", "continuous variable", "overtime of caregiver c"))
    if "multi_center" in op_names:
        plan.sets.append(SymbolItem("D", "set", "healthcare centers or depots"))
        plan.variables.append(SymbolItem("g_{cd}", "binary variable", "1 if caregiver c is assigned to center d"))
    if "preference_matching" in op_names:
        plan.parameters.append(SymbolItem("\\rho_{pc}", "parameter", "preference or compatibility score between patient p and caregiver c"))
    if "balance" in op_names:
        plan.parameters.append(SymbolItem("\\delta", "parameter", "weight of the workload-balance term"))
        plan.variables.append(SymbolItem("B", "continuous variable", "maximum workload imbalance or fairness measure"))

    _dedupe_symbol_items(plan.sets)
    _dedupe_symbol_items(plan.parameters)
    _dedupe_symbol_items(plan.variables)
    return plan


def _build_model_spec(
    problem: str,
    problem_type: str,
    operators: list[OperatorItem],
    symbol_plan: SymbolPlan,
    *,
    omitted_components: list[dict[str, Any]],
) -> ModelSpec:
    op_names = {op.name for op in operators}
    spec = ModelSpec(
        problem_type=problem_type,
        operators=operators,
        sets=symbol_plan.sets,
        parameters=symbol_plan.parameters,
        variables=symbol_plan.variables,
        omitted_components=omitted_components,
    )

    if "routing_flow" in op_names:
        spec.objective_terms.append(
            {
                "name": "travel_term",
                "description": "total travel time or travel cost over selected arcs",
                "symbols": ["t_{ij}", "x_{ijc}"],
            }
        )
    if "waiting_time" in op_names:
        spec.objective_terms.append(
            {
                "name": "waiting_term",
                "description": "total patient waiting time",
                "symbols": ["W_p"],
            }
        )
    if "outsourcing" in op_names:
        spec.objective_terms.append(
            {
                "name": "outsourcing_penalty",
                "description": "penalty for patients not served by internal caregivers",
                "symbols": ["\\pi_p", "o_p"],
            }
        )
    if "priority_class" in op_names:
        spec.objective_terms.append(
            {
                "name": "priority_weighted_service",
                "description": "priority-weighted delay or service quality term",
                "symbols": ["\\omega_p"],
            }
        )
    if "preference_matching" in op_names:
        spec.objective_terms.append(
            {
                "name": "preference_satisfaction",
                "description": "maximize or penalize patient-caregiver preference mismatch",
                "symbols": ["\\rho_{pc}", "y_{pc}"],
            }
        )
    if "overtime" in op_names:
        spec.objective_terms.append(
            {
                "name": "overtime_penalty",
                "description": "penalty for caregiver overtime",
                "symbols": ["O_c"],
            }
        )
    if "balance" in op_names:
        spec.objective_terms.append(
            {
                "name": "workload_balance",
                "description": "fairness or maximum workload imbalance among caregivers",
                "symbols": ["B"],
            }
        )
    if "multi_objective" in op_names:
        if len(spec.objective_terms) < 3:
            spec.objective_terms.append(
                {
                    "name": "completion_term",
                    "description": "total patient completion time or route completion performance used as an explicit third objective when not otherwise specified",
                    "symbols": ["T_p", "s_p"],
                }
            )
        if len(spec.objective_terms) < 3 and "routing_flow" in op_names:
            spec.objective_terms.append(
                {
                    "name": "caregiver_usage_term",
                    "description": "number or fixed dispatch cost of caregivers used, included as an assumed objective slot when the user asks for multiple objectives without naming all criteria",
                    "symbols": ["f_c", "u_c"],
                }
            )
        if len(spec.objective_terms) < 3:
            spec.objective_terms.append(
                {
                    "name": "service_start_term",
                    "description": "aggregate service start time used as a fallback objective slot when other criteria are unspecified",
                    "symbols": ["T_p"],
                }
            )
        spec.assumptions.append(
            "The user requested a multi-objective formulation; if some objective criteria are not specified, additional criteria are marked as modeling assumptions."
        )
    if not spec.objective_terms:
        spec.objective_terms.append(
            {
                "name": "generic_cost",
                "description": "problem-specific cost or performance term to be defined",
                "symbols": [],
            }
        )

    if "assignment" in op_names:
        spec.constraint_groups.append(
            {
                "name": "assignment_coverage",
                "purpose": "ensure each served patient/request is assigned to an eligible caregiver/resource",
                "operator": "assignment",
                "required_symbols": ["P", "C", "y_{pc}"],
            }
        )
    if "routing_flow" in op_names:
        spec.constraint_groups.append(
            {
                "name": "route_flow_consistency",
                "purpose": "connect assignment decisions with route arcs and conserve route flow",
                "operator": "routing_flow",
                "required_symbols": ["N", "A", "C", "x_{ijc}", "y_{pc}"],
            }
        )
    if "time_propagation" in op_names:
        spec.constraint_groups.append(
            {
                "name": "time_propagation",
                "purpose": "propagate service start times along selected travel arcs",
                "operator": "time_propagation",
                "required_symbols": ["T_p", "s_p", "t_{ij}", "x_{ijc}", "M"],
            }
        )
    if "time_window" in op_names:
        spec.constraint_groups.append(
            {
                "name": "time_window_feasibility",
                "purpose": "keep service times within allowed time windows",
                "operator": "time_window",
                "required_symbols": ["T_p", "a_p", "b_p"],
            }
        )
    if "waiting_time" in op_names:
        spec.constraint_groups.append(
            {
                "name": "waiting_time_definition",
                "purpose": "define waiting time from requested or earliest service time",
                "operator": "waiting_time",
                "required_symbols": ["W_p", "T_p", "a_p"],
            }
        )
    if "capacity" in op_names:
        spec.constraint_groups.append(
            {
                "name": "caregiver_capacity",
                "purpose": "limit caregiver workload or route duration",
                "operator": "capacity",
                "required_symbols": ["L_c", "U_c"],
            }
        )
    if "balance" in op_names:
        spec.constraint_groups.append(
            {
                "name": "workload_balance",
                "purpose": "define a workload or fairness measure across caregivers",
                "operator": "balance",
                "required_symbols": ["U_c", "B"],
            }
        )
    if "skill_matching" in op_names:
        spec.constraint_groups.append(
            {
                "name": "skill_compatibility",
                "purpose": "allow assignment only when caregiver skill satisfies patient requirement",
                "operator": "skill_matching",
                "required_symbols": ["q_c", "r_p", "y_{pc}"],
            }
        )
    if "synchronization" in op_names:
        spec.constraint_groups.append(
            {
                "name": "synchronized_visits",
                "purpose": "coordinate service start times for visits that must occur simultaneously or nearly simultaneously",
                "operator": "synchronization",
                "required_symbols": ["S", "T_p", "\\Delta_s"],
            }
        )
    if "break_scheduling" in op_names:
        spec.constraint_groups.append(
            {
                "name": "caregiver_break_scheduling",
                "purpose": "insert required lunch or rest breaks into caregiver routes",
                "operator": "break_scheduling",
                "required_symbols": ["z_{pc}", "\\ell_c", "u^B_c", "d^B_c"],
            }
        )
    if "overtime" in op_names:
        spec.constraint_groups.append(
            {
                "name": "overtime_definition",
                "purpose": "define overtime from workload exceeding regular working time",
                "operator": "overtime",
                "required_symbols": ["U_c", "H_c", "O_c"],
            }
        )
    if "multi_center" in op_names:
        spec.constraint_groups.append(
            {
                "name": "center_assignment",
                "purpose": "assign caregivers or routes to healthcare centers or depots",
                "operator": "multi_center",
                "required_symbols": ["D", "g_{cd}", "u_c"],
            }
        )
    if "preference_matching" in op_names:
        spec.constraint_groups.append(
            {
                "name": "preference_matching",
                "purpose": "use patient-caregiver preference or compatibility scores in assignment decisions",
                "operator": "preference_matching",
                "required_symbols": ["\\rho_{pc}", "y_{pc}"],
            }
        )
    if "outsourcing" in op_names:
        spec.constraint_groups.append(
            {
                "name": "outsourcing_logic",
                "purpose": "link internal service assignment with outsourcing decisions",
                "operator": "outsourcing",
                "required_symbols": ["o_p", "y_{pc}"],
            }
        )
    if "open_route" in op_names:
        spec.constraint_groups.append(
            {
                "name": "open_route_end",
                "purpose": "allow caregiver routes to end at a patient node rather than requiring return to depot",
                "operator": "open_route",
                "required_symbols": ["x_{ijc}", "N", "C"],
            }
        )
    if "multi_objective" in op_names:
        if "routing_flow" in op_names:
            spec.constraint_groups.append(
                {
                    "name": "caregiver_activation",
                    "purpose": "link caregiver dispatch variables with route departure or assignment decisions when caregiver-use objectives are included",
                    "operator": "multi_objective",
                    "required_symbols": ["u_c", "x_{ijc}"],
                }
            )
        spec.constraint_groups.append(
            {
                "name": "multi_objective_handling",
                "purpose": "state whether the vector objective is solved by Pareto search, weighted sum, epsilon-constraint, or lexicographic preference",
                "operator": "multi_objective",
                "required_symbols": [],
            }
        )

    spec.assumptions.extend(_derive_assumptions(problem, op_names))
    return spec


def _validate_model_spec(spec: ModelSpec) -> dict[str, Any]:
    defined_symbols = {item.symbol for item in spec.sets + spec.parameters + spec.variables}
    undefined: set[str] = set()
    for term in spec.objective_terms:
        for symbol in term.get("symbols") or []:
            if symbol and symbol not in defined_symbols:
                undefined.add(symbol)
    for group in spec.constraint_groups:
        for symbol in group.get("required_symbols") or []:
            if symbol and symbol not in defined_symbols:
                undefined.add(symbol)

    warnings: list[str] = []
    if not spec.operators:
        warnings.append("No operators were selected; the problem statement may be too vague.")
    if not spec.constraint_groups:
        warnings.append("No constraint groups were selected.")
    if undefined:
        warnings.append("Some symbols are required by operators but not defined in the symbol plan.")

    return {
        "defined_symbols": sorted(defined_symbols),
        "undefined_symbols": sorted(undefined),
        "warnings": warnings,
        "status": "ok" if not undefined and not warnings else "needs_review",
    }


def _summarize_evidence(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    operators: dict[str, int] = {}
    papers: dict[str, int] = {}
    for item in evidence:
        papers[str(item.get("paper_id", ""))] = papers.get(str(item.get("paper_id", "")), 0) + 1
        chunk_type = str(item.get("chunk_type") or item.get("modality") or "unknown")
        by_type[chunk_type] = by_type.get(chunk_type, 0) + 1
        for op in item.get("operator_hints") or []:
            op = str(op)
            operators[op] = operators.get(op, 0) + 1
    return {
        "papers": dict(sorted(papers.items())),
        "chunk_types": dict(sorted(by_type.items())),
        "operator_hints": dict(sorted(operators.items())),
    }


def _build_component_selector(
    problem: str,
    evidence: list[dict[str, Any]],
    *,
    selected_ops: list[OperatorItem],
    omitted_components: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Explain component selection before formula rendering.

    This is the lightweight HHC Component Selector. It exposes the modeling
    choices made by the harness so the UI can show why a component was used,
    omitted, or left unselected.
    """
    text = problem.lower()
    selected = {op.name: op for op in selected_ops}
    omitted = {item.get("component", ""): item for item in omitted_components}
    evidence_map = _component_evidence_map(evidence)
    rows: list[dict[str, Any]] = []
    for component in _component_order():
        prompt_hits = _component_prompt_hits(component, text)
        evidence_hits = evidence_map.get(component, [])
        if component in selected:
            status = "selected"
            reason_parts = []
            if prompt_hits:
                reason_parts.append("requested or implied by the problem text")
            if evidence_hits:
                reason_parts.append("supported by retrieved model evidence")
            if not reason_parts:
                reason_parts.append("added as a minimal structural requirement")
            reason = "; ".join(reason_parts)
        elif component in omitted:
            status = "omitted"
            reason = str(omitted[component].get("reason") or "explicitly omitted")
        else:
            status = "not_selected"
            if evidence_hits and component not in {"assignment", "routing_flow", "time_propagation", "waiting_time"}:
                reason = "seen in retrieved evidence, but evidence support was too weak to import it as an assumption"
            else:
                reason = "not required by the current problem statement"
        rows.append(
            {
                "component": component,
                "label": _component_label(component),
                "status": status,
                "reason": reason,
                "purpose": _operator_purpose(component),
                "prompt_signals": prompt_hits,
                "evidence_chunk_ids": evidence_hits[:6],
            }
        )
    return rows


def _evidence_can_import_component(component: str, chunk_ids: list[str], problem_lower: str) -> bool:
    if not chunk_ids:
        return False
    if component in {"assignment", "routing_flow", "time_propagation"}:
        return True
    if _component_prompt_hits(component, problem_lower):
        return True
    advanced = {
        "time_window",
        "capacity",
        "balance",
        "skill_matching",
        "outsourcing",
        "priority_class",
        "synchronization",
        "break_scheduling",
        "overtime",
        "multi_center",
        "preference_matching",
        "open_route",
        "multi_objective",
    }
    if component not in advanced:
        return False
    # One strong model-region hit is enough for common HHC assumptions; repeated
    # hits make paper-specific components safer to import.
    if component in {"time_window", "capacity", "skill_matching", "overtime", "multi_center"}:
        return len(chunk_ids) >= 1
    return len(chunk_ids) >= 2


def _component_import_priority(component: str) -> int:
    priority = {
        "assignment": 0,
        "routing_flow": 1,
        "time_propagation": 2,
        "time_window": 10,
        "capacity": 11,
        "skill_matching": 12,
        "overtime": 13,
        "synchronization": 14,
        "break_scheduling": 15,
        "multi_center": 16,
        "preference_matching": 17,
        "balance": 18,
        "outsourcing": 19,
        "priority_class": 20,
        "open_route": 21,
        "multi_objective": 22,
    }
    return priority.get(component, 99)


def _component_evidence_map(evidence: list[dict[str, Any]]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for item in evidence:
        chunk_id = str(item.get("chunk_id") or item.get("figure_id") or "")
        if not chunk_id:
            continue
        for hint in item.get("operator_hints") or []:
            hint = str(hint)
            if hint not in _COMPONENT_KEYWORDS:
                continue
            mapping.setdefault(hint, [])
            if chunk_id not in mapping[hint]:
                mapping[hint].append(chunk_id)
    return mapping


def _component_prompt_hits(component: str, problem_lower: str) -> list[str]:
    hits = _component_keyword_hits(component, problem_lower)
    return hits[:8]


def _component_has_positive_signal(component: str, text: str, keywords: tuple[str, ...]) -> bool:
    return bool(_component_keyword_hits(component, text, keywords))


def _component_keyword_hits(
    component: str,
    text: str,
    keywords: tuple[str, ...] | None = None,
) -> list[str]:
    keywords = keywords or _COMPONENT_KEYWORDS.get(component, ())
    hits: list[str] = []
    for keyword in keywords:
        if keyword in text:
            hits.append(keyword)
    if component == "capacity" and _mentions_balance_concept(text):
        hits = [
            hit
            for hit in hits
            if hit not in {"workload", "工作量"} or _explicit_capacity_omission(text) or _has_capacity_positive_signal(text)
        ]
    if component == "routing_flow" and _mentions_open_route_concept(text) and not _has_route_flow_positive_signal(text):
        hits = [
            hit
            for hit in hits
            if hit not in {"route", "routing", "path", "depot", "路径", "路线"}
        ]
    return hits


def _has_capacity_positive_signal(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "capacity",
            "workload limit",
            "workload limits",
            "workload constraint",
            "workload constraints",
            "working time",
            "route duration",
            "workload capacity",
            "容量",
            "工作量约束",
            "工时",
        )
    )


def _has_route_flow_positive_signal(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "routing",
            "route planning",
            "planning visit routes",
            "visit routes",
            "travel time",
            "travel cost",
            "path",
            "arc",
            "depot after visits",
            "路径",
            "路线",
        )
    )


def _has_multi_objective_signal(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "multi-objective",
            "multi objective",
            "multiobjective",
            "three-objective",
            "tri-objective",
            "3-objective",
            "multi criteria",
            "multiple objectives",
            "pareto",
            "多目标",
            "三目标",
            "3目标",
            "三个目标",
            "三个优化目标",
        )
    )


def _component_order() -> list[str]:
    return [
        "assignment",
        "routing_flow",
        "time_propagation",
        "waiting_time",
        "time_window",
        "capacity",
        "balance",
        "skill_matching",
        "outsourcing",
        "priority_class",
        "synchronization",
        "break_scheduling",
        "overtime",
        "multi_center",
        "preference_matching",
        "open_route",
        "multi_objective",
    ]


def _component_label(component: str) -> str:
    labels = {
        "assignment": "Assignment",
        "routing_flow": "Routing Flow",
        "time_propagation": "Time Propagation",
        "waiting_time": "Waiting Time",
        "time_window": "Time Window",
        "capacity": "Capacity / Workload",
        "balance": "Workload Balance",
        "skill_matching": "Skill Matching",
        "outsourcing": "Outsourcing",
        "priority_class": "Priority / VIP",
        "synchronization": "Synchronized Visits",
        "break_scheduling": "Break Scheduling",
        "overtime": "Overtime",
        "multi_center": "Multi-center",
        "preference_matching": "Preference Matching",
        "open_route": "Open Route",
        "multi_objective": "Multi-objective",
    }
    return labels.get(component, component.replace("_", " ").title())


def _operator_purpose(name: str) -> str:
    purposes = {
        "assignment": "assign requests to resources",
        "routing_flow": "construct feasible visit routes",
        "time_window": "enforce allowable service time intervals",
        "time_propagation": "link route decisions with service start times",
        "waiting_time": "measure patient waiting or delay",
        "capacity": "limit workload, duration, or resource capacity",
        "skill_matching": "match resource qualification with request requirements",
        "outsourcing": "represent external service or unserved request options",
        "priority_class": "represent priority classes only when required",
        "synchronization": "coordinate visits that must occur simultaneously or with limited time gaps",
        "break_scheduling": "insert lunch or rest breaks into caregiver schedules",
        "overtime": "model caregiver overtime beyond regular working limits",
        "multi_center": "assign caregivers or routes to multiple healthcare centers",
        "preference_matching": "model patient-caregiver compatibility or preference scores",
        "multi_objective": "combine multiple performance criteria",
        "open_route": "allow routes to terminate without returning to the depot",
        "balance": "represent workload fairness across caregivers",
    }
    return purposes.get(name, name.replace("_", " "))


def _derive_assumptions(problem: str, op_names: set[str]) -> list[str]:
    assumptions: list[str] = []
    if "routing_flow" in op_names:
        assumptions.append("Travel times between relevant locations are known or estimable.")
    if "waiting_time" in op_names:
        assumptions.append("A requested, earliest, or reference service time is available for each patient.")
    if "time_propagation" in op_names:
        assumptions.append("Service durations are known and can be linked to route timing.")
    if "skill_matching" in op_names:
        assumptions.append("Caregiver qualifications and patient service requirements are encoded numerically or categorically.")
    if "outsourcing" in op_names:
        assumptions.append("The problem allows some patients to be served externally or rejected with a penalty.")
    if "synchronization" in op_names:
        assumptions.append("Some visits may require synchronized or near-synchronized service start times.")
    if "break_scheduling" in op_names:
        assumptions.append("Caregivers may require scheduled lunch or rest breaks during routes.")
    if "overtime" in op_names:
        assumptions.append("Caregiver overtime is allowed and can be penalized or bounded.")
    if "multi_center" in op_names:
        assumptions.append("The service system may involve multiple healthcare centers or depot choices.")
    if "preference_matching" in op_names:
        assumptions.append("Patient-caregiver preference or compatibility data can be represented by scores.")
    if "balance" in op_names:
        assumptions.append("A measurable workload definition is available for comparing caregivers.")
    if "open_route" in op_names:
        assumptions.append("Caregiver routes are allowed to end away from the depot unless the problem states otherwise.")
    return assumptions


def _omitted_component_details(problem_lower: str) -> list[dict[str, str]]:
    omitted = _explicitly_omitted_components(problem_lower)
    details: list[dict[str, str]] = []
    for component in sorted(omitted):
        details.append(
            {
                "component": component,
                "reason": "explicitly excluded by the user problem statement",
            }
        )
    return details


def _explicitly_omitted_components(problem_lower: str) -> set[str]:
    omitted: set[str] = set()
    for component, patterns in _NEGATION_PATTERNS.items():
        if any(pattern in problem_lower for pattern in patterns):
            omitted.add(component)
    negative_scopes = re.findall(
        r"(?:do not include|do not consider|don't include|don't consider|without|exclude|不要|不考虑)[^.。;；\n]*",
        problem_lower,
        flags=re.IGNORECASE,
    )
    for scope in negative_scopes:
        if _is_open_route_positive_scope(scope):
            continue
        for component, keywords in _COMPONENT_KEYWORDS.items():
            if component in omitted:
                continue
            if _scope_omits_component(component, scope, keywords):
                omitted.add(component)
    return omitted


def _scope_omits_component(component: str, scope: str, keywords: tuple[str, ...]) -> bool:
    """Return whether a local negation scope should omit a component.

    Some component names overlap. For example, "do not include open routes"
    should omit open_route, not the foundational routing_flow operator; and
    "do not include workload balance" should omit balance, not workload
    capacity. This keeps optional HHC features from suppressing the base model.
    """
    if not any(keyword in scope for keyword in keywords):
        return False
    if component == "routing_flow" and _mentions_open_route_concept(scope):
        return _explicit_route_flow_omission(scope)
    if component == "capacity" and _mentions_balance_concept(scope):
        return _explicit_capacity_omission(scope)
    return True


def _is_open_route_positive_scope(scope: str) -> bool:
    return any(
        phrase in scope
        for phrase in (
            "do not return",
            "don't return",
            "no return",
            "without returning",
            "not return to depot",
            "无需返回",
            "不返回",
        )
    )


def _mentions_open_route_concept(scope: str) -> bool:
    return any(
        phrase in scope
        for phrase in (
            "open route",
            "open routes",
            "no return",
            "without returning",
            "not return to depot",
            "do not return",
            "don't return",
            "开放路径",
            "不返回",
        )
    )


def _explicit_route_flow_omission(scope: str) -> bool:
    return any(
        phrase in scope
        for phrase in (
            "do not include routing",
            "do not consider routing",
            "don't include routing",
            "without routing",
            "no routing",
            "exclude routing",
            "do not include route planning",
            "without route planning",
            "no route planning",
            "不考虑路径",
            "不考虑路线",
        )
    )


def _mentions_balance_concept(scope: str) -> bool:
    return any(
        phrase in scope
        for phrase in (
            "balance",
            "balanced",
            "fairness",
            "workload balance",
            "均衡",
            "公平",
        )
    )


def _explicit_capacity_omission(scope: str) -> bool:
    return any(
        phrase in scope
        for phrase in (
            "do not include capacity",
            "do not consider capacity",
            "don't include capacity",
            "without capacity",
            "no capacity",
            "do not include workload limits",
            "without workload limits",
            "no workload limits",
            "do not include route duration",
            "without route duration",
            "no route duration",
            "不考虑容量",
            "不考虑工作量约束",
        )
    )


def _mentions_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _dedupe_symbol_items(items: list[SymbolItem]) -> None:
    seen: set[str] = set()
    deduped: list[SymbolItem] = []
    for item in items:
        key = re.sub(r"\s+", " ", item.symbol).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    items[:] = deduped
