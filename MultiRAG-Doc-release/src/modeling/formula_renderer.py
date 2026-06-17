"""Render a harness ModelSpec into a compact mathematical-model JSON."""

from __future__ import annotations

from typing import Any


def render_harness_model(harness_draft: dict[str, Any]) -> dict[str, Any]:
    """Render deterministic formulas from a Harness Draft.

    The renderer follows selected operators and symbols from ModelSpec. It does
    not add domain-specific optional components beyond the harness selection.
    """
    spec = harness_draft.get("model_spec") or {}
    validation = harness_draft.get("validation") or {}
    op_names = {
        str(op.get("name", ""))
        for op in spec.get("operators", []) or []
        if isinstance(op, dict)
    }
    evidence_by_op = _operator_evidence(spec)
    all_evidence_ids = _merge_evidence(evidence_by_op, op_names)
    problem_type = str(spec.get("problem_type") or harness_draft.get("problem_type") or "")
    sets = [_render_symbol_item(item, all_evidence_ids) for item in spec.get("sets", []) or []]
    parameters = [_render_symbol_item(item, all_evidence_ids) for item in spec.get("parameters", []) or []]
    variables = [
        {
            "symbol": item.get("symbol", ""),
            "definition": item.get("description", ""),
            "domain": _variable_domain(str(item.get("role", "")), str(item.get("symbol", ""))),
            "source_chunk_ids": all_evidence_ids,
        }
        for item in spec.get("variables", []) or []
        if isinstance(item, dict)
    ]

    objective = _render_objective(op_names, evidence_by_op)
    constraints = _render_constraints(op_names, evidence_by_op)
    assumptions = list(spec.get("assumptions", []) or [])
    notes = [
        "Rendered from Harness ModelSpec; formulas are a controlled draft for review.",
        "The renderer uses generic optimization structures selected by the harness, not a copied paper formulation.",
    ]
    if validation.get("status") != "ok":
        notes.append("Harness validation requires review before using the rendered model.")

    return {
        "problem_analysis": {
            "problem_type": problem_type,
            "entities": _problem_entities(problem_type),
            "goals": _problem_goals(op_names),
            "requirements": sorted(op_names),
            "missing_information": _missing_information(op_names),
        },
        "reference_models": [],
        "component_applicability": [
            {
                "component": op.get("name", ""),
                "applicable": True,
                "reason": op.get("purpose", ""),
                "source_chunk_ids": op.get("evidence_chunk_ids", []) or [],
            }
            for op in spec.get("operators", []) or []
            if isinstance(op, dict)
        ],
        "omitted_components": list(spec.get("omitted_components", []) or []),
        "sets": sets,
        "parameters": parameters,
        "decision_variables": variables,
        "objective": objective,
        "constraints": constraints,
        "assumptions": assumptions,
        "validation": {
            "undefined_symbols": validation.get("undefined_symbols", []),
            "potential_conflicts": validation.get("warnings", []),
            "notes": notes,
        },
        "confidence": 0.72 if validation.get("status") == "ok" else 0.55,
        "render_mode": "harness_formula_renderer",
    }


def _operator_evidence(spec: dict[str, Any]) -> dict[str, list[str]]:
    evidence: dict[str, list[str]] = {}
    for op in spec.get("operators", []) or []:
        if not isinstance(op, dict):
            continue
        name = str(op.get("name", ""))
        ids = [str(item) for item in op.get("evidence_chunk_ids", []) or [] if item]
        if name:
            evidence[name] = _dedupe(ids)
    return evidence


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _merge_evidence(evidence_by_op: dict[str, list[str]], op_names: set[str]) -> list[str]:
    ids: list[str] = []
    for op_name in sorted(op_names):
        ids.extend(evidence_by_op.get(op_name, []))
    return _dedupe(ids)


def _source_ids(evidence_by_op: dict[str, list[str]], *components: str) -> list[str]:
    ids: list[str] = []
    for component in components:
        ids.extend(evidence_by_op.get(component, []))
    if not ids:
        ids = _merge_evidence(evidence_by_op, set(evidence_by_op))
    return _dedupe(ids)


def _render_symbol_item(item: dict[str, Any], source_chunk_ids: list[str]) -> dict[str, Any]:
    return {
        "symbol": item.get("symbol", ""),
        "definition": item.get("description", ""),
        "source_chunk_ids": source_chunk_ids,
    }


def _render_objective(op_names: set[str], evidence_by_op: dict[str, list[str]]) -> dict[str, Any]:
    terms: list[str] = []
    descriptions: list[str] = []
    if "routing_flow" in op_names:
        terms.append(r"\alpha \sum_{c \in C}\sum_{(i,j)\in A} t_{ij}x_{ijc}")
        descriptions.append("travel time or travel cost")
    if "waiting_time" in op_names:
        terms.append(r"\beta \sum_{p \in P} W_p")
        descriptions.append("patient waiting time")
    if "outsourcing" in op_names:
        terms.append(r"\gamma \sum_{p \in P} \pi_p o_p")
        descriptions.append("outsourcing or rejection penalty")
    if "overtime" in op_names:
        terms.append(r"\eta \sum_{c \in C} O_c")
        descriptions.append("caregiver overtime")
    if "preference_matching" in op_names:
        terms.append(r"-\theta \sum_{p\in P}\sum_{c\in C}\rho_{pc}y_{pc}")
        descriptions.append("patient-caregiver preference satisfaction")
    if "balance" in op_names:
        terms.append(r"\delta B")
        descriptions.append("workload balance")
    if "priority_class" in op_names and "waiting_time" in op_names:
        terms = [term.replace(r"\sum_{p \in P} W_p", r"\sum_{p \in P} \omega_p W_p") for term in terms]
        descriptions.append("priority-weighted service delay")
    if "multi_objective" in op_names and len(terms) < 3:
        terms.append(r"\sum_{p \in P}(T_p+s_p)")
        descriptions.append("patient completion time")
    if "multi_objective" in op_names and len(terms) < 3 and "routing_flow" in op_names:
        terms.append(r"\sum_{c \in C} u_c")
        descriptions.append("number of dispatched caregivers")

    if terms:
        if "multi_objective" in op_names:
            formula = r"\min \; \left(" + ", ".join(terms) + r"\right)"
            description = (
                "Minimize the objective vector: "
                + ", ".join(descriptions)
                + ". A scalarization, Pareto, epsilon-constraint, or lexicographic method can be selected later."
            )
            sense = "multi-objective"
        else:
            formula = r"\min \; " + " + ".join(terms)
            description = "Minimize " + ", ".join(descriptions) + "."
            sense = "minimize"
    else:
        formula = r"\min \; Z"
        description = "Minimize the problem-specific cost or performance measure."
        sense = "minimize"
    return {
        "sense": sense,
        "formula": formula,
        "description": description,
        "source_chunk_ids": _source_ids(
            evidence_by_op,
            "routing_flow",
            "waiting_time",
            "outsourcing",
            "overtime",
            "preference_matching",
            "balance",
            "priority_class",
            "multi_objective",
        ),
    }


def _render_constraints(op_names: set[str], evidence_by_op: dict[str, list[str]]) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    if "assignment" in op_names:
        constraints.append(
            {
                "name": "Assignment coverage",
                "formula": r"\sum_{c \in C} y_{pc} = 1,\quad \forall p \in P",
                "description": "Each patient is assigned to exactly one caregiver.",
                "source_chunk_ids": _source_ids(evidence_by_op, "assignment"),
            }
        )
    if "routing_flow" in op_names:
        constraints.extend(
            [
                {
                    "name": "Visit activation",
                    "formula": (
                        r"\sum_{j:(p,j)\in A} x_{pjc} = y_{pc},\quad "
                        r"\forall p \in P,\; c \in C"
                    ),
                    "description": "A caregiver leaves a patient node only if that patient is assigned to the caregiver.",
                    "source_chunk_ids": _source_ids(evidence_by_op, "routing_flow", "assignment"),
                },
                {
                    "name": "Route flow balance",
                    "formula": (
                        r"\sum_{i:(i,p)\in A} x_{ipc} = \sum_{j:(p,j)\in A} x_{pjc},\quad "
                        r"\forall p \in P,\; c \in C"
                    ),
                    "description": "For every visited patient, route inflow equals route outflow for each caregiver.",
                    "source_chunk_ids": _source_ids(evidence_by_op, "routing_flow"),
                },
            ]
        )
    if "time_propagation" in op_names:
        constraints.append(
            {
                "name": "Time propagation",
                "formula": (
                    r"T_j \ge T_i + s_i + t_{ij} - M(1-x_{ijc}),\quad "
                    r"\forall (i,j)\in A,\; c \in C"
                ),
                "description": "If a caregiver travels from node i to node j, the service start time at j follows travel and service time.",
                "source_chunk_ids": _source_ids(evidence_by_op, "time_propagation", "routing_flow"),
            }
        )
    if "time_window" in op_names:
        constraints.append(
            {
                "name": "Time-window feasibility",
                "formula": r"a_p \le T_p \le b_p,\quad \forall p \in P",
                "description": "Service starts within the allowed time interval for each patient.",
                "source_chunk_ids": _source_ids(evidence_by_op, "time_window"),
            }
        )
    if "waiting_time" in op_names:
        constraints.append(
            {
                "name": "Waiting-time definition",
                "formula": r"W_p \ge T_p - a_p,\quad W_p \ge 0,\quad \forall p \in P",
                "description": "Waiting time is the positive delay between service start and the requested or earliest service time.",
                "source_chunk_ids": _source_ids(evidence_by_op, "waiting_time"),
            }
        )
    if "capacity" in op_names:
        if "routing_flow" in op_names and "assignment" in op_names:
            constraints.extend(
                [
                    {
                        "name": "Workload definition",
                        "formula": (
                            r"U_c = \sum_{(i,j)\in A} t_{ij}x_{ijc} + \sum_{p\in P}s_p y_{pc},"
                            r"\quad \forall c \in C"
                        ),
                        "description": "Caregiver workload combines travel and service time.",
                        "source_chunk_ids": _source_ids(evidence_by_op, "capacity", "routing_flow"),
                    },
                    {
                        "name": "Caregiver workload limit",
                        "formula": r"U_c \le L_c,\quad \forall c \in C",
                        "description": "Each caregiver's workload stays within the available limit.",
                        "source_chunk_ids": _source_ids(evidence_by_op, "capacity"),
                    },
                ]
            )
        else:
            constraints.append(
                {
                    "name": "Resource capacity limit",
                    "formula": r"U_c \le L_c,\quad \forall c \in C",
                    "description": "Each caregiver/resource workload stays within the available limit.",
                    "source_chunk_ids": _source_ids(evidence_by_op, "capacity"),
                }
            )
    if "balance" in op_names:
        constraints.append(
            {
                "name": "Workload balance",
                "formula": r"U_c - U_{c'} \le B,\quad \forall c,c' \in C",
                "description": "The balance variable bounds pairwise workload differences.",
                "source_chunk_ids": _source_ids(evidence_by_op, "balance", "capacity"),
            }
        )
    if "skill_matching" in op_names:
        constraints.append(
            {
                "name": "Skill compatibility",
                "formula": r"r_p y_{pc} \le q_c,\quad \forall p \in P,\; c \in C",
                "description": "A patient can be assigned to a caregiver only if the caregiver has sufficient qualification.",
                "source_chunk_ids": _source_ids(evidence_by_op, "skill_matching"),
            }
        )
    if "synchronization" in op_names:
        constraints.append(
            {
                "name": "Synchronized visit timing",
                "formula": r"|T_p-T_q|\le \Delta_s,\quad \forall s=(p,q)\in S",
                "description": "Paired or synchronized visits start within the allowed time gap.",
                "source_chunk_ids": _source_ids(evidence_by_op, "synchronization", "time_propagation"),
            }
        )
    if "break_scheduling" in op_names:
        constraints.append(
            {
                "name": "Break placement",
                "formula": r"\ell_c z_{pc}\le T_p+s_p+d^B_c \le u^B_c + M(1-z_{pc}),\quad \forall p\in P,c\in C",
                "description": "If caregiver c takes a break after patient p, the break is placed inside the allowed break interval.",
                "source_chunk_ids": _source_ids(evidence_by_op, "break_scheduling", "time_propagation"),
            }
        )
    if "overtime" in op_names:
        constraints.append(
            {
                "name": "Overtime definition",
                "formula": r"O_c\ge U_c-H_c,\quad O_c\ge0,\quad \forall c\in C",
                "description": "Overtime is the positive excess of caregiver workload over regular working time.",
                "source_chunk_ids": _source_ids(evidence_by_op, "overtime", "capacity"),
            }
        )
    if "multi_center" in op_names:
        constraints.append(
            {
                "name": "Center assignment",
                "formula": r"\sum_{d\in D}g_{cd}=u_c,\quad \forall c\in C",
                "description": "Each dispatched caregiver is assigned to exactly one healthcare center.",
                "source_chunk_ids": _source_ids(evidence_by_op, "multi_center", "routing_flow"),
            }
        )
    if "preference_matching" in op_names:
        constraints.append(
            {
                "name": "Preference-aware assignment score",
                "formula": r"R=\sum_{p\in P}\sum_{c\in C}\rho_{pc}y_{pc}",
                "description": "The aggregate compatibility score can be optimized or reported as a service-quality measure.",
                "source_chunk_ids": _source_ids(evidence_by_op, "preference_matching", "assignment"),
            }
        )
    if "outsourcing" in op_names:
        constraints.append(
            {
                "name": "Internal service or outsourcing",
                "formula": r"\sum_{c\in C} y_{pc} + o_p = 1,\quad \forall p \in P",
                "description": "Each patient is either served internally or assigned to external service.",
                "source_chunk_ids": _source_ids(evidence_by_op, "outsourcing", "assignment"),
            }
        )
    if "open_route" in op_names:
        constraints.append(
            {
                "name": "Open route ending",
                "formula": r"\sum_{i\in N}x_{ipc} - \sum_{j\in N}x_{pjc} \le 1,\quad \forall p\in P,\; c\in C",
                "description": "Routes may terminate at a patient node instead of forcing every route to return to the depot.",
                "source_chunk_ids": _source_ids(evidence_by_op, "open_route", "routing_flow"),
            }
        )
    if "multi_objective" in op_names and "routing_flow" in op_names:
        constraints.append(
            {
                "name": "Caregiver activation",
                "formula": r"\sum_{j:(0,j)\in A}x_{0jc}\le u_c,\quad u_c\le \sum_{j:(0,j)\in A}x_{0jc},\quad \forall c\in C",
                "description": "A caregiver is marked as dispatched exactly when a route leaves the depot for that caregiver.",
                "source_chunk_ids": _source_ids(evidence_by_op, "multi_objective", "routing_flow"),
            }
        )
    if "multi_objective" in op_names:
        constraints.append(
            {
                "name": "Multi-objective solution protocol",
                "formula": r"Z=(Z_1,Z_2,Z_3),\quad Z_1,Z_2,Z_3 \text{ evaluated separately or scalarized by user-selected preferences}",
                "description": "The three objective components are kept as a vector objective unless the user specifies weights, lexicographic priority, or epsilon-constraint bounds.",
                "source_chunk_ids": _source_ids(evidence_by_op, "multi_objective"),
            }
        )
    constraints.append(
        {
            "name": "Variable domains",
            "formula": _domain_formula(op_names),
            "description": "Domain restrictions for selected decision variables.",
            "source_chunk_ids": _merge_evidence(evidence_by_op, op_names),
        }
    )
    return constraints


def _domain_formula(op_names: set[str]) -> str:
    parts = []
    if "assignment" in op_names:
        parts.append(r"y_{pc}\in\{0,1\}")
    if "routing_flow" in op_names:
        parts.append(r"x_{ijc}\in\{0,1\}")
    if "multi_objective" in op_names and "routing_flow" in op_names:
        parts.append(r"u_c\in\{0,1\}")
    if "waiting_time" in op_names:
        parts.append(r"W_p\ge 0")
    if "time_propagation" in op_names or "time_window" in op_names:
        parts.append(r"T_p\ge 0")
    if "capacity" in op_names:
        parts.append(r"U_c\ge 0")
    if "balance" in op_names:
        parts.append(r"B\ge 0")
    if "outsourcing" in op_names:
        parts.append(r"o_p\in\{0,1\}")
    if "break_scheduling" in op_names:
        parts.append(r"z_{pc}\in\{0,1\}")
    if "overtime" in op_names:
        parts.append(r"O_c\ge 0")
    if "multi_center" in op_names:
        parts.append(r"g_{cd}\in\{0,1\}")
    if "multi_objective" in op_names:
        parts.append(r"Z_m\in\mathbb{R},\; m\in M^O")
    return r",\quad ".join(parts) if parts else r"Z \in \mathbb{R}"


def _variable_domain(role: str, symbol: str) -> str:
    lower = f"{role} {symbol}".lower()
    if "binary" in lower or symbol in {"y_{pc}", "x_{ijc}", "o_p"}:
        return "binary"
    if "integer" in lower:
        return "integer"
    return "continuous"


def _problem_entities(problem_type: str) -> list[str]:
    if problem_type == "home_health_care_routing_scheduling":
        return ["patients", "caregivers", "routing nodes"]
    return ["items", "resources"]


def _problem_goals(op_names: set[str]) -> list[str]:
    goals = []
    if "routing_flow" in op_names:
        goals.append("minimize travel time or cost")
    if "waiting_time" in op_names:
        goals.append("minimize patient waiting time")
    if "multi_objective" in op_names:
        goals.append("represent multiple objective components")
    return goals or ["optimize problem-specific cost"]


def _missing_information(op_names: set[str]) -> list[str]:
    missing = ["numerical parameter values", "depot definition"]
    if "waiting_time" in op_names:
        missing.append("reference time used to compute waiting time")
    if "time_propagation" in op_names:
        missing.append("big-M calibration")
    return missing
