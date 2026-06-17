"""Grounded math-model generation from modeling knowledge evidence."""

from __future__ import annotations

import copy
import json
import re
import time
from collections.abc import Callable
from typing import Any

from src.config import CFG
from src.generator.llm_client import generate, generate_stream
from src.modeling.formula_renderer import render_harness_model
from src.modeling.harness import build_harness_draft
from src.modeling.model_verifier import verify_model
from src.modeling.platemo_codegen import generate_platemo_code as build_platemo_code
from src.modeling.quality_rubric import evaluate_model_quality
from src.modeling.query_intent import is_math_modeling_query
from src.modeling.skills import ModelingSkill, select_modeling_skill
from src.query.service import QueryService, flatten_dual_results

ProgressCallback = Callable[[str, str, float], None]
CancelCheck = Callable[[], bool]


class ModelingCancelledError(RuntimeError):
    """Raised when a cooperative modeling job cancellation is requested."""


def _emit_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    message: str,
    progress: float,
) -> None:
    if progress_callback is not None:
        progress_callback(stage, message, progress)


def _check_cancelled(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None and cancel_check():
        raise ModelingCancelledError("job cancelled by user")


MODEL_GENERATION_SYSTEM_PROMPT = """You are an expert in operations research and mathematical modeling.

Your task is to draft a new optimization model for the user's problem.
You must ground the draft in the retrieved paper evidence, but the retrieved
papers are reference material, not templates. Construct a new model for the
user's problem. Reuse only structures that fit the user's stated problem, and
explicitly mark missing information.

Return strict JSON only, with this schema:
{
  "problem_analysis": {
    "problem_type": string,
    "entities": [string],
    "goals": [string],
    "requirements": [string],
    "missing_information": [string]
  },
  "reference_models": [
    {
      "paper_id": string,
      "chunk_id": string,
      "reusable_parts": [string]
    }
  ],
  "component_applicability": [
    {
      "component": string,
      "applicable": boolean,
      "reason": string,
      "source_chunk_ids": [string]
    }
  ],
  "sets": [
    {
      "symbol": string,
      "definition": string,
      "source_chunk_ids": [string]
    }
  ],
  "parameters": [
    {
      "symbol": string,
      "definition": string,
      "source_chunk_ids": [string]
    }
  ],
  "decision_variables": [
    {
      "symbol": string,
      "definition": string,
      "domain": string,
      "source_chunk_ids": [string]
    }
  ],
  "objective": {
    "sense": "minimize|maximize|multi-objective|unknown",
    "formula": string,
    "description": string,
    "source_chunk_ids": [string]
  },
  "constraints": [
    {
      "name": string,
      "formula": string,
      "description": string,
      "source_chunk_ids": [string]
    }
  ],
  "assumptions": [string],
  "validation": {
    "undefined_symbols": [string],
    "potential_conflicts": [string],
    "notes": [string]
  },
  "confidence": number
}

Rules:
- First complete component_applicability. Use only components judged applicable,
  plus clearly marked optional assumptions when the user problem is underspecified.
- If a Harness component selector is provided, it is binding:
  components with status="selected" must be represented in the final formulas.
  Components with status="omitted" must not appear in sets, parameters,
  variables, objective terms, constraints, or validation notes because the user
  excluded them. Components with status="not_selected" are not default
  requirements, but may be used as clearly marked evidence-supported assumptions
  when they improve a paper-level HHC formulation.
- Evidence is for analogy and grounding. Do not reproduce a source paper's full
  formulation, full variable set, full constraint list, or equation numbering.
- Do not copy source-paper equation labels into constraint names. Use semantic
  names such as "Assignment coverage" or "Time-window feasibility".
- Preserve mathematical notation only when it is generic and useful. Prefer new
  symbols that fit the user's problem over mechanically reusing source symbols.
- Use source_chunk_ids copied exactly from the retrieved evidence.
- Keep the output concise. Include enough variables and constraints to be
  coherent, but do not include components just because they appear in evidence.
- If evidence does not support a field, keep it empty and add a note in
  missing_information or validation.notes.
- The objective and constraints may be a template if the user problem lacks
  numeric details, but the template must be mathematically coherent.
- Every symbol used in objective and constraints must be defined in sets,
  parameters, or decision_variables. If subtour-elimination variables,
  precedence variables, or auxiliary slack variables are used, define them.
- For scheduling/sequencing models, avoid infeasible predecessor/successor
  equalities unless dummy start/end jobs are explicitly defined. Pairwise
  ordering variables must be linked to same-machine assignment, and big-M
  non-overlap constraints must activate only when two jobs are assigned to the
  same machine.
- In validation, explicitly flag any formula that is a modeling assumption or
  template rather than directly supported by retrieved evidence.
- validation.undefined_symbols should be empty unless a symbol is intentionally
  left for the user to define. Prefer defining needed auxiliary symbols instead.
- If most variables or constraints would be identical to one retrieved paper,
  revise the formulation before finalizing and add a validation note explaining
  how the model was adapted to the user's problem.
- Do not include markdown fences or explanatory prose outside JSON.
"""


JSON_REPAIR_SYSTEM_PROMPT = """You repair malformed JSON.

Return one strict JSON object only. Do not add markdown fences or explanations.
Preserve the original content as much as possible, but fix truncation, missing
commas, unescaped characters, and invalid JSON syntax. If a field is incomplete,
replace it with a concise valid value rather than leaving invalid JSON.
"""


MODEL_REVISION_SYSTEM_PROMPT = """You revise mathematical-model JSON.

Return one strict JSON object only. Do not add markdown fences or explanations.
Preserve the user's problem and useful evidence grounding, but fix the listed
issues. The revised model must be an adapted model for the user's problem, not
a copy of a retrieved paper formulation.
"""


MODELING_AGENT_REVIEW_PROMPT = """You are a mathematical-modeling quality critic.

Your job is not to write the model. Your job is to decide whether the current
model should be accepted or revised based on the verifier, quality rubric, and
the user's problem specification.
"""


LIGHTWEIGHT_AGENT_PLAN_SYSTEM_PROMPT = """You are the Planning Agent in a lightweight mathematical-modeling team.

Your job is to make a compact modeling decision record for the user's problem.
Do not write formulas. Do not copy a paper model. Decide which components the
Formula Agent should use, which optional evidence-supported structures may be
added, and which risks the Verification Agent must check.

Return strict JSON only, with this schema:
{
  "problem_type": string,
  "modeling_scope": string,
  "component_decisions": [
    {"component": string, "decision": "use|omit|assumption", "reason": string}
  ],
  "formula_agent_brief": [string],
  "verification_focus": [string],
  "expected_model_size": "compact|medium|rich"
}

Keep the output compact. Prefer 6-10 component decisions.
"""


MODEL_POLISH_SYSTEM_PROMPT = """You are the Polisher Agent in a lightweight mathematical-modeling team.

Your job is to improve academic exposition without changing the mathematical
structure. You may refine descriptions, assumptions, validation notes, problem
analysis text, component reasons, and reusable_parts. You must not change any
formula, symbol, domain, source_chunk_ids, component applicability decision, or
the number/order of constraints.

Return strict JSON only with this schema:
{
  "problem_analysis": object,
  "reference_models": [object],
  "component_applicability": [object],
  "objective_description": string,
  "constraint_descriptions": [string],
  "assumptions": [string],
  "validation_notes": [string],
  "polish_summary": [string]
}

The constraint_descriptions array must have exactly the same length and order
as the current constraints. Do not include markdown.
"""


MODEL_PLAN_SYSTEM_PROMPT = """You are an operations-research modeling planner.

Before writing any mathematical formulation, decide the modeling scope. The
plan is not a template. It is a compact decision record that tells the final
model generator which components to use, which to omit, and which risks to
check.

Return strict JSON only, with this schema:
{
  "problem_type": string,
  "modeling_scope": string,
  "component_decisions": [
    {
      "component": string,
      "decision": "use|omit|assumption",
      "reason": string
    }
  ],
  "retrieval_focus": [string],
  "modeling_steps": [string],
  "quality_checks": [string],
  "expected_model_size": "compact|medium|rich"
}

Rules:
- Use "use" only when the user problem requires the component.
- Use "omit" when the component appears in retrieved evidence but is not needed.
- Use "assumption" only when the component is reasonable but not stated by the user.
- Do not copy a paper's formulation. Decide the scope of a new model.
- Keep the plan concise.
"""


MODEL_BLUEPRINT_SYSTEM_PROMPT = """You are an operations-research model architect.

Convert the modeling plan into a formulation blueprint before writing formulas.
The blueprint is not the final model and not a paper template. It should specify
which sets, parameters, variables, objective terms, and constraint groups the
final model should contain.

Return strict JSON only, with this schema:
{
  "formulation_type": string,
  "sets": [
    {"name": string, "role": string, "include": boolean, "reason": string}
  ],
  "parameters": [
    {"name": string, "role": string, "include": boolean, "reason": string}
  ],
  "decision_variables": [
    {"name": string, "role": string, "include": boolean, "reason": string}
  ],
  "objective_terms": [
    {"name": string, "role": string, "include": boolean, "reason": string}
  ],
  "constraint_groups": [
    {"name": string, "role": string, "include": boolean, "reason": string, "risk_notes": [string]}
  ],
  "omitted_components": [string],
  "notation_guidance": [string],
  "solver_readiness_notes": [string]
}

Rules:
- Include only components required by the user problem or explicitly marked as
  assumptions in the plan.
- If the domain is home health care, optional structures such as VIP classes,
  outsourcing, caregiver skills, workload balance, synchronization, breaks,
  overtime, preferences, and multi-center assignment may be included when they
  are requested, strongly implied, or supported by retrieved evidence. Mark
  evidence-supported additions as assumptions and adapt their notation.
- Use semantic component names. Do not copy source-paper equation names,
  numbering, or complete variable lists.
- Keep the blueprint compact enough for the final generator to follow.
"""


def _modeling_retrieval_query(problem: str, skill: ModelingSkill) -> str:
    return skill.build_queries(problem)[0]


def _page_label(raw_page: Any) -> str:
    if isinstance(raw_page, list):
        return "-".join(str(p) for p in raw_page) if raw_page else "N/A"
    if raw_page in ("", None):
        return "N/A"
    return str(raw_page)


def _evidence_content(chunk: dict[str, Any]) -> str:
    if chunk.get("modality") == "figure":
        return (
            chunk.get("caption_merged")
            or chunk.get("content")
            or chunk.get("caption")
            or ""
        ).strip()
    return str(chunk.get("content", "")).strip()


def _chunk_content_budget(chunk: dict[str, Any]) -> int:
    chunk_type = str(chunk.get("chunk_type", ""))
    modality = str(chunk.get("modality", ""))
    if chunk_type in {"sets", "parameters", "variables", "objective", "constraints"}:
        return 1800
    if chunk_type == "model_card":
        return 1200
    if modality == "model_region":
        return 900
    if modality == "figure":
        return 900
    return 900


def _build_evidence_blocks(
    evidence: list[dict[str, Any]],
    *,
    max_items: int | None = None,
    compact: bool = False,
) -> list[str]:
    blocks: list[str] = []
    selected = evidence[:max_items] if max_items is not None else evidence
    for chunk in selected:
        chunk_id = chunk.get("chunk_id") or chunk.get("figure_id") or ""
        if not chunk_id:
            continue
        content = _evidence_content(chunk)
        if not content:
            continue
        score = chunk.get("rerank_score", chunk.get("score", 0.0))
        budget = min(_chunk_content_budget(chunk), 450 if compact else _chunk_content_budget(chunk))
        blocks.append(
            "[{chunk_id}] ({paper_id}, page {page}, modality={modality}, "
            "chunk_type={chunk_type}, score={score})\n{content}".format(
                chunk_id=chunk_id,
                paper_id=chunk.get("paper_id", ""),
                page=_page_label(chunk.get("page")),
                modality=chunk.get("modality", "text"),
                chunk_type=chunk.get("chunk_type", ""),
                score=f"{float(score):.4f}" if isinstance(score, (int, float)) else score,
                content=content[:budget],
            )
        )
    return blocks


def _build_user_prompt(
    problem: str,
    evidence: list[dict[str, Any]],
    *,
    skill: ModelingSkill,
    problem_spec: dict[str, Any] | None = None,
    modeling_plan: dict[str, Any] | None = None,
    modeling_blueprint: dict[str, Any] | None = None,
    harness_draft: dict[str, Any] | None = None,
    compact: bool = False,
) -> str:
    evidence_blocks = _build_evidence_blocks(
        evidence,
        max_items=5 if compact else None,
        compact=compact,
    )
    evidence_section = "\n\n".join(evidence_blocks) if evidence_blocks else "(no evidence)"
    valid_ids = ", ".join(
        f"[{chunk.get('chunk_id') or chunk.get('figure_id')}]"
        for chunk in evidence
        if chunk.get("chunk_id") or chunk.get("figure_id")
    )
    candidate_components = "\n".join(f"- {item}" for item in skill.candidate_components)
    quality_checks = "\n".join(f"- {item}" for item in skill.quality_checks)
    spec_section = (
        json.dumps(_compact_problem_spec(problem_spec) if compact else problem_spec, ensure_ascii=False, indent=2)
        if problem_spec
        else "(no explicit problem spec)"
    )
    plan_section = (
        json.dumps(_compact_modeling_plan(modeling_plan) if compact else modeling_plan, ensure_ascii=False, indent=2)
        if modeling_plan
        else "(no explicit plan)"
    )
    blueprint_section = (
        json.dumps(_compact_blueprint(modeling_blueprint) if compact else modeling_blueprint, ensure_ascii=False, indent=2)
        if modeling_blueprint
        else "(no explicit blueprint)"
    )
    harness_section = _build_harness_control_section(harness_draft, compact=compact)
    return f"""User modeling problem:
{problem}

Selected modeling skill:
{skill.name} — {skill.description}

Candidate components to consider, not mandatory:
{candidate_components}

Skill-specific modeling guidance:
{skill.modeling_guidance}

Skill-specific quality checks:
{quality_checks or "- No additional skill-specific checks."}

Problem spec to obey:
{spec_section}

Modeling plan to follow:
{plan_section}

Modeling blueprint to follow:
{blueprint_section}

Harness component selector and symbol scaffold to obey:
{harness_section}

Hard exclusion rule:
- Do not introduce any term listed under forbidden_terms.
- If a forbidden term appears in retrieved evidence, ignore it for this user
  problem unless the user explicitly asks for that component.
- Do not add omitted capacity/workload limits unless capacity is selected or
  explicitly required by a new user goal.
- Do not add omitted time windows unless time_window is selected or explicitly
  required by the user.
- Do not add omitted skill, outsourcing, priority, fairness, or open-route
  structures unless the matching component is selected or explicitly required.
- Not-selected HHC components may still be introduced as clearly labeled
  evidence-supported assumptions in paper-level generation, provided they are
  not listed under forbidden_terms and do not copy a source paper verbatim.

Retrieved modeling evidence:
{evidence_section}

Valid source_chunk_ids:
{valid_ids or "(none)"}

Generate a grounded mathematical modeling draft for the user problem.
Before defining variables and constraints, use the Problem Spec as the binding
task contract and the Harness component selector as the binding component
boundary. component_applicability must mirror hard exclusions: selected
components are applicable=true; omitted components are applicable=false.
Not-selected components should not be treated as mandatory, but may become
applicable=true when they are useful evidence-supported assumptions for a richer
paper-level HHC model.
Do not include all candidate components by default.
Do not copy the retrieved paper's complete model. Use evidence as modeling
inspiration and cite it, but adapt notation, variables, objectives, and
constraints to the user's problem.
If problem_spec.generation_depth is "paper_level", produce a richer formulation
with all selected objective terms and constraint groups covered. Otherwise keep
the output medium-sized: prefer 5-8 core constraints unless the user asks for a
full paper-level formulation.
Return JSON only.
"""


def _build_harness_control_section(harness_draft: dict[str, Any] | None, *, compact: bool = False) -> str:
    if not harness_draft:
        return "(no harness selector)"
    selector = harness_draft.get("component_selector") or []
    selected = [
        str(item.get("component", ""))
        for item in selector
        if isinstance(item, dict) and item.get("status") == "selected"
    ]
    omitted = [
        str(item.get("component", ""))
        for item in selector
        if isinstance(item, dict) and item.get("status") == "omitted"
    ]
    not_selected = [
        str(item.get("component", ""))
        for item in selector
        if isinstance(item, dict) and item.get("status") == "not_selected"
    ]
    spec = harness_draft.get("model_spec") or {}
    symbols = harness_draft.get("symbol_plan") or {}
    if compact:
        compact_payload = {
            "binding_rule": (
                "selected required; omitted forbidden; not_selected optional only as evidence-supported assumptions"
            ),
            "selected_components": selected,
            "omitted_components_forbidden": omitted,
            "not_selected_components": not_selected[:8],
            "forbidden_terms": _forbidden_terms_for_components(omitted),
            "constraint_groups_to_cover": [
                {
                    "name": item.get("name", ""),
                    "operator": item.get("operator", ""),
                }
                for item in spec.get("constraint_groups", []) or []
                if isinstance(item, dict)
            ],
            "objective_terms_to_cover": [
                {
                    "name": item.get("name", ""),
                    "operator": item.get("operator", ""),
                }
                for item in spec.get("objective_terms", []) or []
                if isinstance(item, dict)
            ],
        }
        return json.dumps(compact_payload, ensure_ascii=False, indent=2)
    compact = {
        "binding_rule": (
            "Selected components must be represented. Omitted components are "
            "forbidden because the user excluded them. Not-selected components "
            "are not default requirements, but they may be introduced as clearly "
            "marked evidence-supported assumptions for richer paper-level HHC "
            "models when they are not forbidden."
        ),
        "selected_components": selected,
        "omitted_components_forbidden": omitted,
        "not_selected_components": not_selected,
        "forbidden_terms": _forbidden_terms_for_components(omitted),
        "recommended_symbols": {
            "sets": symbols.get("sets", []),
            "parameters": symbols.get("parameters", []),
            "variables": symbols.get("variables", []),
        },
        "constraint_groups_to_cover": spec.get("constraint_groups", []),
        "objective_terms_to_cover": spec.get("objective_terms", []),
    }
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _forbidden_terms_for_components(components: list[str]) -> dict[str, list[str]]:
    marker_map: dict[str, list[str]] = {
        "capacity": ["capacity", "workload limit", "route duration limit", "L_c", "U_c"],
        "balance": ["balance", "fairness", "imbalance", "B", "delta B"],
        "skill_matching": ["skill", "qualification", "q_c", "r_p"],
        "outsourcing": ["outsourcing", "outsource", "external service", "rejection", "o_p", "pi_p"],
        "priority_class": ["VIP", "ordinary patient", "priority", "omega_p"],
        "time_window": ["time window", "earliest/latest window", "b_p"],
        "open_route": ["open route", "do not return", "no return", "terminate away from depot"],
    }
    return {
        component: marker_map[component]
        for component in components
        if component in marker_map
    }


def _compact_problem_spec(spec: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(spec, dict):
        return spec
    return {
        "domain": spec.get("domain"),
        "problem_type": spec.get("problem_type"),
        "generation_depth": spec.get("generation_depth"),
        "task_summary": spec.get("task_summary"),
        "selected_components": spec.get("selected_components") or [],
        "forbidden_components": spec.get("forbidden_components") or [],
        "not_selected_components": (spec.get("not_selected_components") or [])[:8],
        "required_objective_terms": spec.get("required_objective_terms") or [],
        "required_constraint_groups": spec.get("required_constraint_groups") or [],
        "missing_information": (spec.get("missing_information") or [])[:8],
    }


def _compact_modeling_plan(plan: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(plan, dict):
        return plan
    return {
        "problem_type": plan.get("problem_type"),
        "modeling_scope": plan.get("modeling_scope"),
        "component_decisions": (plan.get("component_decisions") or [])[:12],
        "modeling_steps": (plan.get("modeling_steps") or [])[:8],
        "quality_checks": (plan.get("quality_checks") or [])[:8],
        "expected_model_size": plan.get("expected_model_size"),
    }


def _compact_blueprint(blueprint: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(blueprint, dict):
        return blueprint
    return {
        "formulation_type": blueprint.get("formulation_type"),
        "decision_variables": [
            item for item in (blueprint.get("decision_variables") or []) if isinstance(item, dict) and item.get("include")
        ][:12],
        "objective_terms": [
            item for item in (blueprint.get("objective_terms") or []) if isinstance(item, dict) and item.get("include")
        ][:8],
        "constraint_groups": [
            item for item in (blueprint.get("constraint_groups") or []) if isinstance(item, dict) and item.get("include")
        ][:14],
        "omitted_components": (blueprint.get("omitted_components") or [])[:10],
        "solver_readiness_notes": (blueprint.get("solver_readiness_notes") or [])[:6],
    }


def _build_problem_spec(
    *,
    problem: str,
    skill: ModelingSkill,
    modeling_plan: dict[str, Any] | None,
    harness_draft: dict[str, Any],
    evidence: list[dict[str, Any]],
    generation_depth: str,
) -> dict[str, Any]:
    selector = harness_draft.get("component_selector") or []
    selected = [
        str(item.get("component", ""))
        for item in selector
        if isinstance(item, dict) and item.get("status") == "selected"
    ]
    omitted = [
        str(item.get("component", ""))
        for item in selector
        if isinstance(item, dict) and item.get("status") == "omitted"
    ]
    not_selected = [
        str(item.get("component", ""))
        for item in selector
        if isinstance(item, dict) and item.get("status") == "not_selected"
    ]
    model_spec = harness_draft.get("model_spec") or {}
    symbol_plan = harness_draft.get("symbol_plan") or {}
    validation = harness_draft.get("validation") or {}
    evidence_ids = [
        str(item.get("chunk_id") or item.get("figure_id"))
        for item in evidence[:8]
        if item.get("chunk_id") or item.get("figure_id")
    ]
    objective_terms = [
        {
            "name": str(item.get("name", "")),
            "description": str(item.get("description", "")),
            "operator": str(item.get("operator", "")),
        }
        for item in model_spec.get("objective_terms", []) or []
        if isinstance(item, dict)
    ]
    constraint_groups = [
        {
            "name": str(item.get("name", "")),
            "purpose": str(item.get("purpose", "")),
            "operator": str(item.get("operator", "")),
        }
        for item in model_spec.get("constraint_groups", []) or []
        if isinstance(item, dict)
    ]
    plan_missing = []
    plan_requirements = []
    if isinstance(modeling_plan, dict):
        for key in ("missing_information", "unknowns", "assumptions_needed"):
            values = modeling_plan.get(key) or []
            if isinstance(values, list):
                plan_missing.extend(str(value) for value in values if value)
        for key in ("requirements", "modeling_steps", "quality_checks"):
            values = modeling_plan.get(key) or []
            if isinstance(values, list):
                plan_requirements.extend(str(value) for value in values if value)
    return {
        "domain": skill.name,
        "problem_type": model_spec.get("problem_type") or harness_draft.get("problem_type") or skill.name,
        "generation_depth": generation_depth,
        "model_type": "MILP_or_mixed_integer_optimization",
        "task_summary": problem.strip(),
        "selected_components": selected,
        "forbidden_components": omitted,
        "not_selected_components": not_selected,
        "forbidden_terms": _forbidden_terms_for_components(omitted),
        "required_objective_terms": objective_terms,
        "required_constraint_groups": constraint_groups,
        "recommended_symbols": {
            "sets": symbol_plan.get("sets", []),
            "parameters": symbol_plan.get("parameters", []),
            "variables": symbol_plan.get("variables", []),
        },
        "requirements_from_plan": plan_requirements[:12],
        "missing_information": _dedupe_strings(
            plan_missing + _derive_missing_information(selected, omitted + not_selected)
        ),
        "evidence_ids": evidence_ids,
        "validation_status": validation.get("status", ""),
    }


def _derive_missing_information(selected: list[str], excluded: list[str]) -> list[str]:
    missing: list[str] = []
    if "time_window" in excluded:
        missing.append("time windows are not specified or not selected")
    if "skill_matching" in excluded:
        missing.append("caregiver skill levels and patient skill requirements are not specified or not selected")
    if "capacity" in excluded:
        missing.append("caregiver workload or route-duration limits are not specified or not selected")
    if "outsourcing" in excluded:
        missing.append("outsourcing or rejection is not allowed or not specified")
    if "priority_class" in excluded:
        missing.append("patient priority classes are not specified or not selected")
    if "routing_flow" in selected and "time_propagation" not in selected:
        missing.append("route timing is not fully specified beyond routing arcs")
    return missing


def _dedupe_strings(values: list[str]) -> list[str]:
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


def _dedupe_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in evidence:
        key = str(item.get("chunk_id") or item.get("figure_id") or "")
        if not key:
            key = f"{item.get('paper_id')}:{item.get('page')}:{item.get('modality')}:{len(deduped)}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _merge_evidence_candidates(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged_by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for group in groups:
        for item in group:
            key = str(item.get("chunk_id") or item.get("figure_id") or "")
            if not key:
                key = f"{item.get('paper_id')}:{item.get('page')}:{item.get('modality')}:{len(order)}"
            if key not in merged_by_id:
                merged_by_id[key] = dict(item)
                order.append(key)
                continue
            current = merged_by_id[key]
            current_score = float(current.get("score", 0.0) or 0.0)
            new_score = float(item.get("score", 0.0) or 0.0)
            if new_score > current_score:
                current["score"] = item.get("score", current.get("score", 0.0))
            for metadata_key in (
                "metadata_score",
                "modeling_boosted_score",
                "model_elements",
                "operator_hints",
                "domain_signals",
                "hhc_signals",
                "formula_signal_count",
                "model_evidence_score",
            ):
                if metadata_key in item and metadata_key not in current:
                    current[metadata_key] = item[metadata_key]
    return [merged_by_id[key] for key in order]


def _evidence_priority(item: dict[str, Any]) -> tuple[int, float]:
    """Rank evidence for model generation prompt budget."""
    chunk_type = str(item.get("chunk_type", ""))
    modality = str(item.get("modality", ""))
    score = float(
        item.get(
            "metadata_score",
            item.get("modeling_boosted_score", item.get("score", 0.0)),
        )
        or 0.0
    )
    type_priority = {
        "objective": 100,
        "constraints": 98,
        "variables": 96,
        "parameters": 94,
        "sets": 92,
        "model_card": 88,
    }.get(chunk_type, 0)
    if modality == "model_region":
        type_priority = max(type_priority, 80)
    elif modality == "math_model":
        type_priority = max(type_priority, 86)
    elif modality == "figure":
        type_priority = max(type_priority, 40)
    return type_priority, score


def _select_prompt_evidence(
    text_results: list[dict[str, Any]],
    figure_results: list[dict[str, Any]],
    *,
    skill: ModelingSkill,
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    text = _dedupe_evidence(text_results)
    figures = _dedupe_evidence(figure_results)
    text = sorted(text, key=_evidence_priority, reverse=True)
    if skill.name == "home_health_care_routing_scheduling":
        # Keep focused model evidence; large prompts make API models time out easily.
        cap = min(max(top_k, 3), 4)
    else:
        cap = top_k
    return {
        "text_results": text[:cap],
        "figure_results": figures[: min(1, len(figures))],
    }


def _retrieve_with_skill(
    service: QueryService,
    *,
    problem: str,
    skill: ModelingSkill,
    top_k: int,
    paper_id: str | None,
) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    queries = skill.build_queries(problem)
    all_text: list[dict[str, Any]] = []
    all_figures: list[dict[str, Any]] = []
    per_query_k = max(top_k, 8) if skill.name != "generic_optimization" else top_k
    for query in queries:
        retrieval = service.retrieve_core(
            query,
            top_k=per_query_k,
            paper_id=paper_id,
            figure_top_k=CFG.retriever.top_k_fig,
        )
        all_text.extend(retrieval["text_results"])
        all_figures.extend(retrieval["figure_results"])
        if is_math_modeling_query(query) or skill.name != "generic_optimization":
            all_text.extend(
                service.retrieve_model_metadata(
                    query,
                    top_k=max(top_k, 8),
                    paper_id=paper_id,
                )
            )

    # Keep enough domain evidence for rich skills, but do not let figures crowd out formulas.
    all_text = _merge_evidence_candidates(all_text)
    return queries[0], _select_prompt_evidence(
        all_text,
        all_figures,
        skill=skill,
        top_k=top_k,
    )


def _retrieve_metadata_fallback_with_skill(
    service: QueryService,
    *,
    problem: str,
    skill: ModelingSkill,
    top_k: int,
    paper_id: str | None,
) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    """Retrieve model-aware evidence without calling the embedding API."""
    queries = skill.build_queries(problem)
    all_text: list[dict[str, Any]] = []
    for query in queries:
        all_text.extend(
            service.retrieve_model_metadata(
                query,
                top_k=max(top_k, 8),
                paper_id=paper_id,
            )
        )
    all_text = _merge_evidence_candidates(all_text)
    return queries[0], _select_prompt_evidence(
        all_text,
        [],
        skill=skill,
        top_k=top_k,
    )


def _strip_json_fence(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response."""
    stripped = _strip_json_fence(text)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("model generation output is not a JSON object")
    return data


def _repair_json_output(
    raw_output: str,
    *,
    max_tokens: int,
) -> tuple[dict[str, Any] | None, str, str]:
    repair_messages = [
        {"role": "system", "content": JSON_REPAIR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Repair this malformed model-generation JSON. Return JSON only.\n\n"
                f"{raw_output}"
            ),
        },
    ]
    repaired = generate(
        messages=repair_messages,
        model=CFG.generator.model_name,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    try:
        return _normalize_generated_model(parse_json_object(repaired)), repaired, ""
    except Exception as exc:
        return None, repaired, str(exc)


def _build_plan_prompt(
    problem: str,
    evidence: list[dict[str, Any]],
    *,
    skill: ModelingSkill,
) -> str:
    evidence_blocks = _build_evidence_blocks(evidence[:4])
    evidence_section = "\n\n".join(evidence_blocks) if evidence_blocks else "(no evidence)"
    candidate_components = "\n".join(f"- {item}" for item in skill.candidate_components)
    quality_checks = "\n".join(f"- {item}" for item in skill.quality_checks)
    return f"""User modeling problem:
{problem}

Selected modeling skill:
{skill.name} - {skill.description}

Candidate components:
{candidate_components}

Skill-specific quality checks:
{quality_checks or "- No additional skill-specific checks."}

Top retrieved evidence snippets:
{evidence_section}

Build a concise modeling plan. Decide what to use, omit, or treat as an
assumption before the final model is written.
Return JSON only.
"""


def _build_modeling_plan(
    problem: str,
    evidence: list[dict[str, Any]],
    *,
    skill: ModelingSkill,
    max_tokens: int,
    reasoning_effort: str | None = None,
    timeout: float | None = None,
    client_max_retries: int | None = None,
) -> tuple[dict[str, Any] | None, str, str]:
    messages = [
        {"role": "system", "content": MODEL_PLAN_SYSTEM_PROMPT},
        {"role": "user", "content": _build_plan_prompt(problem, evidence, skill=skill)},
    ]
    raw = ""
    last_error = ""
    for attempt in range(1, 3):
        try:
            raw = generate(
                messages=messages,
                model=CFG.generator.model_name,
                temperature=0.0,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                max_retries=client_max_retries,
            )
            break
        except Exception as exc:
            last_error = str(exc)
            if attempt < 2:
                time.sleep(1.5 * attempt)
    if not raw:
        return None, "", last_error
    try:
        return parse_json_object(raw), raw, ""
    except Exception as exc:
        return None, raw, str(exc)


def _build_lightweight_agent_plan_prompt(
    problem: str,
    evidence: list[dict[str, Any]],
    *,
    skill: ModelingSkill,
) -> str:
    evidence_blocks = _build_evidence_blocks(evidence, max_items=3, compact=True)
    evidence_section = "\n\n".join(evidence_blocks) if evidence_blocks else "(no evidence)"
    candidate_components = "\n".join(f"- {item}" for item in skill.candidate_components[:14])
    quality_checks = "\n".join(f"- {item}" for item in skill.quality_checks[:8])
    return f"""User modeling problem:
{problem}

Selected modeling skill:
{skill.name} - {skill.description}

Candidate components:
{candidate_components}

Verification checks to keep in mind:
{quality_checks or "- No additional skill-specific checks."}

Top retrieved modeling evidence:
{evidence_section}

Produce a compact plan for a lightweight three-stage modeling agent:
1. Planning Agent: decide modeling scope and components.
2. Formula Agent: use Harness to render a coherent formula skeleton.
3. Verification Agent: verify symbols, objectives, constraints, and evidence grounding.

Return JSON only.
"""


def _build_lightweight_agent_plan(
    problem: str,
    evidence: list[dict[str, Any]],
    *,
    skill: ModelingSkill,
    max_tokens: int,
    reasoning_effort: str | None = None,
    timeout: float | None = None,
    client_max_retries: int | None = None,
) -> tuple[dict[str, Any] | None, str, str]:
    messages = [
        {"role": "system", "content": LIGHTWEIGHT_AGENT_PLAN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_lightweight_agent_plan_prompt(
                problem,
                evidence,
                skill=skill,
            ),
        },
    ]
    raw = ""
    try:
        raw = generate(
            messages=messages,
            model=CFG.generator.model_name,
            temperature=0.0,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            max_retries=client_max_retries,
        )
    except Exception as exc:
        return None, "", str(exc)
    try:
        return parse_json_object(raw), raw, ""
    except Exception as exc:
        return None, raw, str(exc)


def _build_blueprint_prompt(
    problem: str,
    evidence: list[dict[str, Any]],
    *,
    skill: ModelingSkill,
    modeling_plan: dict[str, Any] | None,
    problem_spec: dict[str, Any] | None = None,
) -> str:
    evidence_blocks = _build_evidence_blocks(evidence[:5])
    evidence_section = "\n\n".join(evidence_blocks) if evidence_blocks else "(no evidence)"
    plan_section = (
        json.dumps(modeling_plan, ensure_ascii=False, indent=2)
        if modeling_plan
        else "(no explicit plan)"
    )
    spec_section = (
        json.dumps(problem_spec, ensure_ascii=False, indent=2)
        if problem_spec
        else "(no explicit problem spec)"
    )
    quality_checks = "\n".join(f"- {item}" for item in skill.quality_checks)
    return f"""User modeling problem:
{problem}

Selected modeling skill:
{skill.name} - {skill.description}

Modeling plan:
{plan_section}

Problem spec:
{spec_section}

Skill-specific quality checks:
{quality_checks or "- No additional skill-specific checks."}

Top retrieved evidence snippets:
{evidence_section}

Build a formulation blueprint. The final model generator will follow this
blueprint, so explicitly include or omit each major modeling structure.
Return JSON only.
"""


def _build_modeling_blueprint(
    problem: str,
    evidence: list[dict[str, Any]],
    *,
    skill: ModelingSkill,
    modeling_plan: dict[str, Any] | None,
    problem_spec: dict[str, Any] | None = None,
    max_tokens: int,
    reasoning_effort: str | None = None,
    timeout: float | None = None,
    client_max_retries: int | None = None,
) -> tuple[dict[str, Any] | None, str, str]:
    messages = [
        {"role": "system", "content": MODEL_BLUEPRINT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_blueprint_prompt(
                problem,
                evidence,
                skill=skill,
                modeling_plan=modeling_plan,
                problem_spec=problem_spec,
            ),
        },
    ]
    raw = ""
    last_error = ""
    for attempt in range(1, 3):
        try:
            raw = generate(
                messages=messages,
                model=CFG.generator.model_name,
                temperature=0.0,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                max_retries=client_max_retries,
            )
            break
        except Exception as exc:
            last_error = str(exc)
            if attempt < 2:
                time.sleep(1.5 * attempt)
    if not raw:
        return None, "", last_error
    try:
        return parse_json_object(raw), raw, ""
    except Exception as exc:
        return None, raw, str(exc)


def _revise_model_output(
    *,
    problem: str,
    model: dict[str, Any],
    warnings: list[str],
    evidence: list[dict[str, Any]],
    skill: ModelingSkill,
    harness_draft: dict[str, Any] | None = None,
    max_tokens: int,
    reasoning_effort: str | None = None,
    timeout: float | None = None,
    client_max_retries: int | None = None,
) -> tuple[dict[str, Any] | None, str, str]:
    evidence_blocks = _build_evidence_blocks(evidence)
    evidence_section = "\n\n".join(evidence_blocks) if evidence_blocks else "(no evidence)"
    valid_ids = ", ".join(
        f"[{chunk.get('chunk_id') or chunk.get('figure_id')}]"
        for chunk in evidence
        if chunk.get("chunk_id") or chunk.get("figure_id")
    )
    revision_prompt = f"""User modeling problem:
{problem}

Selected modeling skill:
{skill.name} - {skill.description}

Issues to fix:
{json.dumps(warnings, ensure_ascii=False, indent=2)}

Current model JSON:
{json.dumps(model, ensure_ascii=False, indent=2)}

Retrieved modeling evidence:
{evidence_section}

Valid source_chunk_ids:
{valid_ids or "(none)"}

Harness component selector and symbol scaffold to obey:
{_build_harness_control_section(harness_draft)}

Revision requirements:
- Keep component_applicability.applicable as a boolean only.
- Treat the Harness selector as binding. Remove every omitted component from
  sets, parameters, decision variables, objective, constraints, and validation.
- Ensure every selected Harness component appears in objective or constraints.
- Remove source-paper equation labels from all names, formulas, notes, and
  reference reusable_parts. Do not mention removed equation labels in the
  revised output.
- Do not reproduce the source paper's full variable set or full constraint list.
- Use semantic constraint names and notation adapted to the user's problem.
- Omit or mark as an assumption any component that the user problem does not
  specify. Evidence-supported optional HHC components may be included as
  assumptions, but do not present them as if the user explicitly required them.
- Define every symbol used in objective and constraints. If a listed issue says
  there are undefined symbols, either add proper definitions or remove the
  formulas that require them.
- Return strict JSON matching the original schema.
"""
    revised = generate(
        messages=[
            {"role": "system", "content": MODEL_REVISION_SYSTEM_PROMPT},
            {"role": "user", "content": revision_prompt},
        ],
        model=CFG.generator.model_name,
        temperature=0.0,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        max_retries=client_max_retries,
    )
    try:
        return _normalize_generated_model(parse_json_object(revised)), revised, ""
    except Exception as exc:
        return None, revised, str(exc)


def _compact_model_for_polish(model: dict[str, Any]) -> dict[str, Any]:
    constraints = model.get("constraints") or []
    return {
        "problem_analysis": model.get("problem_analysis") or {},
        "reference_models": (model.get("reference_models") or [])[:6],
        "component_applicability": model.get("component_applicability") or [],
        "sets": model.get("sets") or [],
        "parameters": model.get("parameters") or [],
        "decision_variables": model.get("decision_variables") or [],
        "objective": model.get("objective") or {},
        "constraints": [
            {
                "name": item.get("name", ""),
                "formula": item.get("formula", ""),
                "description": item.get("description", ""),
                "source_chunk_ids": item.get("source_chunk_ids", []),
            }
            for item in constraints
            if isinstance(item, dict)
        ],
        "assumptions": model.get("assumptions") or [],
        "validation": model.get("validation") or {},
    }


def _apply_polish_patch(model: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    polished = copy.deepcopy(model)

    if isinstance(patch.get("problem_analysis"), dict):
        original = polished.get("problem_analysis") if isinstance(polished.get("problem_analysis"), dict) else {}
        merged = dict(original)
        for key in ("problem_type", "entities", "goals", "requirements", "missing_information"):
            if key in patch["problem_analysis"]:
                merged[key] = patch["problem_analysis"][key]
        polished["problem_analysis"] = merged

    if isinstance(patch.get("reference_models"), list):
        cleaned_refs = [item for item in patch["reference_models"] if isinstance(item, dict)]
        if cleaned_refs:
            polished["reference_models"] = cleaned_refs

    if isinstance(patch.get("component_applicability"), list):
        original_items = polished.get("component_applicability") or []
        patched_items = patch.get("component_applicability") or []
        if len(patched_items) == len(original_items):
            merged_items: list[dict[str, Any]] = []
            for original, patched in zip(original_items, patched_items):
                if not isinstance(original, dict):
                    merged_items.append(original)
                    continue
                item = dict(original)
                if isinstance(patched, dict):
                    if "reason" in patched:
                        item["reason"] = patched.get("reason", item.get("reason", ""))
                    if "component" in patched and patched.get("component") == original.get("component"):
                        item["component"] = patched.get("component")
                merged_items.append(item)
            polished["component_applicability"] = merged_items

    objective_description = patch.get("objective_description")
    if isinstance(objective_description, str) and isinstance(polished.get("objective"), dict):
        polished["objective"]["description"] = objective_description

    constraint_descriptions = patch.get("constraint_descriptions")
    constraints = polished.get("constraints")
    if isinstance(constraint_descriptions, list) and isinstance(constraints, list):
        if len(constraint_descriptions) == len(constraints):
            for constraint, description in zip(constraints, constraint_descriptions):
                if isinstance(constraint, dict) and isinstance(description, str):
                    constraint["description"] = description

    if isinstance(patch.get("assumptions"), list):
        polished["assumptions"] = [str(item) for item in patch["assumptions"] if str(item).strip()]

    validation_notes = patch.get("validation_notes")
    if isinstance(validation_notes, list):
        validation = polished.get("validation")
        if not isinstance(validation, dict):
            validation = {"undefined_symbols": [], "potential_conflicts": [], "notes": []}
        validation["notes"] = [str(item) for item in validation_notes if str(item).strip()]
        polished["validation"] = validation

    summary = patch.get("polish_summary")
    if isinstance(summary, list):
        validation = polished.get("validation")
        if isinstance(validation, dict):
            notes = list(validation.get("notes") or [])
            notes.extend(f"Polisher: {item}" for item in summary if str(item).strip())
            validation["notes"] = _dedupe_strings([str(item) for item in notes])

    return _normalize_generated_model(polished)


def _model_structure_signature(model: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(model, dict):
        return {}
    objective = model.get("objective") if isinstance(model.get("objective"), dict) else {}
    constraints = model.get("constraints") if isinstance(model.get("constraints"), list) else []
    return {
        "sets": [item.get("symbol", "") for item in model.get("sets") or [] if isinstance(item, dict)],
        "parameters": [
            item.get("symbol", "") for item in model.get("parameters") or [] if isinstance(item, dict)
        ],
        "decision_variables": [
            {
                "symbol": item.get("symbol", ""),
                "domain": item.get("domain", ""),
            }
            for item in model.get("decision_variables") or []
            if isinstance(item, dict)
        ],
        "objective": {
            "sense": objective.get("sense", ""),
            "formula": objective.get("formula", ""),
            "source_chunk_ids": objective.get("source_chunk_ids", []),
        },
        "constraints": [
            {
                "name": item.get("name", ""),
                "formula": item.get("formula", ""),
                "source_chunk_ids": item.get("source_chunk_ids", []),
            }
            for item in constraints
            if isinstance(item, dict)
        ],
    }


def _structure_audit(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any]:
    before_sig = _model_structure_signature(before)
    after_sig = _model_structure_signature(after)
    changed_fields = [
        key
        for key in ("sets", "parameters", "decision_variables", "objective", "constraints")
        if before_sig.get(key) != after_sig.get(key)
    ]
    return {
        "formula_preserved": before_sig == after_sig,
        "changed_fields": changed_fields,
        "constraint_count_before": len(before_sig.get("constraints") or []),
        "constraint_count_after": len(after_sig.get("constraints") or []),
    }


def _compact_verifier(verifier: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(verifier, dict):
        return {}
    checks = verifier.get("checks") or []
    return {
        "status": verifier.get("status"),
        "score": verifier.get("score"),
        "warn_checks": [
            {
                "name": item.get("name"),
                "message": item.get("message"),
            }
            for item in checks
            if isinstance(item, dict) and item.get("status") == "warn"
        ][:8],
    }


def _compact_quality(quality: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(quality, dict):
        return {}
    return {
        "status": quality.get("status"),
        "overall_score": quality.get("overall_score"),
        "issues": (quality.get("issues") or [])[:8],
        "strengths": (quality.get("strengths") or [])[:8],
    }


def _polish_model_output(
    *,
    problem: str,
    model: dict[str, Any],
    evidence: list[dict[str, Any]],
    skill: ModelingSkill,
    harness_draft: dict[str, Any] | None,
    verifier: dict[str, Any] | None,
    quality: dict[str, Any] | None,
    max_tokens: int,
    reasoning_effort: str | None = None,
    timeout: float | None = None,
    client_max_retries: int | None = None,
) -> tuple[dict[str, Any] | None, str, str]:
    evidence_blocks = _build_evidence_blocks(evidence, max_items=4, compact=True)
    evidence_section = "\n\n".join(evidence_blocks) if evidence_blocks else "(no evidence)"
    compact_model = _compact_model_for_polish(model)
    prompt = f"""User modeling problem:
{problem}

Selected modeling skill:
{skill.name} - {skill.description}

Current verified model summary:
{json.dumps(compact_model, ensure_ascii=False, indent=2)}

Harness control summary:
{_build_harness_control_section(harness_draft, compact=True)}

Verifier summary:
{json.dumps(_compact_verifier(verifier), ensure_ascii=False, indent=2)}

Quality summary:
{json.dumps(_compact_quality(quality), ensure_ascii=False, indent=2)}

Retrieved modeling evidence:
{evidence_section}

Polish requirements:
- Preserve all formulas exactly as written.
- Preserve all symbols, domains, source_chunk_ids, and component decisions.
- Preserve the number and order of constraints.
- Improve academic modeling exposition for HHC mathematical modeling.
- Explain why the objective terms and constraint groups are appropriate.
- Keep descriptions concise but more paper-like.
- Return JSON only.
"""
    raw = ""
    try:
        raw = generate(
            messages=[
                {"role": "system", "content": MODEL_POLISH_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=CFG.generator.model_name,
            temperature=0.0,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            max_retries=client_max_retries,
        )
        patch = parse_json_object(raw)
        return _apply_polish_patch(model, patch), raw, ""
    except Exception as exc:
        return None, raw, str(exc)


def _generate_with_retry(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    attempts: int = 3,
    reasoning_effort: str | None = None,
    timeout: float | None = None,
    client_max_retries: int | None = None,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return generate_stream(
                messages=messages,
                model=CFG.generator.model_name,
                temperature=CFG.generator.temperature,
                max_tokens=max_tokens,
                on_token=None,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                max_retries=client_max_retries,
            )
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            time.sleep(1.5 * attempt)
    raise last_error or RuntimeError("LLM generation failed")


def _collect_source_ids(value: Any) -> set[str]:
    ids: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"source_chunk_ids", "chunk_ids"} and isinstance(child, list):
                ids.update(_normalize_source_id(str(item)) for item in child if item)
            elif key in {"source_chunk_id", "chunk_id"} and isinstance(child, str):
                ids.add(_normalize_source_id(child))
            else:
                ids.update(_collect_source_ids(child))
    elif isinstance(value, list):
        for child in value:
            ids.update(_collect_source_ids(child))
    return ids


def _normalize_generated_model(model: Any) -> Any:
    """Make LLM-produced model JSON safe for verifier/agent iteration.

    Providers sometimes return null for array fields in otherwise valid JSON.
    The downstream verifier treats these fields as lists, so normalize them
    immediately after parsing and after every agent revision.
    """
    if not isinstance(model, dict):
        return model

    list_fields = (
        "reference_models",
        "component_applicability",
        "sets",
        "parameters",
        "decision_variables",
        "constraints",
        "assumptions",
        "omitted_components",
    )
    for field in list_fields:
        value = model.get(field)
        if value is None:
            model[field] = []
        elif not isinstance(value, list):
            model[field] = [value]

    objective = model.get("objective")
    if objective is None:
        model["objective"] = {}
    elif not isinstance(objective, dict):
        model["objective"] = {"formula": str(objective), "description": "", "source_chunk_ids": []}

    validation = model.get("validation")
    if validation is None:
        model["validation"] = {"undefined_symbols": [], "potential_conflicts": [], "notes": []}
    elif isinstance(validation, dict):
        for field in ("undefined_symbols", "potential_conflicts", "notes"):
            value = validation.get(field)
            if value is None:
                validation[field] = []
            elif not isinstance(value, list):
                validation[field] = [value]

    for field in ("component_applicability", "reference_models", "constraints"):
        for item in model.get(field) or []:
            if not isinstance(item, dict):
                continue
            if item.get("source_chunk_ids") is None:
                item["source_chunk_ids"] = []
            elif not isinstance(item.get("source_chunk_ids"), list):
                item["source_chunk_ids"] = [item.get("source_chunk_ids")]
            if item.get("reusable_parts") is None:
                item["reusable_parts"] = []
            elif "reusable_parts" in item and not isinstance(item.get("reusable_parts"), list):
                item["reusable_parts"] = [item.get("reusable_parts")]

    if isinstance(model.get("objective"), dict):
        source_ids = model["objective"].get("source_chunk_ids")
        if source_ids is None:
            model["objective"]["source_chunk_ids"] = []
        elif not isinstance(source_ids, list):
            model["objective"]["source_chunk_ids"] = [source_ids]

    return model


def _normalize_source_id(value: str) -> str:
    return value.strip().strip("[]").strip()


def _validate_generated_model(
    model: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    valid_ids = {
        str(chunk.get("chunk_id") or chunk.get("figure_id"))
        for chunk in evidence
        if chunk.get("chunk_id") or chunk.get("figure_id")
    }
    cited_ids = _collect_source_ids(model)
    unknown = sorted(cid for cid in cited_ids if cid not in valid_ids)
    if unknown:
        warnings.append("unknown source_chunk_ids: " + ", ".join(unknown))

    for field in ("sets", "parameters", "decision_variables", "constraints"):
        if not isinstance(model.get(field), list):
            warnings.append(f"{field} is not a list")

    applicability = model.get("component_applicability")
    if not isinstance(applicability, list) or not applicability:
        warnings.append("component_applicability is missing or empty")
    else:
        for idx, item in enumerate(applicability, 1):
            if not isinstance(item, dict):
                warnings.append(f"component_applicability {idx} is not an object")
                continue
            if not item.get("component"):
                warnings.append(f"component_applicability {idx} has no component name")
            if not isinstance(item.get("applicable"), bool):
                warnings.append(f"component_applicability {idx} applicable is not boolean")
            if not item.get("reason"):
                warnings.append(f"component_applicability {idx} has no reason")

    objective = model.get("objective")
    if not isinstance(objective, dict):
        warnings.append("objective is not an object")
    elif not objective.get("formula"):
        warnings.append("objective.formula is empty")
    elif objective.get("formula") and not objective.get("source_chunk_ids"):
        warnings.append("objective has no source_chunk_ids")

    constraints = model.get("constraints")
    if isinstance(constraints, list):
        for idx, constraint in enumerate(constraints, 1):
            if not isinstance(constraint, dict):
                continue
            formula = str(constraint.get("formula", ""))
            description = str(constraint.get("description", ""))
            source_ids = constraint.get("source_chunk_ids") or []
            if formula and not source_ids:
                warnings.append(f"constraint {idx} has no source_chunk_ids")
            risk_text = f"{formula} {description}".lower()
            has_predecessor_equality = (
                ("predecessor" in risk_text or "successor" in risk_text)
                and ("exactly one" in risk_text or " = " in formula)
            )
            has_dummy_nodes = "dummy" in risk_text or "start/end" in risk_text
            if has_predecessor_equality and not has_dummy_nodes:
                warnings.append(
                    f"constraint {idx} may be infeasible: exact predecessor/successor "
                    "logic usually needs dummy start/end jobs or inequality degree constraints"
                )

    confidence = model.get("confidence")
    if not isinstance(confidence, (int, float)):
        warnings.append("confidence is missing or not numeric")

    validation = model.get("validation")
    if isinstance(validation, dict):
        undefined_symbols = validation.get("undefined_symbols")
        if isinstance(undefined_symbols, list) and undefined_symbols:
            warnings.append(
                "model has undefined symbols: "
                + ", ".join(str(item) for item in undefined_symbols)
            )
    return warnings


def _append_copy_risk_warnings(
    warnings: list[str],
    model: dict[str, Any],
    *,
    skill: ModelingSkill,
    evidence: list[dict[str, Any]],
) -> None:
    text = json.dumps(model, ensure_ascii=False).lower()
    equation_label_fields: list[str] = []
    for ref in model.get("reference_models") or []:
        if not isinstance(ref, dict):
            continue
        equation_label_fields.extend(str(part) for part in ref.get("reusable_parts") or [])
    objective = model.get("objective")
    if isinstance(objective, dict):
        equation_label_fields.extend(
            str(objective.get(key, "")) for key in ("formula", "description")
        )
    for constraint in model.get("constraints") or []:
        if not isinstance(constraint, dict):
            continue
        equation_label_fields.extend(
            str(constraint.get(key, "")) for key in ("name", "formula", "description")
        )
    if re.search(r"\beq\.?\s*\d+", "\n".join(equation_label_fields), flags=re.IGNORECASE):
        warnings.append(
            "copy-risk: generated model appears to use source-paper equation labels; "
            "constraint names should be semantic and adapted to the new problem"
        )

    constraints = model.get("constraints")
    if not isinstance(constraints, list):
        return

    model_constraint_ids = {
        str(item.get("chunk_id") or item.get("figure_id"))
        for item in evidence
        if item.get("chunk_type") == "constraints" and (item.get("chunk_id") or item.get("figure_id"))
    }
    cited_model_constraint_count = 0
    for constraint in constraints:
        if not isinstance(constraint, dict):
            continue
        source_ids = {
            str(source_id)
            for source_id in (constraint.get("source_chunk_ids") or [])
            if source_id
        }
        if source_ids & model_constraint_ids:
            cited_model_constraint_count += 1

    if len(constraints) >= 12 and cited_model_constraint_count >= max(8, int(len(constraints) * 0.75)):
        warnings.append(
            "copy-risk: many generated constraints cite retrieved constraint chunks; "
            "check that this is an adapted formulation rather than a full source-paper reproduction"
        )

    if skill.name == "home_health_care_routing_scheduling":
        source_specific_markers = (
            "i1",
            "i2",
            "ordinary patient",
            "vip patient",
            "x_i",
            "y_ik",
            "z_ijk",
            "c_ik",
            "e_ik",
        )
        marker_hits = sum(1 for marker in source_specific_markers if marker in text)
        if marker_hits >= 6:
            warnings.append(
                "copy-risk: the HHC output contains many source-paper-specific symbols or "
                "patient classes; verify that each one is required by the user's problem"
            )


def _needs_revision(warnings: list[str]) -> bool:
    blocking_prefixes = (
        "component_applicability",
        "copy-risk:",
        "model has undefined symbols:",
    )
    return any(warning.startswith(blocking_prefixes) for warning in warnings)


def _verifier_warnings(verification: dict[str, Any] | None) -> list[str]:
    if not verification or verification.get("status") == "pass":
        return []
    warnings: list[str] = []
    for check in verification.get("checks") or []:
        if not isinstance(check, dict):
            continue
        if check.get("status") in {"fail", "warn"}:
            warnings.append(
                "model_verifier {status}: {name}: {message}".format(
                    status=check.get("status", ""),
                    name=check.get("name", ""),
                    message=check.get("message", ""),
                )
            )
    return warnings


def _quality_warnings(quality: dict[str, Any] | None, *, threshold: float) -> list[str]:
    if not isinstance(quality, dict):
        return ["quality rubric missing"]
    warnings: list[str] = []
    overall = quality.get("overall_score")
    if isinstance(overall, (int, float)) and float(overall) < threshold:
        warnings.append(f"quality overall_score {float(overall):.3f} is below target {threshold:.3f}")
    for issue in quality.get("issues") or []:
        warnings.append(f"quality issue: {issue}")
    scores = quality.get("scores") or {}
    if isinstance(scores, dict):
        for name, value in scores.items():
            if name == "academic_depth_score":
                continue
            if isinstance(value, (int, float)) and float(value) < 0.45:
                warnings.append(f"quality subscore {name} is low: {float(value):.3f}")
    return _dedupe_strings(warnings)


def _agent_status(
    *,
    model: dict[str, Any] | None,
    warnings: list[str],
    verification: dict[str, Any] | None,
    quality: dict[str, Any] | None,
    target_quality: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not isinstance(model, dict):
        reasons.append("no structured model to accept")
    reasons.extend(warning for warning in warnings if _needs_revision([warning]))
    reasons.extend(_verifier_warnings(verification))
    reasons.extend(_quality_warnings(quality, threshold=target_quality))
    blocking = [
        reason
        for reason in _dedupe_strings(reasons)
        if reason
        and not reason.startswith("modeling_plan failed:")
        and not reason.startswith("modeling_blueprint failed:")
    ]
    return not blocking, blocking


def _agent_trace_entry(
    *,
    iteration: int,
    action: str,
    message: str,
    warnings: list[str],
    verification: dict[str, Any] | None,
    quality: dict[str, Any] | None,
) -> dict[str, Any]:
    role_map = {
        "planning_agent": "planning",
        "formula_agent": "formula",
        "verification_agent": "verification",
        "polisher_agent": "polisher",
        "evaluate": "verification",
        "revise_and_evaluate": "revision",
        "revision_failed": "revision",
    }
    return {
        "iteration": iteration,
        "action": action,
        "message": message,
        "warnings": _dedupe_strings(warnings)[:16],
        "verifier_status": (verification or {}).get("status", ""),
        "verifier_score": (verification or {}).get("score"),
        "quality_status": (quality or {}).get("status", ""),
        "quality_score": (quality or {}).get("overall_score"),
        "quality_issues": list((quality or {}).get("issues") or [])[:12],
        "agent_role": role_map.get(action, action),
    }


def generate_math_model(
    problem: str,
    *,
    top_k: int | None = None,
    paper_id: str | None = None,
    service: QueryService | None = None,
    max_tokens: int | None = None,
    generation_attempts: int = 3,
    reasoning_effort: str | None = None,
    timeout: float | None = None,
    client_max_retries: int | None = None,
    skip_llm_plan: bool = False,
    skip_revision: bool = False,
    agent_mode: bool = False,
    agent_max_rounds: int = 2,
    agent_quality_threshold: float = 0.86,
    use_blueprint: bool = False,
    generate_platemo_code: bool = False,
    platemo_root: str | None = None,
    platemo_class_name: str | None = None,
    write_platemo_file: bool = False,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    """Retrieve modeling evidence and generate a structured model draft."""
    if not problem.strip():
        raise ValueError("problem must not be empty")

    owns_service = service is None
    service = service or QueryService.from_disk()
    resolved_top_k = top_k or max(CFG.retriever.top_k, 8)
    skill = select_modeling_skill(problem)

    try:
        _emit_progress(progress_callback, "retrieval", "正在检索论文模型证据", 0.12)
        _check_cancelled(cancel_check)
        retrieval_error = ""
        try:
            retrieval_query, retrieval = _retrieve_with_skill(
                service,
                problem=problem,
                skill=skill,
                top_k=resolved_top_k,
                paper_id=paper_id,
            )
        except Exception as exc:
            retrieval_error = str(exc)
            retrieval_query, retrieval = _retrieve_metadata_fallback_with_skill(
                service,
                problem=problem,
                skill=skill,
                top_k=resolved_top_k,
                paper_id=paper_id,
            )
        evidence = flatten_dual_results(retrieval)
        _emit_progress(
            progress_callback,
            "retrieval_done",
            f"检索完成：获得 {len(evidence)} 条建模证据",
            0.22,
        )
        _check_cancelled(cancel_check)
        retrieval_warnings = (
            [f"vector retrieval failed; used metadata fallback: {retrieval_error}"]
            if retrieval_error
            else []
        )
        if max_tokens is not None:
            model_max_tokens = max_tokens
        elif skill.name == "home_health_care_routing_scheduling":
            model_max_tokens = CFG.generator.max_new_tokens
        else:
            model_max_tokens = CFG.generator.max_new_tokens
        plan: dict[str, Any] | None = None
        plan_output = ""
        plan_error = ""
        if agent_mode and not skip_llm_plan:
            _emit_progress(
                progress_callback,
                "planning_agent",
                "Planning Agent 正在确定建模范围和组件",
                0.30,
            )
            _check_cancelled(cancel_check)
            plan, plan_output, plan_error = _build_lightweight_agent_plan(
                problem,
                evidence,
                skill=skill,
                max_tokens=min(model_max_tokens, 500),
                reasoning_effort=reasoning_effort,
                timeout=min(timeout, 120.0) if timeout is not None else 120.0,
                client_max_retries=client_max_retries,
            )
        elif not skip_llm_plan:
            _emit_progress(progress_callback, "planner", "正在生成建模规划", 0.30)
            _check_cancelled(cancel_check)
            plan, plan_output, plan_error = _build_modeling_plan(
                problem,
                evidence,
                skill=skill,
                max_tokens=min(model_max_tokens, 650),
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                client_max_retries=client_max_retries,
            )
        else:
            plan_error = "skipped"
            _emit_progress(progress_callback, "planner_skipped", "已跳过大模型规划阶段", 0.32)
        _check_cancelled(cancel_check)
        _emit_progress(progress_callback, "harness", "正在选择建模组件并生成 Harness 边界", 0.40)
        harness_draft = build_harness_draft(
            problem,
            evidence,
            skill=skill,
            modeling_plan=plan,
        )
        problem_spec = _build_problem_spec(
            problem=problem,
            skill=skill,
            modeling_plan=plan,
            harness_draft=harness_draft,
            evidence=evidence,
            generation_depth="paper_level" if use_blueprint else "standard",
        )
        _check_cancelled(cancel_check)
        blueprint: dict[str, Any] | None = None
        blueprint_output = ""
        blueprint_error = ""
        if use_blueprint and not agent_mode:
            _emit_progress(progress_callback, "blueprint", "正在生成论文级建模蓝图", 0.48)
            _check_cancelled(cancel_check)
            blueprint, blueprint_output, blueprint_error = _build_modeling_blueprint(
                problem,
                evidence,
                skill=skill,
                modeling_plan=plan,
                problem_spec=problem_spec,
                max_tokens=min(model_max_tokens, 1200),
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                client_max_retries=client_max_retries,
            )
        else:
            blueprint_error = "" if agent_mode and use_blueprint else blueprint_error
            _emit_progress(progress_callback, "blueprint_skipped", "未启用建模蓝图阶段", 0.50)
        _check_cancelled(cancel_check)
        parse_error = ""
        repair_output = ""
        revision_output = ""
        revision_note = ""
        agent_trace: list[dict[str, Any]] = []
        agent_terminate_reason = "not_enabled"
        model: dict[str, Any] | None
        raw_output = ""
        if agent_mode:
            _emit_progress(
                progress_callback,
                "formula_agent",
                "Formula Agent 正在使用 Harness 生成可校验公式骨架",
                0.58,
            )
            model = _normalize_generated_model(render_harness_model(harness_draft))
            raw_output = json.dumps(model, ensure_ascii=False, indent=2)
            revision_note = (
                "lightweight Modeling Agent used Planning Agent + Harness Formula Agent "
                "before verifier-based review"
            )
        else:
            messages = [
                {"role": "system", "content": MODEL_GENERATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_user_prompt(
                        problem,
                        evidence,
                        skill=skill,
                        problem_spec=problem_spec,
                        modeling_plan=plan,
                        modeling_blueprint=blueprint,
                        harness_draft=harness_draft,
                        compact=skill.name == "home_health_care_routing_scheduling",
                    ),
                },
            ]
            _emit_progress(
                progress_callback,
                "llm_generation",
                "正在调用大模型生成完整数学模型",
                0.58,
            )
            _check_cancelled(cancel_check)
            raw_output = _generate_with_retry(
                messages=messages,
                max_tokens=model_max_tokens,
                attempts=generation_attempts,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                client_max_retries=client_max_retries,
            )
            _emit_progress(progress_callback, "parse", "正在解析结构化模型 JSON", 0.78)
            _check_cancelled(cancel_check)
            try:
                model = _normalize_generated_model(parse_json_object(raw_output))
            except Exception as exc:
                parse_error = str(exc)
                _emit_progress(progress_callback, "repair", "JSON 解析失败，正在尝试修复输出", 0.82)
                model, repair_output, repair_error = _repair_json_output(
                    raw_output,
                    max_tokens=model_max_tokens,
                )
                model = _normalize_generated_model(model)
                if model is None:
                    parse_error = f"{parse_error}; repair failed: {repair_error}"
                else:
                    parse_error = ""

        warnings = []
        if plan is None:
            warnings.append(f"modeling_plan failed: {plan_error}")
        if use_blueprint and blueprint is None and not agent_mode:
            warnings.append(f"modeling_blueprint failed: {blueprint_error}")
        if model is not None:
            _emit_progress(progress_callback, "verifier", "正在校验模型组件、符号和约束结构", 0.86)
            _check_cancelled(cancel_check)
            warnings.extend(_validate_generated_model(model, evidence))
            _append_copy_risk_warnings(warnings, model, skill=skill, evidence=evidence)
            initial_verification = verify_model(
                model,
                harness_draft=harness_draft,
                problem=problem,
            )
            verifier_revision_warnings = _verifier_warnings(initial_verification)
            if not agent_mode and not skip_revision and (_needs_revision(warnings) or verifier_revision_warnings):
                _emit_progress(progress_callback, "revision", "Verifier 发现问题，正在请求大模型修正一次", 0.90)
                _check_cancelled(cancel_check)
                revised_model, revision_output, revision_error = _revise_model_output(
                    problem=problem,
                    model=model,
                    warnings=warnings + verifier_revision_warnings,
                    evidence=evidence,
                    skill=skill,
                    harness_draft=harness_draft,
                    max_tokens=min(model_max_tokens, 1600),
                    reasoning_effort=reasoning_effort,
                    timeout=timeout,
                    client_max_retries=client_max_retries,
                )
                if revised_model is not None:
                    model = revised_model
                    revision_note = "model was revised once to enforce harness/verifier requirements"
                    warnings = retrieval_warnings + _validate_generated_model(model, evidence)
                    _append_copy_risk_warnings(warnings, model, skill=skill, evidence=evidence)
                    if warnings:
                        warnings.append("model was revised once but still has warnings")
                else:
                    warnings.append(f"model revision failed: {revision_error}")
            warnings = retrieval_warnings + warnings
        else:
            warnings = retrieval_warnings + ["failed to parse model JSON"]
        _emit_progress(progress_callback, "final_verifier", "正在生成最终校验报告", 0.94)
        _check_cancelled(cancel_check)
        model_verification = verify_model(
            model,
            harness_draft=harness_draft,
            problem=problem,
        )
        model_quality = evaluate_model_quality(
            model,
            problem_spec=problem_spec,
            harness_draft=harness_draft,
            verifier=model_verification,
        )
        polish_output = ""
        polish_error = ""
        polish_audit: dict[str, Any] = {}
        if agent_mode and use_blueprint and model is not None:
            _emit_progress(
                progress_callback,
                "polisher_agent",
                "Polisher Agent 正在增强论文级建模表达",
                0.965,
            )
            _check_cancelled(cancel_check)
            model_before_polish = copy.deepcopy(model)
            polished_model, polish_output, polish_error = _polish_model_output(
                problem=problem,
                model=model,
                evidence=evidence,
                skill=skill,
                harness_draft=harness_draft,
                verifier=model_verification,
                quality=model_quality,
                max_tokens=min(model_max_tokens, 900),
                reasoning_effort=reasoning_effort,
                timeout=min(timeout, 180.0) if timeout is not None else 180.0,
                client_max_retries=client_max_retries,
            )
            if polished_model is not None:
                polish_audit = _structure_audit(model_before_polish, polished_model)
                if not polish_audit.get("formula_preserved"):
                    polish_error = (
                        "polisher changed protected mathematical structure: "
                        + ", ".join(polish_audit.get("changed_fields") or [])
                    )
                    polished_model = None
            if polished_model is not None:
                model = polished_model
                validation = model.get("validation")
                if isinstance(validation, dict):
                    notes = list(validation.get("notes") or [])
                    notes.append(
                        "Polisher structure audit: formulas, symbols, domains, constraints, and source ids preserved."
                    )
                    validation["notes"] = _dedupe_strings([str(item) for item in notes])
                revision_output = (
                    f"{revision_output}\n\n--- polisher agent ---\n{polish_output}"
                    if revision_output
                    else polish_output
                )
                revision_note = (
                    revision_note
                    + "; Polisher Agent enhanced academic exposition without changing formulas"
                )
                warnings = retrieval_warnings + _validate_generated_model(model, evidence)
                _append_copy_risk_warnings(warnings, model, skill=skill, evidence=evidence)
                model_verification = verify_model(
                    model,
                    harness_draft=harness_draft,
                    problem=problem,
                )
                model_quality = evaluate_model_quality(
                    model,
                    problem_spec=problem_spec,
                    harness_draft=harness_draft,
                    verifier=model_verification,
                )
            else:
                warnings.append(f"polisher agent failed: {polish_error}")
        accepted, agent_reasons = _agent_status(
            model=model,
            warnings=warnings,
            verification=model_verification,
            quality=model_quality,
            target_quality=agent_quality_threshold,
        )
        if agent_mode:
            agent_trace.append(
                {
                    "iteration": 0,
                    "action": "planning_agent",
                    "message": (
                        "Planning Agent produced a compact component plan."
                        if plan is not None
                        else f"Planning Agent failed or was skipped: {plan_error or 'unknown error'}"
                    ),
                    "warnings": [] if plan is not None else [plan_error or "planning agent unavailable"],
                    "verifier_status": "",
                    "verifier_score": None,
                    "quality_status": "",
                    "quality_score": None,
                    "quality_issues": [],
                    "agent_role": "planning",
                    "modeling_plan": _compact_modeling_plan(plan),
                }
            )
            agent_trace.append(
                {
                    "iteration": 0,
                    "action": "formula_agent",
                    "message": "Formula Agent rendered the Harness-selected components into a structured model.",
                    "warnings": [],
                    "verifier_status": "",
                    "verifier_score": None,
                    "quality_status": "",
                    "quality_score": None,
                    "quality_issues": [],
                    "agent_role": "formula",
                    "selected_components": (problem_spec or {}).get("selected_components") or [],
                    "constraint_count": len((model or {}).get("constraints") or []),
                }
            )
            agent_trace.append(
                _agent_trace_entry(
                    iteration=0,
                    action="verification_agent",
                    message="Verification Agent evaluated the model with verifier and quality rubric.",
                    warnings=agent_reasons,
                    verification=model_verification,
                    quality=model_quality,
                )
            )
            if use_blueprint:
                agent_trace.append(
                    {
                        "iteration": 0,
                        "action": "polisher_agent",
                        "message": (
                            "Polisher Agent enhanced academic exposition while preserving formulas."
                            if polish_output
                            else f"Polisher Agent was unavailable: {polish_error or 'no output'}"
                        ),
                        "warnings": [] if polish_output else [polish_error or "polisher unavailable"],
                        "verifier_status": (model_verification or {}).get("status", ""),
                        "verifier_score": (model_verification or {}).get("score"),
                        "quality_status": (model_quality or {}).get("status", ""),
                        "quality_score": (model_quality or {}).get("overall_score"),
                        "quality_issues": list((model_quality or {}).get("issues") or [])[:12],
                        "agent_role": "polisher",
                        "structure_audit": polish_audit,
                    }
                )
            if accepted:
                agent_terminate_reason = "accepted_initial"
            else:
                rounds = max(0, int(agent_max_rounds))
                for round_idx in range(1, rounds + 1):
                    _emit_progress(
                        progress_callback,
                        f"agent_review_{round_idx}",
                        f"Agent 第 {round_idx} 轮评审：准备反馈并修正模型",
                        min(0.96, 0.94 + round_idx * 0.01),
                    )
                    _check_cancelled(cancel_check)
                    if model is None:
                        agent_terminate_reason = "no_model"
                        break
                    revised_model, revised_output, revision_error = _revise_model_output(
                        problem=problem,
                        model=model,
                        warnings=agent_reasons,
                        evidence=evidence,
                        skill=skill,
                        harness_draft=harness_draft,
                        max_tokens=min(model_max_tokens, 1800),
                        reasoning_effort=reasoning_effort,
                        timeout=timeout,
                        client_max_retries=client_max_retries,
                    )
                    if revised_output:
                        revision_output = (
                            f"{revision_output}\n\n--- agent round {round_idx} ---\n{revised_output}"
                            if revision_output
                            else revised_output
                        )
                    if revised_model is None:
                        agent_trace.append(
                            _agent_trace_entry(
                                iteration=round_idx,
                                action="revision_failed",
                                message=f"Agent revision failed: {revision_error}",
                                warnings=agent_reasons + [revision_error],
                                verification=model_verification,
                                quality=model_quality,
                            )
                        )
                        agent_terminate_reason = "revision_failed"
                        break
                    model = revised_model
                    revision_note = (
                        f"model was revised by Modeling Agent for {round_idx} round(s)"
                    )
                    warnings = _validate_generated_model(model, evidence)
                    _append_copy_risk_warnings(warnings, model, skill=skill, evidence=evidence)
                    model_verification = verify_model(
                        model,
                        harness_draft=harness_draft,
                        problem=problem,
                    )
                    model_quality = evaluate_model_quality(
                        model,
                        problem_spec=problem_spec,
                        harness_draft=harness_draft,
                        verifier=model_verification,
                    )
                    accepted, agent_reasons = _agent_status(
                        model=model,
                        warnings=warnings,
                        verification=model_verification,
                        quality=model_quality,
                        target_quality=agent_quality_threshold,
                    )
                    agent_trace.append(
                        _agent_trace_entry(
                            iteration=round_idx,
                            action="revise_and_evaluate",
                            message=(
                                "Agent accepted the revised model."
                                if accepted
                                else "Agent still found issues after revision."
                            ),
                            warnings=agent_reasons,
                            verification=model_verification,
                            quality=model_quality,
                        )
                    )
                    if accepted:
                        agent_terminate_reason = f"accepted_round_{round_idx}"
                        break
                else:
                    agent_terminate_reason = "max_rounds_reached"
        elif accepted:
            agent_terminate_reason = "accepted_without_agent"
        else:
            agent_terminate_reason = "needs_review_without_agent"
        code_generation: dict[str, Any] | None = None
        if generate_platemo_code:
            _emit_progress(progress_callback, "code_generation", "正在生成 PlatEMO MATLAB 问题类", 0.985)
            _check_cancelled(cancel_check)
            try:
                code_generation = build_platemo_code(
                    problem=problem,
                    model=model,
                    harness_draft=harness_draft,
                    problem_spec=problem_spec,
                    platemo_root=platemo_root,
                    class_name=platemo_class_name,
                    write_file=write_platemo_file,
                )
                warnings.extend(code_generation.get("warnings") or [])
            except Exception as exc:
                warnings.append(f"PlatEMO code generation failed: {exc}")
        _emit_progress(progress_callback, "done", "数学模型生成完成", 1.0)

        return {
            "problem": problem,
            "retrieval_query": retrieval_query,
            "skill": skill.name,
            "skill_description": skill.description,
            "generation_mode": "lightweight_agent" if agent_mode else "llm",
            "modeling_plan": plan,
            "plan_output": plan_output,
            "plan_error": plan_error,
            "problem_spec": problem_spec,
            "harness_draft": harness_draft,
            "modeling_blueprint": blueprint,
            "blueprint_output": blueprint_output,
            "blueprint_error": blueprint_error,
            "paper_id": paper_id,
            "top_k": resolved_top_k,
            "text_results": retrieval["text_results"],
            "figure_results": retrieval["figure_results"],
            "model": model,
            "raw_output": raw_output,
            "repair_output": repair_output,
            "revision_output": revision_output,
            "revision_note": revision_note,
            "parse_error": parse_error,
            "warnings": warnings,
            "model_verification": model_verification,
            "model_quality": model_quality,
            "agent_mode": agent_mode,
            "agent_trace": agent_trace,
            "agent_terminate_reason": agent_terminate_reason,
            "code_generation": code_generation,
        }
    finally:
        if owns_service:
            service.close()


def generate_harness_draft(
    problem: str,
    *,
    top_k: int | None = None,
    paper_id: str | None = None,
    service: QueryService | None = None,
    render_formulas: bool = False,
) -> dict[str, Any]:
    """Retrieve modeling evidence and build a fast deterministic harness draft."""
    if not problem.strip():
        raise ValueError("problem must not be empty")

    owns_service = service is None
    service = service or QueryService.from_disk()
    resolved_top_k = top_k or max(CFG.retriever.top_k, 8)
    skill = select_modeling_skill(problem)

    try:
        retrieval_error = ""
        try:
            retrieval_query, retrieval = _retrieve_with_skill(
                service,
                problem=problem,
                skill=skill,
                top_k=resolved_top_k,
                paper_id=paper_id,
            )
        except Exception as exc:
            retrieval_error = str(exc)
            retrieval_query, retrieval = _retrieve_metadata_fallback_with_skill(
                service,
                problem=problem,
                skill=skill,
                top_k=resolved_top_k,
                paper_id=paper_id,
            )
        evidence = flatten_dual_results(retrieval)
        harness_draft = build_harness_draft(
            problem,
            evidence,
            skill=skill,
            modeling_plan=None,
        )
        rendered_model = render_harness_model(harness_draft) if render_formulas else None
        model_verification = (
            verify_model(
                rendered_model,
                harness_draft=harness_draft,
                problem=problem,
            )
            if rendered_model is not None
            else None
        )
        model_quality = (
            evaluate_model_quality(
                rendered_model,
                problem_spec=None,
                harness_draft=harness_draft,
                verifier=model_verification,
            )
            if rendered_model is not None
            else None
        )
        return {
            "problem": problem,
            "retrieval_query": retrieval_query,
            "skill": skill.name,
            "skill_description": skill.description,
            "generation_mode": "harness_formula" if render_formulas else "harness_draft",
            "paper_id": paper_id,
            "top_k": resolved_top_k,
            "text_results": retrieval["text_results"],
            "figure_results": retrieval["figure_results"],
            "harness_draft": harness_draft,
            "model": rendered_model,
            "model_verification": model_verification,
            "model_quality": model_quality,
            "warnings": (
                ([f"vector retrieval failed; used metadata fallback: {retrieval_error}"] if retrieval_error else [])
                + harness_draft.get("validation", {}).get("warnings", [])
            ),
        }
    finally:
        if owns_service:
            service.close()
