"""Deterministic PlatEMO code generation from structured model drafts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_PLATEMO_ROOT = Path("PlatEMO")
DEFAULT_TARGET_RELATIVE = Path(
    "PlatEMO",
    "Problems",
    "Multi-objective optimization",
    "hhc",
    "generated",
)

SUPPORTED_COMPONENTS = {
    "assignment",
    "routing_flow",
    "time_propagation",
    "time_window",
    "waiting_time",
    "capacity",
    "skill_matching",
    "outsourcing",
    "priority_class",
    "multi_objective",
    "open_route",
    "balance",
}


def generate_platemo_code(
    *,
    problem: str = "",
    model: dict[str, Any] | None = None,
    harness_draft: dict[str, Any] | None = None,
    problem_spec: dict[str, Any] | None = None,
    platemo_root: str | Path | None = None,
    class_name: str | None = None,
    write_file: bool = False,
) -> dict[str, Any]:
    """Generate a PlatEMO PROBLEM class from the modeling IR.

    This first compiler pass intentionally favors executable, reviewable
    scaffolding over a direct symbolic-to-code translation of every formula.
    It maps selected HHC components to a stable mixed encoding and reports
    unsupported components as warnings.
    """
    model = model or {}
    harness_draft = harness_draft or {}
    problem_spec = problem_spec or {}
    selected = _selected_components(model, harness_draft, problem_spec)
    omitted = _omitted_components(model, harness_draft)
    unsupported = sorted(component for component in selected if component not in SUPPORTED_COMPONENTS)
    warnings = _build_warnings(model, harness_draft, selected, unsupported)

    resolved_class_name = _safe_class_name(
        class_name or _default_class_name(problem, model, harness_draft, selected)
    )
    objective_count = _objective_count(model, harness_draft, selected)
    matlab_code = _render_matlab_class(
        class_name=resolved_class_name,
        selected_components=selected,
        omitted_components=omitted,
        objective_count=objective_count,
    )

    target_dir: Path | None = None
    target_path: Path | None = None
    resolved_root = Path(platemo_root) if platemo_root else DEFAULT_PLATEMO_ROOT
    if write_file:
        if not resolved_root.exists():
            warnings.append(f"PlatEMO root does not exist: {resolved_root}")
        else:
            target_dir = resolved_root / DEFAULT_TARGET_RELATIVE
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / f"{resolved_class_name}.m"
            target_path.write_text(matlab_code, encoding="utf-8", newline="\n")
    else:
        target_dir = resolved_root / DEFAULT_TARGET_RELATIVE
        target_path = target_dir / f"{resolved_class_name}.m"

    return {
        "platform": "PlatEMO",
        "language": "MATLAB",
        "class_name": resolved_class_name,
        "target_path": str(target_path) if target_path is not None else "",
        "target_dir": str(target_dir) if target_dir is not None else "",
        "written": bool(write_file and target_path is not None and target_path.exists()),
        "matlab_code": matlab_code,
        "warnings": _dedupe(warnings),
        "component_map": {
            "selected": selected,
            "implemented": [component for component in selected if component in SUPPORTED_COMPONENTS],
            "omitted": omitted,
            "unsupported": unsupported,
        },
    }


def _selected_components(
    model: dict[str, Any],
    harness_draft: dict[str, Any],
    problem_spec: dict[str, Any],
) -> list[str]:
    components: list[str] = []
    for item in harness_draft.get("component_selector") or []:
        if isinstance(item, dict) and item.get("status") == "selected":
            components.append(str(item.get("component", "")))
    if not components:
        for item in model.get("component_applicability") or []:
            if isinstance(item, dict) and item.get("applicable"):
                components.append(str(item.get("component", "")))
    for item in problem_spec.get("selected_components") or []:
        components.append(str(item))
    components = _normalise_components(components)
    if "assignment" not in components:
        components.insert(0, "assignment")
    if "routing_flow" not in components:
        components.insert(1, "routing_flow")
    return _dedupe(components)


def _omitted_components(model: dict[str, Any], harness_draft: dict[str, Any]) -> list[str]:
    omitted: list[str] = []
    for item in harness_draft.get("component_selector") or []:
        if isinstance(item, dict) and item.get("status") == "omitted":
            omitted.append(str(item.get("component", "")))
    for item in model.get("omitted_components") or []:
        if isinstance(item, dict):
            omitted.append(str(item.get("component", "")))
        else:
            omitted.append(str(item))
    return _normalise_components(omitted)


def _normalise_components(components: list[str]) -> list[str]:
    aliases = {
        "routing": "routing_flow",
        "route_flow": "routing_flow",
        "time windows": "time_window",
        "time-window": "time_window",
        "workload": "capacity",
        "workload_balance": "balance",
        "skill": "skill_matching",
        "skills": "skill_matching",
        "priority": "priority_class",
        "vip": "priority_class",
        "multi objective": "multi_objective",
        "multi-objective": "multi_objective",
        "open routes": "open_route",
    }
    out: list[str] = []
    for component in components:
        clean = re.sub(r"[^a-zA-Z0-9]+", "_", str(component).strip().lower()).strip("_")
        if not clean:
            continue
        out.append(aliases.get(clean, clean))
    return _dedupe(out)


def _build_warnings(
    model: dict[str, Any],
    harness_draft: dict[str, Any],
    selected: list[str],
    unsupported: list[str],
) -> list[str]:
    warnings: list[str] = []
    problem_type = (
        (harness_draft.get("model_spec") or {}).get("problem_type")
        or harness_draft.get("problem_type")
        or ((model.get("problem_analysis") or {}).get("problem_type"))
        or ""
    )
    if problem_type and problem_type != "home_health_care_routing_scheduling":
        warnings.append(
            "This compiler currently emits an HHC-style PlatEMO problem skeleton; verify it before using non-HHC models."
        )
    if unsupported:
        warnings.append(
            "Selected components not yet translated into executable MATLAB logic: "
            + ", ".join(unsupported)
        )
    if "assignment" not in selected or "routing_flow" not in selected:
        warnings.append(
            "Assignment and routing encoding were added because PlatEMO needs executable decision variables."
        )
    if not model:
        warnings.append("No final model JSON was supplied; generated code is based on Harness signals only.")
    warnings.append(
        "Generated data are deterministic placeholders; replace generate_test_data or load a dataset before experiments."
    )
    return warnings


def _objective_count(
    model: dict[str, Any],
    harness_draft: dict[str, Any],
    selected: list[str],
) -> int:
    objective = model.get("objective") or {}
    sense = str(objective.get("sense") or "").lower()
    objective_terms = ((harness_draft.get("model_spec") or {}).get("objective_terms") or [])
    if "multi_objective" in selected or sense == "multi-objective":
        return 3 if len(objective_terms) >= 3 else 2
    return 2


def _default_class_name(
    problem: str,
    model: dict[str, Any],
    harness_draft: dict[str, Any],
    selected: list[str],
) -> str:
    payload = {
        "problem": problem,
        "objective": (model.get("objective") or {}).get("formula", ""),
        "components": selected,
        "problem_type": harness_draft.get("problem_type", ""),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:8]
    return f"HHCGenerated_{digest}"


def _safe_class_name(raw: str) -> str:
    name = re.sub(r"\W+", "_", str(raw or "").strip())
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "HHCGenerated"
    if not re.match(r"^[A-Za-z]", name):
        name = f"HHC{name}"
    if len(name) > 63:
        name = name[:63].rstrip("_")
    return name


def _render_matlab_class(
    *,
    class_name: str,
    selected_components: list[str],
    omitted_components: list[str],
    objective_count: int,
) -> str:
    has = {component: component in selected_components for component in SUPPORTED_COMPONENTS}
    selected_literal = _matlab_cellstr(selected_components)
    omitted_literal = _matlab_cellstr(omitted_components)
    header_comments = _component_comments("Selected components", selected_components)
    header_comments += _component_comments("Omitted components", omitted_components)
    if not header_comments:
        header_comments = ["% Selected components: assignment, routing_flow"]
    objective_count = max(2, min(3, objective_count))

    return "\n".join(
        [
            f"classdef {class_name} < PROBLEM",
            "% <auto> <multi> <constrained> <mixed>",
            "% Generated by MultiRAG-Doc modeling PlatEMO compiler.",
            *header_comments,
            "",
            "    properties",
            "        K = 4;",
            "        P = 12;",
            "        depot = 1;",
            "        locations;",
            "        travel_time;",
            "        service_time;",
            "        ready_time;",
            "        due_time;",
            "        skill_level;",
            "        required_skill;",
            "        priority;",
            "        workload_limit;",
            "        c_travel = 1.0;",
            "        c_wait = 1.0;",
            "        c_late = 2.0;",
            "        c_skill = 10.0;",
            "        c_balance = 0.5;",
            "        c_outsource = 100.0;",
            f"        selected_components = {selected_literal};",
            f"        omitted_components = {omitted_literal};",
            f"        has_assignment = {_matlab_bool(has['assignment'])};",
            f"        has_routing_flow = {_matlab_bool(has['routing_flow'])};",
            f"        has_time_propagation = {_matlab_bool(has['time_propagation'])};",
            f"        has_time_window = {_matlab_bool(has['time_window'])};",
            f"        has_waiting_time = {_matlab_bool(has['waiting_time'])};",
            f"        has_capacity = {_matlab_bool(has['capacity'])};",
            f"        has_skill_matching = {_matlab_bool(has['skill_matching'])};",
            f"        has_outsourcing = {_matlab_bool(has['outsourcing'])};",
            f"        has_priority_class = {_matlab_bool(has['priority_class'])};",
            f"        has_open_route = {_matlab_bool(has['open_route'])};",
            f"        has_balance = {_matlab_bool(has['balance'])};",
            "    end",
            "",
            "    methods",
            "        function Setting(obj)",
            "            obj.Setting@PROBLEM();",
            "            [obj.K,obj.P] = obj.ParameterSet(obj.K,obj.P);",
            "            obj.K = max(1,round(obj.K));",
            "            obj.P = max(1,round(obj.P));",
            "            obj = obj.generate_test_data();",
            f"            obj.M = {objective_count};",
            "            obj.D = 2*obj.P;",
            "            obj.lower = [ones(1,obj.P), zeros(1,obj.P)];",
            "            obj.upper = [obj.K*ones(1,obj.P), ones(1,obj.P)];",
            "            obj.encoding = [3*ones(1,obj.P), ones(1,obj.P)];",
            "            if obj.has_outsourcing",
            "                obj.D = obj.D + obj.P;",
            "                obj.lower = [obj.lower, zeros(1,obj.P)];",
            "                obj.upper = [obj.upper, ones(1,obj.P)];",
            "                obj.encoding = [obj.encoding, 4*ones(1,obj.P)];",
            "            end",
            "        end",
            "",
            "        function PopObj = CalObj(obj,PopDec)",
            "            N = size(PopDec,1);",
            "            PopObj = zeros(N,obj.M);",
            "            for s = 1:N",
            "                [assign,routes,outsourced] = obj.decode(PopDec(s,:));",
            "                [travel,waitPenalty,latePenalty,workload,skillPenalty,priorityPenalty] = obj.evaluate_routes(assign,routes,outsourced);",
            "                outsourcePenalty = obj.c_outsource * sum(outsourced);",
            "                balancePenalty = obj.c_balance * obj.workload_imbalance(workload);",
            "                if obj.M >= 3",
            "                    PopObj(s,1) = obj.c_travel*travel + outsourcePenalty;",
            "                    PopObj(s,2) = obj.c_wait*waitPenalty + obj.c_late*latePenalty + priorityPenalty;",
            "                    PopObj(s,3) = skillPenalty + balancePenalty;",
            "                else",
            "                    PopObj(s,1) = obj.c_travel*travel + obj.c_wait*waitPenalty + obj.c_late*latePenalty + outsourcePenalty;",
            "                    PopObj(s,2) = skillPenalty + priorityPenalty + balancePenalty;",
            "                end",
            "            end",
            "        end",
            "",
            "        function PopCon = CalCon(obj,PopDec)",
            "            nCon = double(obj.has_time_window) + double(obj.has_skill_matching) + double(obj.has_capacity);",
            "            nCon = max(1,nCon);",
            "            PopCon = zeros(size(PopDec,1),nCon);",
            "            for s = 1:size(PopDec,1)",
            "                [assign,routes,outsourced] = obj.decode(PopDec(s,:));",
            "                [~,~,latePenalty,workload,skillPenalty,~] = obj.evaluate_routes(assign,routes,outsourced);",
            "                col = 1;",
            "                if obj.has_time_window",
            "                    PopCon(s,col) = latePenalty;",
            "                    col = col + 1;",
            "                end",
            "                if obj.has_skill_matching",
            "                    PopCon(s,col) = skillPenalty / max(1,obj.c_skill);",
            "                    col = col + 1;",
            "                end",
            "                if obj.has_capacity",
            "                    PopCon(s,col) = sum(max(0,workload - obj.workload_limit));",
            "                    col = col + 1;",
            "                end",
            "            end",
            "        end",
            "",
            "        function [assign,routes,outsourced] = decode(obj,chromosome)",
            "            chromosome = real(double(reshape(chromosome,1,[])));",
            "            chromosome(~isfinite(chromosome)) = 0;",
            "            assign = round(chromosome(1:obj.P));",
            "            assign = max(1,min(obj.K,assign));",
            "            orderKey = chromosome(obj.P+1:2*obj.P);",
            "            outsourced = false(1,obj.P);",
            "            if obj.has_outsourcing && numel(chromosome) >= 3*obj.P",
            "                outsourced = round(chromosome(2*obj.P+1:3*obj.P)) > 0;",
            "            end",
            "            routes = cell(1,obj.K);",
            "            for k = 1:obj.K",
            "                patients = find(assign == k & ~outsourced);",
            "                if isempty(patients)",
            "                    routes{k} = [];",
            "                else",
            "                    [~,order] = sortrows([orderKey(patients)', patients'], [1 2]);",
            "                    routes{k} = patients(order) + 1;",
            "                end",
            "            end",
            "        end",
            "",
            "        function [travel,waitPenalty,latePenalty,workload,skillPenalty,priorityPenalty] = evaluate_routes(obj,assign,routes,outsourced)",
            "            %#ok<INUSD>",
            "            travel = 0; waitPenalty = 0; latePenalty = 0;",
            "            workload = zeros(1,obj.K);",
            "            skillPenalty = 0; priorityPenalty = 0;",
            "            for k = 1:obj.K",
            "                currentTime = 0;",
            "                previous = obj.depot;",
            "                route = routes{k};",
            "                for idx = 1:length(route)",
            "                    node = route(idx);",
            "                    p = node - 1;",
            "                    leg = obj.travel_time(previous,node);",
            "                    travel = travel + leg;",
            "                    currentTime = currentTime + leg;",
            "                    wait = max(0,obj.ready_time(p) - currentTime);",
            "                    if obj.has_waiting_time",
            "                        waitPenalty = waitPenalty + wait;",
            "                    end",
            "                    currentTime = max(currentTime,obj.ready_time(p));",
            "                    late = max(0,currentTime - obj.due_time(p));",
            "                    if obj.has_time_window",
            "                        latePenalty = latePenalty + late;",
            "                    end",
            "                    if obj.has_priority_class",
            "                        priorityPenalty = priorityPenalty + obj.priority(p) * (wait + late);",
            "                    end",
            "                    if obj.has_skill_matching",
            "                        skillPenalty = skillPenalty + obj.c_skill * max(0,obj.required_skill(p) - obj.skill_level(k));",
            "                    end",
            "                    currentTime = currentTime + obj.service_time(p);",
            "                    workload(k) = workload(k) + leg + wait + obj.service_time(p);",
            "                    previous = node;",
            "                end",
            "                if ~obj.has_open_route && previous ~= obj.depot",
            "                    leg = obj.travel_time(previous,obj.depot);",
            "                    travel = travel + leg;",
            "                    currentTime = currentTime + leg;",
            "                    workload(k) = workload(k) + leg;",
            "                end",
            "            end",
            "            if obj.has_outsourcing && obj.has_priority_class",
            "                priorityPenalty = priorityPenalty + 10*sum(obj.priority(logical(outsourced)));",
            "            end",
            "        end",
            "",
            "        function penalty = workload_imbalance(obj,workload)",
            "            %#ok<INUSD>",
            "            if isempty(workload)",
            "                penalty = 0;",
            "            else",
            "                penalty = max(workload) - min(workload);",
            "            end",
            "        end",
            "",
            "        function obj = generate_test_data(obj)",
            "            rng(20260605);",
            "            theta = linspace(0,2*pi,obj.P+1)';",
            "            radius = 10 + 30*rand(obj.P,1);",
            "            obj.locations = [0 0; radius.*cos(theta(1:obj.P)), radius.*sin(theta(1:obj.P))];",
            "            obj.travel_time = zeros(obj.P+1,obj.P+1);",
            "            for i = 1:obj.P+1",
            "                for j = 1:obj.P+1",
            "                    obj.travel_time(i,j) = norm(obj.locations(i,:) - obj.locations(j,:));",
            "                end",
            "            end",
            "            obj.service_time = 20 + 25*rand(obj.P,1);",
            "            baseReady = 480 + 300*rand(obj.P,1);",
            "            width = 90 + 120*rand(obj.P,1);",
            "            obj.ready_time = baseReady;",
            "            obj.due_time = baseReady + width;",
            "            obj.skill_level = 2 + mod((1:obj.K)'-1,4);",
            "            obj.required_skill = 1 + mod((1:obj.P)'-1,4);",
            "            obj.priority = 0.2 + 0.8*rand(obj.P,1);",
            "            avgService = sum(obj.service_time)/max(1,obj.K);",
            "            avgTravel = 2*mean(obj.travel_time(1,2:end))*ceil(obj.P/max(1,obj.K));",
            "            obj.workload_limit = max(240,1.15*(avgService + avgTravel))*ones(1,obj.K);",
            "        end",
            "",
            "        function R = GetOptimum(obj,N)",
            "            %#ok<INUSD>",
            "            if obj.M >= 3",
            "                R = [max(1,obj.P*120), max(1,obj.P*80), max(1,obj.P*20)];",
            "            else",
            "                R = [max(1,obj.P*160), max(1,obj.P*30)];",
            "            end",
            "        end",
            "",
            "        function R = GetPF(obj)",
            "            R = [];",
            "        end",
            "    end",
            "end",
            "",
        ]
    )


def _matlab_bool(value: bool) -> str:
    return "true" if value else "false"


def _matlab_cellstr(values: list[str]) -> str:
    if not values:
        return "{}"
    escaped = [str(value).replace("'", "''") for value in values]
    return "{" + ",".join(f"'{value}'" for value in escaped) + "}"


def _component_comments(label: str, values: list[str]) -> list[str]:
    if not values:
        return []
    text = f"% {label}: " + ", ".join(values)
    max_len = 96
    if len(text) <= max_len:
        return [text]
    lines: list[str] = []
    prefix = f"% {label}: "
    current = prefix
    for value in values:
        piece = value if current == prefix else ", " + value
        if len(current) + len(piece) > max_len:
            lines.append(current.rstrip())
            current = "%   " + value
        else:
            current += piece
    if current.strip():
        lines.append(current.rstrip())
    return lines


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out
