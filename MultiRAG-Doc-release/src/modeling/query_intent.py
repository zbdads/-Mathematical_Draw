"""Lightweight intent helpers for math-modeling retrieval."""

from __future__ import annotations

_MATH_QUERY_KEYWORDS = (
    "mathematical model",
    "optimization model",
    "formulation",
    "five-element",
    "sets",
    "parameters",
    "decision variable",
    "variables",
    "objective function",
    "objective",
    "constraints",
    "constraint",
    "assumption",
    "model card",
    "pyomo",
    "目标函数",
    "约束",
    "变量",
    "参数",
    "集合",
    "数学模型",
    "优化模型",
    "建模",
)

_FIELD_KEYWORDS = {
    "variables": ("decision variable", "variables", "变量", "决策变量"),
    "objective": ("objective function", "objective", "目标函数", "目标"),
    "constraints": ("constraints", "constraint", "约束"),
    "parameters": ("parameters", "parameter", "参数"),
    "sets": ("sets", "indices", "index sets", "集合", "索引"),
}


def is_math_modeling_query(question: str) -> bool:
    q = question.lower()
    return any(keyword in q for keyword in _MATH_QUERY_KEYWORDS)


def expected_model_fields(question: str) -> list[str]:
    q = question.lower()
    fields = [
        field
        for field, keywords in _FIELD_KEYWORDS.items()
        if any(keyword in q for keyword in keywords)
    ]
    return fields
