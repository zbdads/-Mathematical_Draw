"""Prompts for extracting LLMOPT-style five-element model cards."""

from __future__ import annotations

MODEL_CARD_SYSTEM_PROMPT = """You are an expert in operations research and optimization modeling.

Your task is to extract a faithful optimization model card from paper evidence.
Use only the provided evidence. Do not invent symbols, objectives, or constraints.
If a field is not supported by the evidence, use an empty list or an empty string.

Return strict JSON only, with this schema:
{
  "title": string,
  "problem_type": string,
  "application_domain": string,
  "model_name": string,
  "sets": [string],
  "parameters": [string],
  "variables": [string],
  "objective": [string],
  "constraints": [string],
  "assumptions": [string],
  "algorithm": [string],
  "evidence_refs": [
    {
      "chunk_id": string,
      "role": "problem_definition|sets|parameters|variables|objective|constraints|assumptions|algorithm|other",
      "quote": string
    }
  ],
  "confidence": number,
  "warnings": [string]
}

Extraction rules:
- Prefer exact mathematical statements, variable definitions, objective functions, and constraints.
- Preserve equation numbers and symbol names when present.
- Separate decision variables from parameters.
- Classify the optimization type, e.g. LP, IP, MILP, NLP, MOP, scheduling, routing, combinatorial optimization.
- The evidence_refs must cite chunk IDs from the provided evidence blocks only.
- confidence should be 0.0 to 1.0 based on how complete the evidence is.
"""


def build_model_card_user_prompt(
    paper_id: str,
    evidence_blocks: list[str],
) -> str:
    evidence = "\n\n".join(evidence_blocks) if evidence_blocks else "(no evidence)"
    return f"""Extract the optimization model card for paper_id={paper_id}.

The five-element formulation should follow the LLMOPT abstraction:
Sets, Parameters, Variables, Objective, Constraints.

Paper evidence:
{evidence}

Return JSON only.
"""


MODEL_FIELD_SYSTEM_PROMPT = """You are an expert in operations research and optimization modeling.

Your task is to extract one field of an optimization model card from paper evidence.
Use only the provided evidence. Do not invent symbols, objectives, or constraints.

Return strict JSON only, with this schema:
{
  "items": [
    {
      "text": string,
      "chunk_id": string,
      "quote": string
    }
  ],
  "confidence": number,
  "warnings": [string]
}

Extraction rules:
- Each item.text must be one complete retrieval-friendly entry for the requested field.
- Preserve mathematical symbols, equation numbers, quantifiers, and index ranges when present.
- chunk_id must be copied from one of the provided evidence blocks.
- quote should be a short evidence excerpt supporting item.text.
- If the requested field is not supported by the evidence, return items=[].
- Do not include markdown fences or explanatory prose.
"""


FIELD_DESCRIPTIONS = {
    "sets": (
        "sets / indices / entity collections, such as patients, jobs, machines, "
        "vehicles, nodes, stages, caregivers, arcs, or time periods. Exclude "
        "input coefficients and decision variables"
    ),
    "parameters": (
        "input parameters / constants / coefficients, such as costs, times, "
        "capacities, due dates, demands, weights, service requirements, and big-M. "
        "Exclude sets, indices, and decision variables"
    ),
    "variables": (
        "decision variables only, including binary, integer, and continuous "
        "variables with their meanings and domains. Exclude parameters and sets"
    ),
    "objective": (
        "objective function(s), optimization direction, objective components, "
        "and equation numbers. Exclude constraints unless they define an "
        "objective component"
    ),
    "constraints": (
        "constraints only, including equations/inequalities, equation numbers, "
        "and concise meanings. Exclude variable declarations unless they are "
        "domain constraints"
    ),
}


def build_model_field_user_prompt(
    paper_id: str,
    field: str,
    evidence_blocks: list[str],
) -> str:
    evidence = "\n\n".join(evidence_blocks) if evidence_blocks else "(no evidence)"
    field_description = FIELD_DESCRIPTIONS.get(field, field)
    return f"""Extract the {field} field for paper_id={paper_id}.

Requested field meaning: {field_description}.

Evidence:
{evidence}

Return JSON only.
"""
