"""Answer Prompt Selector：根据 question_type 路由差异化 system prompt。

设计：
- 共享基座 + type-specific guidance（第 2 条规则）。
- unknown / 缺失 / 非法值统一回退到默认 guidance，与改动前行为一致。
- 对外暴露 get_guidance() 和 render_system_prompt()。
"""

from __future__ import annotations

import json

VALID_QUESTION_TYPES = frozenset(
    {"factoid", "explanation", "comparison", "analysis", "how_to", "unknown"}
)

_DEFAULT_GUIDANCE = (
    "Answer in a concise but complete paragraph. Stay grounded in the evidence. "
    "Prioritize the most directly relevant findings, and omit anything that is "
    "not explicitly supported."
)

_GUIDANCE: dict[str, str] = {
    "factoid": (
        "Answer briefly and directly. Prefer a single concise paragraph. "
        "State the exact fact, entity, number, or result asked for. "
        "If multiple candidate answers appear in the evidence, mention only those "
        "explicitly supported and distinguish them clearly. Do not speculate. "
        "Avoid unnecessary formatting."
    ),
    "explanation": (
        "Explain the concept or mechanism step by step. Organize the answer as: "
        "**Core Idea** -> **Mechanism / Process** -> **Supporting Evidence or Example**. "
        "Use clear paragraph breaks or bullet points to separate these parts. "
        "If equations are present in the evidence, include them using LaTeX format and explain their role. "
        "Make the logical flow explicit, and ensure each step is supported by citations."
    ),
    "comparison": (
        "Organize the answer around similarities and differences. Compare the items "
        "dimension by dimension (e.g., objective, method, architecture, data, performance, "
        "limitations), but include only dimensions supported by the evidence. "
        "Use structured formatting (bullet points or a Markdown table) if it improves clarity. "
        "If one side of the comparison is missing evidence for a dimension, state "
        "'Not mentioned in the evidence' rather than inferring."
    ),
    "analysis": (
        "Provide a structured analytical answer. Organize the answer using clear sections or bullet points. "
        "Cover, where supported by evidence: "
        "(1) main finding or claim, (2) supporting evidence, (3) implications, and "
        "(4) limitations, assumptions, or caveats. "
        "Clearly distinguish between what is directly stated in the evidence and what is not addressed."
    ),
    "how_to": (
        "Provide a step-by-step answer using a numbered list. Include only steps that are "
        "explicitly supported by the evidence. Each step must be concrete and actionable. "
        "If prerequisites, conditions, or limitations are mentioned in the evidence, include them. "
        "Use Markdown numbered lists for clarity."
    ),
    "unknown": _DEFAULT_GUIDANCE,
}

_SYSTEM_PROMPT_TEMPLATE = """\
You are a scientific paper Q&A assistant. Answer the user's question based ONLY on the
provided evidence, which has been pre-selected and ranked by relevance.

Rules:
1. Use ONLY the retrieved evidence below. Do NOT use external knowledge.
2. First determine whether the evidence is sufficient to answer the question.
3. If the evidence is insufficient or no relevant evidence is provided, reply exactly:
   I cannot answer based on the available evidence.
4. Every factual statement MUST be directly supported by at least one cited evidence block.
5. Attach citations immediately after the statement they support, using the format [chunk_id].
6. Do NOT cite any chunk that does not directly support the statement.
7. Do NOT make unsupported claims, and do NOT infer missing details. If something is not stated
   in the evidence, either omit it or say: Not mentioned in the evidence.
8. Prefer specific details from the evidence, such as numbers, equations, experimental results,
   settings, ablations, and direct claims, when they are relevant.
9. Answer in {answer_language}. If the user's question is in a different language, still answer in {answer_language}.
10. Keep technical terms, model names, metrics, and key concepts in their original English form. Do NOT translate standard terminology.
11. Format the answer using Markdown for readability:
    - Use paragraphs, bullet points, and bold text where appropriate.
    - Use LaTeX for mathematical expressions:
        - Inline math MUST use $...$
        - Block math MUST use $$...$$
    - Do NOT wrap the entire answer in code blocks.

12. Follow this answer-style guidance for the current question type:
   {guidance}

13. Respond in this exact format:

Answer:
<your answer body in Markdown format; 
use multiple paragraphs, lists, or sections if needed for clarity.
Mathematical expressions must be written in LaTeX format.
Inline citations must be attached immediately after the statements they support.>

Citations: 
<comma-separated chunk IDs actually used in the answer, e.g. [chunk_id_1], [chunk_id_2]>

14. Only use citation keys from this valid set:
   {valid_keys}
"""


# ── 公开接口 ──────────────────────────────────────────────────────────────────
# NOTE: 后续可以继续控制多语言输出 + 输出长度偏好设计等 prompt 注入
def get_guidance(question_type: str | None) -> str:
    """返回对应 question_type 的 guidance 文本；非法值回退到默认。"""
    if question_type and question_type in _GUIDANCE:
        return _GUIDANCE[question_type]
    return _DEFAULT_GUIDANCE


def render_system_prompt(
    question_type: str | None,
    valid_keys: str,
    answer_language: str = "English",
) -> str:
    """渲染最终 system prompt，嵌入 type-specific guidance、valid_keys 和 answer_language。"""
    return _SYSTEM_PROMPT_TEMPLATE.format(
        guidance=get_guidance(question_type),
        valid_keys=valid_keys if valid_keys else "(none)",
        answer_language=answer_language or "English",
    )


def parse_question_type_from_rationale(rationale: str | None) -> str:
    """从 planner_rationale JSON 字符串中解析 question_type；非法值或解析失败时回退到 'unknown'。"""
    try:
        data = json.loads(rationale) if rationale else {}
        qt = data.get("question_type", "unknown")
        return qt if qt in VALID_QUESTION_TYPES else "unknown"
    except Exception:
        return "unknown"
