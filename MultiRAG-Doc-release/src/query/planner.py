"""Query Planner：将用户原始问题结构化为 QueryPlan。"""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.config import CFG
from src.generator.llm_client import generate

_MAX_SUB_QUERIES = 3

_PLANNER_SYSTEM_PROMPT = """\
You are a query planner for a scientific paper retrieval system.

Given a user question, analyze it and output a JSON query plan.
Be CONSERVATIVE: default to 1 sub_query. Only expand to 2-3 when the question \
clearly contains multiple independent sub-questions or requires fundamentally \
different retrieval aspects.

Output ONLY a valid JSON object with this exact structure:
{
  "normalized_question": "<rewritten query, retrieval-friendly; use original if already clear>",
  "intent": "<one-line description of what the user wants to know>",
  "use_original_question": <true if original question is already retrieval-friendly>,
  "sub_queries": ["<query1>", "<query2?>", "<query3?>"],
  "retrieval_routes": {"text": true, "caption": true, "image": true},
  "expected_evidence_types": ["<method|experiment|figure|table|definition|comparison>"],
  "answer_mode": "default",
  "planner_rationale": {
    "question_type": "<factoid|explanation|comparison|analysis|how_to|unknown>",
    "complexity": "<simple|medium|complex>",
    "subquery_strategy": "<single_query|rewrite_only|split_by_subquestion|split_by_aspect>",
    "answer_expectation": "default",
    "use_original_question": <true|false>,
    "query_decision": {
      "reason": "<why original/rewrite/expansion was chosen>",
      "selected_query_count": <int equal to len(sub_queries)>
    },
    "evidence_expectation": {
      "needs_text": true,
      "needs_caption": true,
      "needs_image": true,
      "expected_evidence_types": ["<list>"]
    },
    "risk_flags": [],
    "why_this_plan": "<short structured explanation>"
  }
}

Constraints:
- sub_queries: maximum 3 items
- retrieval_routes: all three always true (routing not yet implemented)
- answer_mode: always "default" for now
- normalized_question and all sub_queries MUST be written in English, regardless of the user's input language
- Output ONLY the JSON object, no extra text"""


@dataclass
class QueryPlan:
    original_question: str
    normalized_question: str
    intent: str
    use_original_question: bool
    sub_queries: list[str]
    retrieval_routes: dict[str, bool]
    expected_evidence_types: list[str]
    answer_mode: str
    planner_rationale: str


class QueryPlanner:
    """将用户原始问题规划为结构化 QueryPlan。"""

    def plan(self, question: str) -> QueryPlan:
        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": f"User question: {question}"},
        ]
        raw = generate(
            messages=messages,
            model=CFG.generator.model_name,
            temperature=0.0,
            max_tokens=1024,
        )
        return self._parse(question, raw)

    def _parse(self, original_question: str, raw: str) -> QueryPlan:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"[QueryPlanner] JSON 解析失败：{e}\nLLM 原始输出：\n{raw}"
            ) from e

        sub_queries: list[str] = data.get("sub_queries", [])
        if not sub_queries:
            raise ValueError("[QueryPlanner] sub_queries 为空，planner 输出无效。")
        if len(sub_queries) > _MAX_SUB_QUERIES:
            sub_queries = sub_queries[:_MAX_SUB_QUERIES]

        rationale = data.get("planner_rationale", {})
        if isinstance(rationale, dict):
            rationale_str = json.dumps(rationale, ensure_ascii=False)
        else:
            rationale_str = str(rationale)

        return QueryPlan(
            original_question=original_question,
            normalized_question=data.get("normalized_question", original_question),
            intent=data.get("intent", ""),
            use_original_question=bool(data.get("use_original_question", True)),
            sub_queries=sub_queries,
            retrieval_routes=data.get(
                "retrieval_routes", {"text": True, "caption": True, "image": True}
            ),
            expected_evidence_types=data.get("expected_evidence_types", []),
            answer_mode=data.get("answer_mode", "default"),
            planner_rationale=rationale_str,
        )

