"""Loop Controller：Agentic RAG Loop 的业务协议层。

职责：
- 定义 step 执行协议、termination 调用顺序、system hook 规则。
- 负责与 AgentState、ToolRegistry、TerminationPolicy、EvidenceStore 的业务集成。
- 不重复 langgraph_agent.py 的框架适配逻辑。

业务规则（这里是唯一权威定义）：
- agent_policy.decide() 第一版为结构化决策输出，不采用自由文本 ReAct 解析。
- 证据直接由 reranker 打分，无事后 LLM 评分步骤。
- citation recovery 最多 1 轮（expand_evidence -> regenerate）。
- rolling_summary 每步 observation 后更新一次。
- budget 耗尽（evidence_budget_remaining == 0）时系统自动路由到 finish。
- 总步数上界：max_steps（普通 loop）+ 1（citation recovery）= 9。
"""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.query.service import QueryService

from collections.abc import Callable

from src.agent.evidence_compressor import EvidenceCompressor
from src.agent.evidence_scorer import ScoredEvidence
from src.agent.evidence_store import EvidenceStore
from src.agent.state import (
    AgentState,
    Decision,
    StateDelta,
    StepObservation,
)
from src.agent.termination import TerminationPolicy
from src.agent.tool_registry import ToolRegistry
from src.config import CFG
from src.generator.answer_prompt_selector import parse_question_type_from_rationale
from src.generator.llm_client import generate
from src.query.answer_synthesizer import AnswerSynthesizer

# ── 默认参数 ─────────────────────────────────────────────────────────────────

_CONTEXT_OBS_WINDOW = 2            # LLM 可见最近 N 轮 observation


# ── Agent Policy Prompt ───────────────────────────────────────────────────────

_POLICY_SYSTEM_PROMPT = """\
You are an evidence-gathering agent for a scientific paper Q&A system.
Your job is to iteratively collect evidence to answer a research question.

At each step, you MUST follow this two-phase process:
PHASE 1 — REFLECT: Analyze what you know and what you still need.
PHASE 2 — ACT: Based on the gaps identified, decide ONE action.

Actions:
- search_evidence: Retrieve more evidence with a specific query
- finish: You have enough evidence to answer the question
- abort: Evidence is insufficient and more searching won't help

Rules:
1. For search_evidence:
  - Query must be in English, specific, and NOT semantically similar to previous_queries
  - Prefer refining or narrowing queries instead of rephrasing
  - Avoid repeating the same intent with different wording
2. modalities: a JSON array; pick one or both from ["text", "figure"]. Use ["text"] for
  concept/method questions, ["figure"] for visual results, ["text", "figure"] for both.
3. expected_evidence_count: how many items to retrieve (1-5); 
  use fewer for factoid questions,  more for explanatory/survey questions.
  Evidence budget is limited. If the accumulated evidence reaches the budget limit,
  the system will automatically compress the evidence set by dropping the least useful items.
  Compression is a fallback, not a substitute for good retrieval decisions.
4. For finish:
  - Only choose finish if ALL reflection.gaps are covered by existing evidence
  - AND you have sufficient high-quality evidence (at least 2–3 strong pieces)
5. For abort:
  - Choose abort if after 3 steps reflection.gaps have not decreased
  - OR evidence is consistently irrelevant or insufficient
6. state_update is REQUIRED every step; update hypotheses and open_questions to reflect
  your current understanding (first step: keep original open_questions if unchanged)

Output ONLY a valid JSON object with this exact structure:
{
  "reflection": {
    "current_understanding": "<based on existing evidence, your current understanding of the answer (1-2 sentences)>",
    "hypotheses": ["<current hypothesis about the answer>"],
    "gaps": ["<what information is still missing to confirm or refute the hypothesis>"]
  },
  "state_update": {
    "normalized_question": "<optional: refined question>",
    "current_objective": "<optional: what you're currently trying to find>",
    "open_questions": ["<align with reflection.gaps, written as searchable sub-questions>"],
    "hypotheses": ["<align with reflection.hypotheses>"]
  },
  "action": "<search_evidence|finish|abort>",
  "args": {
    "query": "<search query in English, only if action=search_evidence>",
    "modalities": ["text"],
    "expected_evidence_count": <int 1-5, how many evidence items you want from this search>
  },
  "reasoning": "<1 sentence: why you chose this action given the gaps>"
}
"""


# ── Loop Controller ───────────────────────────────────────────────────────────

class LoopController:
    """Agentic RAG Loop 的业务协议实现（纯 Python fallback）。

    LangGraph 适配层（langgraph_agent.py）调用此类的各方法完成节点职责，
    不在此类之外重复定义业务规则。
    """

    def __init__(
        self,
        service: "QueryService",
        termination: TerminationPolicy | None = None,
        max_steps: int = 10,
    ) -> None:
        self._service = service
        self._store = EvidenceStore()
        self._evidence_cap = CFG.agent.evidence_cap
        self._registry = ToolRegistry(
            service=service,
            store=self._store,
            evidence_cap=self._evidence_cap,
        )
        self._termination = termination or TerminationPolicy(max_steps=max_steps)
        self._answer_agent = AnswerSynthesizer()
        self._compressor = EvidenceCompressor()

    # ── 决策节点（LangGraph: decide_action node）────────────────────────────

    def decide_action(
        self,
        state: AgentState,
        debug_callback: Callable[[dict], None] | None = None,
    ) -> Decision:
        """调用 LLM，获取结构化 Decision。

        不采用自由文本 ReAct 解析，直接输出结构化 JSON。
        debug_callback: 若非 None，调用后以 {prompt, raw_output} 通知调用方。
        """
        messages = [
            {"role": "system", "content": _POLICY_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_policy_user_message(state)},
        ]
        raw = generate(
            messages=messages,
            model=CFG.generator.model_name,
            temperature=0.0,
            max_tokens=800,
        )
        if debug_callback is not None:
            debug_callback({"prompt": messages, "raw_output": raw})
        return self._parse_decision(raw)

    # ── Tool 执行节点（LangGraph: run_tool node）────────────────────────────

    def run_tool_step(
        self,
        decision: Decision,
        state: AgentState,
        step: int,
    ) -> StepObservation:
        """执行 tool，生成并记录 observation，更新 rolling_summary。"""
        success = True
        error = ""
        new_ids: list[str] = []

        try:
            if decision.action == "search_evidence":
                new_ids, _ = self._registry.execute("search_evidence", decision.args, state)
                state.record_tool_event(
                    step=step, tool_name="search_evidence",
                    args=decision.args, success=True,
                )
            # expand_evidence（系统内部，Phase 1 citation recovery 专用）
            # 此处不直接处理，由 _citation_recovery 调用
        except Exception as exc:
            success = False
            error = str(exc)
            state.record_tool_event(
                step=step, tool_name=decision.action,
                args=decision.args, success=False, error=error,
            )

        # 更新 state 的 evidence 视图（单一视图）
        state.evidence = self._store.to_evidence_views()

        # 构造 observation
        obs = self._build_step_observation(state, new_ids, step)
        state.observation_history.append(obs)

        # 更新 rolling_summary
        self._update_rolling_summary(state, obs)

        return obs

    # ── Citation Recovery（系统内部，Phase 1 最多 1 轮）────────────────────

    def citation_recovery(
        self,
        state: AgentState,
        all_chunks: list[dict[str, Any]],
        citation_report: dict[str, Any],
    ) -> None:
        """citation check 失败后执行 expand_evidence 补救。

        Phase 1 默认使用 anchor_set="selected"（fallback）。
        邻接 chunk 直接追加到 EvidenceStore，retrieval_score=0.0，排在末尾。
        """
        if not self._termination.can_do_citation_recovery(state):
            return
        state.citation_recovery_used = True

        anchor_ids = [v.chunk_id for v in state.evidence]
        self._store.expand_adjacent(anchor_ids, all_chunks, window=1)

        # 更新证据视图
        state.evidence = self._store.to_evidence_views()

    # ── 回答生成 ─────────────────────────────────────────────────────────────

    def _generate_and_validate(
        self,
        state: AgentState,
        answer_language: str,
        stream_callback: Any,
        trace: list[dict],
        debug_callback: Any = None,
    ) -> dict[str, Any]:
        """调用回答生成器生成回答，citation 失败时尝试 recovery。"""
        # 低证据 warning
        for code in self._termination.check_low_evidence_warning(state):
            state.add_warning(
                code=code,
                message=(
                    "Evidence coverage is below the recommended threshold; "
                    "the answer was generated due to an early finish signal."
                ),
                step=state.current_step(),
            )

        scored_ev = self._store.get_all_as_scored_evidence()
        result = self._answer_agent.answer(
            question=state.original_question,
            evidence=scored_ev,
            generate_answer=True,
            question_type=state.question_type,
            stream_callback=stream_callback,
            answer_language=answer_language,
            debug_callback=debug_callback,
        )

        # Citation check 失败 -> recovery
        if result.get("guardrail_reason") and not state.citation_recovery_used:
            all_chunks = self._get_all_chunks()
            if all_chunks:
                self.citation_recovery(
                    state=state,
                    all_chunks=all_chunks,
                    citation_report={"ok": False},
                )
                scored_ev2 = self._store.get_all_as_scored_evidence()
                result2 = self._answer_agent.answer(
                    question=state.original_question,
                    evidence=scored_ev2,
                    generate_answer=True,
                    question_type=state.question_type,
                    stream_callback=stream_callback,
                    answer_language=answer_language,
                    debug_callback=debug_callback,
                )
                if not result2.get("guardrail_reason"):
                    result = result2
                else:
                    state.add_warning(
                        code="CITATION_RECOVERY_FAILED",
                        message=(
                            "Citation grounding may be incomplete because "
                            "the selected evidence set did not meet the target threshold."
                        ),
                        step=state.current_step(),
                    )

        return self._build_final_result(
            state,
            generate_answer=True,
            answer_language=answer_language,
            stream_callback=stream_callback,
            trace=trace,
            answer_result=result,
            question_type=state.question_type,
        )

    def _get_all_chunks(self) -> list[dict[str, Any]]:
        """从 MetadataStore 获取全量 chunk 列表（供 expand_adjacent 使用）。"""
        try:
            return self._service._ms.all_chunks()
        except Exception:
            return []

    # ── Observation 构造 ─────────────────────────────────────────────────────

    def _build_step_observation(
        self,
        state: AgentState,
        new_ids: list[str],
        step: int,
    ) -> StepObservation:
        """构造结构化 observation（模板生成，不引入额外 LLM）。"""
        new_evidence_count = len(new_ids)
        new_recs = [self._store.get_by_record_id(rid) for rid in new_ids if rid]
        papers_hit = list({r.paper_id for r in new_recs if r})
        modalities_hit = list({r.modality for r in new_recs if r})

        # coverage_delta：本轮新增证据覆盖了多少 open_questions
        coverage_delta = min(1, len(state.open_questions)) if new_evidence_count > 0 else 0

        last_query = ""
        for ev in reversed(state.tool_history):
            if ev.tool_name == "search_evidence":
                last_query = ev.args.get("query", "")
                break

        summary = (
            f"searched={last_query!r}; new={new_evidence_count}; "
            f"text_evidence={self._store.count_text()}; "
            f"budget_remaining={state.evidence_budget_remaining}; "
            f"papers={papers_hit}; "
            f"open_q={len(state.open_questions)}"
        )

        return StepObservation(
            step=step,
            new_evidence_count=new_evidence_count,
            new_ids=new_ids,
            papers_hit=papers_hit,
            modalities_hit=modalities_hit,
            coverage_delta=coverage_delta,
            conflicts=[],
            summary=summary,
        )

    # ── Rolling Summary ───────────────────────────────────────────────────────

    def _update_rolling_summary(self, state: AgentState, obs: StepObservation) -> None:
        """规则式更新 rolling_summary（代码侧生成，不引入 LLM）。"""
        parts = []
        if state.evidence:
            parts.append(f"evidence={len(state.evidence)} items")
        if state.open_questions:
            parts.append(f"open_questions={state.open_questions}")
        if state.hypotheses:
            parts.append(f"hypotheses={state.hypotheses}")
        failures = [e for e in state.tool_history if not e.success]
        if failures:
            parts.append(f"last_failure={failures[-1].error!r}")

        state.rolling_summary = "; ".join(parts)

    # ── Evidence 压缩（budget 耗尽时触发，最多 1 次）──────────────────────────

    def _try_compress_evidence(
        self,
        state: AgentState,
        trace: list[dict],
        step: int,
    ) -> bool:
        """budget 耗尽时尝试压缩。返回 True 表示成功腾出 budget，False 表示无法压缩。

        触发条件（全部满足）：
        - state.evidence_budget_remaining == 0
        - not state.compression_used
        - store.count_text() > 3（至少有可丢弃空间）
        """
        if state.compression_used or self._store.count_text() <= 3:
            return False

        before = self._store.count_text()
        discard_ids, reasoning = self._compressor.compress(
            original_question=state.original_question,
            store=self._store,
            model=CFG.generator.model_name,
        )

        state.compression_used = True

        if not discard_ids:
            trace.append({
                "step": "compress_evidence",
                "before": before,
                "kept": self._store.count_text(),
                "discarded": [],
                "reasoning": reasoning,
            })
            return False

        actually_removed = self._store.remove_by_chunk_ids(discard_ids)
        state.evidence = self._store.to_evidence_views()

        fallback = reasoning == "fallback: score-based pruning (LLM parse failed)"
        trace_entry: dict = {
            "step": "compress_evidence",
            "before": before,
            "kept": self._store.count_text(),
            "discarded": actually_removed,
            "reasoning": reasoning,
        }
        if fallback:
            trace_entry["fallback"] = True
        trace.append(trace_entry)

        state.add_warning(
            code="EVIDENCE_COMPRESSED",
            message=(
                f"Evidence pool compressed: {before} → {self._store.count_text()} "
                f"text chunks (discarded {len(actually_removed)})."
            ),
            step=step,
        )

        return self._store.count_text() < self._evidence_cap

    # ── 可选 plan_query（Step 0 辅助）───────────────────────────────────────

    def _try_plan_query(
        self,
        state: AgentState,
        trace: list[dict],
        step: int,
    ) -> None:
        """可选调用 QueryPlanner，更新 open_questions（Step 0 辅助）。"""
        try:
            from src.query.planner import QueryPlanner
            planner = QueryPlanner()
            plan = planner.plan(state.original_question)
            state.question_type = parse_question_type_from_rationale(plan.planner_rationale)
            if plan.sub_queries:
                delta = StateDelta(
                    normalized_question=plan.normalized_question,
                    current_objective=plan.normalized_question,
                    open_questions=plan.sub_queries,
                )
                state.apply_state_delta(delta)
            trace.append({
                "step": "plan_query",
                "normalized_question": plan.normalized_question,
                "open_questions": state.open_questions,
            })
        except Exception as exc:
            trace.append({"step": "plan_query", "error": str(exc)})

    # ── Policy Message 构造 ────────────────────────────────────────────────────

    def _build_policy_user_message(self, state: AgentState) -> str:
        """Build the user message for the agent policy LLM call (state summary)."""
        used = self._store.count_text()
        cap = self._evidence_cap
        remaining = max(0, cap - used)

        lines = [
            f"Original question: {state.original_question}",
            f"Current objective: {state.current_objective}",
            f"Open questions: {state.open_questions}",
            f"Hypotheses: {state.hypotheses or '(none)'}",
            f"Step: {state.current_step()} / {CFG.agent.max_steps}",
            "",
            f"Evidence budget: {used} / {cap} used, {remaining} remaining",
            "",
        ]

        recent_obs = state.get_recent_observations(_CONTEXT_OBS_WINDOW)
        if recent_obs:
            lines.append("Recent observations:")
            for obs in recent_obs:
                lines.append(f"  Step {obs.step}: {obs.summary}")
            lines.append("")
        else:
            lines.append("Observations: (none yet)")
            lines.append("")

        if state.rolling_summary:
            lines.append(f"Rolling summary: {state.rolling_summary}")
            lines.append("")

        if state.evidence:
            # NOTE: 全量展示（cap=8 时无截断问题）；若未来 cap 变大需考虑 prompt 压缩
            lines.append(f"Current evidence ({len(state.evidence)} items):")
            for v in state.evidence:
                preview = v.content[:120].replace("\n", " ")
                lines.append(
                    f"  [{v.chunk_id}] ({v.paper_id}, page {v.page}, rerank={v.final_score:.4f}): {preview}..."
                )
            lines.append("")

        prior_queries = [e.args.get("query", "") for e in state.tool_history if e.tool_name == "search_evidence"]
        if prior_queries:
            lines.append(f"Prior search queries: {prior_queries}")

        return "\n".join(lines)

    # ── Decision 解析 ─────────────────────────────────────────────────────────

    def _parse_decision(self, raw: str) -> Decision:
        """解析 LLM 的结构化决策输出。"""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            ).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return Decision(action="abort")

        action = str(data.get("action", "abort")).lower()
        if action not in ("search_evidence", "expand_evidence", "finish", "abort"):
            action = "abort"

        args = data.get("args") or {}
        if not isinstance(args, dict):
            args = {}

        state_update_data = data.get("state_update") or {}
        state_update: StateDelta | None = None
        if isinstance(state_update_data, dict) and state_update_data:
            state_update = StateDelta(
                normalized_question=state_update_data.get("normalized_question"),
                current_objective=state_update_data.get("current_objective"),
                open_questions=state_update_data.get("open_questions"),
                hypotheses=state_update_data.get("hypotheses"),
            )

        reflection = data.get("reflection") or {}
        reasoning_parts = []
        if isinstance(reflection, dict):
            if reflection.get("current_understanding"):
                reasoning_parts.append(reflection["current_understanding"])
            gaps = reflection.get("gaps", [])
            if gaps:
                reasoning_parts.append(f"gaps: {gaps}")
        if not reasoning_parts:
            reasoning_parts.append(str(data.get("reasoning", "")))
        reasoning = " | ".join(reasoning_parts)
        return Decision(action=action, args=args, state_update=state_update, reasoning=reasoning)

    # ── 结果构造 ─────────────────────────────────────────────────────────────

    def _build_final_result(
        self,
        state: AgentState,
        generate_answer: bool,
        answer_language: str,
        stream_callback: Any,
        trace: list[dict],
        answer_result: dict[str, Any] | None = None,
        question_type: str = "unknown",
    ) -> dict[str, Any]:
        """构造最终返回结果（含 warnings、evidence、agent_trace）。"""
        scored_ev = self._store.get_all_as_scored_evidence()

        if answer_result is None:
            if generate_answer and scored_ev:
                answer_result = self._answer_agent.answer(
                    question=state.original_question,
                    evidence=scored_ev,
                    generate_answer=True,
                    question_type=question_type,
                    stream_callback=stream_callback,
                    answer_language=answer_language,
                )
            else:
                answer_result = {
                    "results": [se.to_evidence_dict() for se in scored_ev],
                    "answer": None,
                    "guardrail_reason": state.terminate_reason or "",
                }

        return {
            "results": answer_result.get("results", []),
            "answer": answer_result.get("answer"),
            "guardrail_reason": answer_result.get("guardrail_reason", ""),
            "warnings": [
                {"code": w.code, "message": w.message, "step": w.step}
                for w in state.warnings
            ],
            "agent_trace": trace,
            "terminate_reason": state.terminate_reason,
            "selected_evidence_count": len(state.evidence),
        }


# ── 模块级私有辅助函数 ─────────────────────────────────────────────────────────

def _serialize_new_item(rid: str, store: "EvidenceStore") -> dict:
    """将 EvidenceStore 中的 record 序列化为前端 new_item schema。

    figure 的 image_path → image_url 转换在此处完成，不依赖 web 层。
    """
    from pathlib import Path
    from urllib.parse import quote

    rec = store.get_by_record_id(rid)
    if rec is None:
        return {
            "chunk_id": rid,
            "paper_id": "",
            "modality": "text",
            "page": None,
            "score": 0.0,
            "content": "",
            "content_preview": "",
            "section": "",
        }

    image_url: str | None = None
    if rec.image_path:
        try:
            parts = Path(rec.image_path).parts
            if len(parts) >= 2:
                pid, fname = parts[-2], parts[-1]
                if fname.endswith(".png"):
                    image_url = f"/figures/{quote(pid)}/{quote(fname)}"
        except Exception:
            pass

    item: dict = {
        "chunk_id": rec.chunk_id,
        "paper_id": rec.paper_id,
        "modality": rec.modality,
        "page": rec.page,
        "score": float(rec.retrieval_score),
        "content": rec.content,
        "content_preview": rec.content_preview(),
        "section": rec.section,
    }
    if rec.figure_id:
        item["figure_id"] = rec.figure_id
    if image_url:
        item["image_url"] = image_url
    if rec.modality == "figure":
        item["caption"] = rec.content
    return item
