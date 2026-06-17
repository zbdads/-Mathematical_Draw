"""LangGraph Agent：把 LoopController 的业务协议映射为 LangGraph state graph。

职责：
- 只负责 graph 节点连接、route signal 映射、runtime 调用入口。
- 不重复定义业务规则（termination、rolling_summary 等由 loop_controller.py 维护）。

graph 节点：
  decide_action -> run_tool / finish / abort
  run_tool -> apply_observation -> system_select_or_compact -> check_termination
  check_termination -> decide_action（继续）| answer_or_abort（终止）
  answer_or_abort -> END
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from src.query.service import QueryService

from langgraph.graph import StateGraph, END

from src.agent.loop_controller import LoopController, _serialize_new_item
from src.agent.state import AgentState, Decision, StateDelta
from src.agent.termination import TerminationPolicy


# ── LangGraph State ───────────────────────────────────────────────────────────

class GraphState(TypedDict, total=False):
    """LangGraph graph state（包含 controller 引用与 agent_state）。

    TypedDict 仅用于 LangGraph 框架层的状态传递；
    业务规则与字段定义仍在 AgentState / LoopController 中维护。
    """
    controller: LoopController        # 业务协议层（含 store/registry/termination）
    agent_state: AgentState           # loop 工作状态
    decision: Decision | None         # 当前步决策
    step: int                         # 当前 step 编号
    trace: list[dict]                 # 调试追踪列表
    should_finish: bool               # route: 进入回答阶段
    should_abort: bool                # route: 终止不回答
    generate_answer: bool             # 是否生成回答
    stream_callback: Any              # 流式回调
    node_callback: Any                # SSE 节点事件回调 (event_type, data) -> None
    debug_callback: Any               # SSE debug 事件回调 (data: dict) -> None，可为 None
    answer_language: str              # 回答语言
    final_result: dict | None         # 最终结果（answer_or_abort 写入）


# ── 辅助函数 ─────────────────────────────────────────────────────────────────

def _delta_to_dict(delta: "StateDelta | None") -> dict | None:
    """将 StateDelta 转为 JSON-safe dict（仅含非 None 字段）。"""
    if delta is None:
        return None
    return {
        k: v for k, v in {
            "normalized_question": delta.normalized_question,
            "current_objective": delta.current_objective,
            "open_questions": delta.open_questions,
            "hypotheses": delta.hypotheses,
        }.items() if v is not None
    }


# ── Public API ────────────────────────────────────────────────────────────────

class LangGraphAgent:
    """LangGraph agent entry point."""

    def __init__(
        self,
        service: "QueryService",
        max_steps: int = 10,
    ) -> None:
        self._controller = LoopController(
            service=service,
            max_steps=max_steps,
        )
        self._graph = _build_graph()

    def run(
        self,
        question: str,
        paper_id: str | None = None,
        generate_answer: bool = True,
        stream_callback: Any = None,
        node_callback: Any = None,
        debug_callback: Any = None,
        answer_language: str = "English",
    ) -> dict[str, Any]:
        """Run the agent loop and return a structured result dict."""
        state = AgentState.from_question(question, paper_id=paper_id)
        state.evidence_cap = self._controller._evidence_cap
        trace: list[dict] = []

        # Step 0: 始终运行 plan_query（获取 question_type + 可选 sub_queries）
        self._controller._try_plan_query(state, trace, step=0)
        if node_callback:
            node_callback("plan_query", {
                "original_question": state.original_question,
                "normalized_question": state.normalized_question,
                "sub_queries": state.open_questions,
                "question_type": state.question_type,
            })

        initial_gs: GraphState = {
            "controller": self._controller,
            "agent_state": state,
            "decision": None,
            "step": 0,
            "trace": trace,
            "should_finish": False,
            "should_abort": False,
            "generate_answer": generate_answer,
            "stream_callback": stream_callback,
            "node_callback": node_callback,
            "debug_callback": debug_callback,
            "answer_language": answer_language,
            "final_result": None,
        }

        final_gs = self._graph.invoke(initial_gs)
        result = final_gs.get("final_result")
        if result is None:
            result = self._controller._build_final_result(
                state=final_gs.get("agent_state", state),
                generate_answer=False,
                answer_language=answer_language,
                stream_callback=stream_callback,
                trace=final_gs.get("trace", trace),
            )
        return result


# ── Graph 节点定义 ────────────────────────────────────────────────────────────

def _node_decide_action(gs: "GraphState") -> "GraphState":
    """decide_action 节点：budget 耗尽时自动 finish，否则调用 LLM 获取结构化 Decision。

    所有终止信号（should_finish / should_abort）在本节点内解析完毕后写入返回值，
    route 函数只读不写。
    """
    controller: LoopController = gs["controller"]
    state: AgentState = gs["agent_state"]

    # budget 耗尽：先尝试压缩，成功则继续；否则路由到 finish
    if controller._store.count_text() >= controller._evidence_cap:
        step = gs.get("step", 0)
        trace: list[dict] = list(gs.get("trace", []))
        compressed = controller._try_compress_evidence(state, trace, step)

        # 发送压缩事件到前端
        _node_cb = gs.get("node_callback")
        if _node_cb:
            compress_entry = next(
                (e for e in reversed(trace) if e.get("step") == "compress_evidence"),
                None,
            )
            if compress_entry:
                _node_cb("compress_evidence", {
                    "step": step,
                    "before": compress_entry.get("before", 0),
                    "kept": compress_entry.get("kept", 0),
                    "discarded": compress_entry.get("discarded", []),
                    "reasoning": compress_entry.get("reasoning", ""),
                    "fallback": compress_entry.get("fallback", False),
                })

        if not compressed:
            state.terminate_reason = "budget_exhausted"
            return {**gs, "should_finish": True, "should_abort": False,
                    "agent_state": state, "trace": trace}
        gs = {**gs, "trace": trace}

    step = gs.get("step", 0)
    raw_debug_cb = gs.get("debug_callback")

    def _debug_cb_with_step(data: dict) -> None:
        if raw_debug_cb:
            raw_debug_cb({**data, "step": step})

    decision = controller.decide_action(
        state,
        debug_callback=_debug_cb_with_step if raw_debug_cb else None,
    )
    if decision.state_update:
        state.apply_state_delta(decision.state_update)

    node_callback = gs.get("node_callback")
    if node_callback:
        node_callback("decide_action", {
            "step": gs.get("step", 0),
            "action": decision.action,
            "args": decision.args,
            "reasoning": decision.reasoning,
            "state_update": _delta_to_dict(decision.state_update),
        })

    if decision.action == "finish":
        state.terminate_reason = "agent_finish"
        return {**gs, "decision": decision, "agent_state": state,
                "should_finish": True, "should_abort": False}
    if decision.action == "abort":
        state.terminate_reason = "agent_abort"
        return {**gs, "decision": decision, "agent_state": state,
                "should_finish": False, "should_abort": True}

    return {**gs, "decision": decision, "agent_state": state,
            "should_finish": False, "should_abort": False}


def _node_run_tool(gs: "GraphState") -> "GraphState":
    """run_tool 节点：执行工具，生成并记录 observation。"""
    controller: LoopController = gs["controller"]
    state: AgentState = gs["agent_state"]
    decision: Decision = gs["decision"]
    step: int = gs.get("step", 0)
    trace: list[dict] = list(gs.get("trace", []))

    obs = controller.run_tool_step(decision, state, step)

    trace.append({
        "step": step,
        "action": decision.action,
        "args": decision.args,
        "observation": obs.summary,
        "new_evidence": obs.new_evidence_count,
    })

    node_callback = gs.get("node_callback")
    if node_callback:
        node_callback("run_tool", {
            "step": step,
            "query": decision.args.get("query", ""),
            "new_evidence_count": obs.new_evidence_count,
            "papers_hit": obs.papers_hit,
            "modalities_hit": obs.modalities_hit,
            "budget_remaining": state.evidence_budget_remaining,
            "total_evidence": len(state.evidence),
            "cap": state.evidence_cap,
            "new_items": [_serialize_new_item(rid, controller._store) for rid in obs.new_ids],
        })

    return {**gs, "trace": trace, "step": step + 1, "agent_state": state}


def _node_system_select_or_compact(gs: "GraphState") -> "GraphState":
    """system_select_or_compact 节点：Phase 2 触发候选池阈值压缩；Phase 1 跳过。"""
    # Phase 1：此节点仅作为占位
    return gs


def _node_check_termination(gs: "GraphState") -> "GraphState":
    """check_termination 节点：调用 TerminationPolicy 判断是否终止。"""
    controller: LoopController = gs["controller"]
    state: AgentState = gs["agent_state"]

    should_term, term_type, reason = controller._termination.should_terminate(state)

    if should_term:
        state.terminate_reason = reason
        should_finish = term_type == "finish"
        should_abort = term_type == "abort"
    else:
        should_finish = False
        should_abort = False

    return {**gs, "should_finish": should_finish, "should_abort": should_abort, "agent_state": state}


def _node_answer_or_abort(gs: "GraphState") -> "GraphState":
    """answer_or_abort 节点：生成回答或返回 insufficiency。"""
    controller: LoopController = gs["controller"]
    state: AgentState = gs["agent_state"]
    generate_answer: bool = gs.get("generate_answer", True)
    stream_callback = gs.get("stream_callback")
    answer_language: str = gs.get("answer_language", "English")
    should_abort: bool = gs.get("should_abort", False)
    trace: list[dict] = list(gs.get("trace", []))

    node_callback = gs.get("node_callback")
    debug_callback = gs.get("debug_callback")
    if node_callback:
        node_callback("answer_start", {
            "terminate_reason": state.terminate_reason,
            "selected_evidence_count": len(state.evidence),
            "warnings": [
                {"code": w.code, "message": w.message, "step": w.step}
                for w in state.warnings
            ],
        })

    if should_abort or not generate_answer:
        result = controller._build_final_result(
            state,
            generate_answer=False,
            answer_language=answer_language,
            stream_callback=stream_callback,
            trace=trace,
        )
    else:
        result = controller._generate_and_validate(
            state,
            answer_language=answer_language,
            stream_callback=stream_callback,
            trace=trace,
            debug_callback=debug_callback,
        )

    return {**gs, "final_result": result}


# ── Route Functions ───────────────────────────────────────────────────────────

def _route_after_decide(gs: "GraphState") -> str:
    """decide_action 后的路由：只读 gs，不修改状态。终止信号由节点写入。"""
    if gs.get("should_finish") or gs.get("should_abort"):
        return "answer_or_abort"
    decision: Decision | None = gs.get("decision")
    if decision is None or decision.action not in ("search_evidence", "expand_evidence"):
        return "answer_or_abort"
    return "run_tool"


def _route_after_termination(gs: "GraphState") -> str:
    """check_termination 后的路由。"""
    if gs.get("should_finish") or gs.get("should_abort"):
        return "answer_or_abort"
    return "decide_action"


# ── Graph Builder ─────────────────────────────────────────────────────────────

def _build_graph():
    """构建 LangGraph state graph。"""
    g = StateGraph(GraphState)

    g.add_node("decide_action", _node_decide_action)
    g.add_node("run_tool", _node_run_tool)
    g.add_node("system_select_or_compact", _node_system_select_or_compact)
    g.add_node("check_termination", _node_check_termination)
    g.add_node("answer_or_abort", _node_answer_or_abort)

    g.set_entry_point("decide_action")

    g.add_conditional_edges(
        "decide_action",
        _route_after_decide,
        {
            "run_tool": "run_tool",
            "answer_or_abort": "answer_or_abort",
        },
    )
    g.add_edge("run_tool", "system_select_or_compact")
    g.add_edge("system_select_or_compact", "check_termination")
    g.add_conditional_edges(
        "check_termination",
        _route_after_termination,
        {
            "decide_action": "decide_action",
            "answer_or_abort": "answer_or_abort",
        },
    )
    g.add_edge("answer_or_abort", END)

    return g.compile()
