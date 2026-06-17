"""Tool Registry：将 agent 的 function call 映射到仓库内可复用能力。

Phase 1 只注册 search_evidence，后端为 QueryService.retrieve_core()。
expand_evidence 与 select_evidence 是系统内部动作，不注册给 agent。
finish / abort 是 route signal，不注册为业务 tool。

不直接暴露 TextRetriever / ImageRetriever 等底层接口给 agent。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.query.service import QueryService
    from src.agent.state import AgentState

from src.agent.evidence_store import EvidenceStore
from src.agent.tool_schemas import SearchEvidenceArgs
from src.config import CFG
from src.query.retrieval_orchestrator import EvidenceCandidate, hit_to_candidate


# ── search_evidence ───────────────────────────────────────────────────────────

def execute_search_evidence(
    args: SearchEvidenceArgs,
    state: "AgentState",
    service: "QueryService",
    store: EvidenceStore,
    evidence_cap: int,
) -> tuple[list[str], list[EvidenceCandidate]]:
    """执行 search_evidence：单 query 检索，写入 EvidenceStore。

    top_k = min(args.expected_evidence_count, remaining_budget)，启用 reranker。
    budget 耗尽时直接返回空列表。

    Returns:
        (new_record_ids, candidates)
    """
    args.validate()

    remaining_budget = evidence_cap - store.count_text()
    effective_top_k = min(args.expected_evidence_count, remaining_budget)
    if effective_top_k <= 0:
        return [], []

    # paper_id 优先级：用户指定 > agent hint
    paper_id = state.user_paper_id if state.user_paper_id else args.paper_id_hint

    # 确定 figure_top_k
    use_figure = args.modalities in ("figure", "text+figure")
    figure_top_k = CFG.retriever.top_k_fig if use_figure else 0

    hits_by_track = service.retrieve_core(
        question=args.query,
        top_k=effective_top_k,
        paper_id=paper_id,
        skip_rerank=False,          # 启用 reranker，由 agent 的 expected_evidence_count 控制数量
        figure_top_k=figure_top_k,
    )

    candidates: list[EvidenceCandidate] = []

    # text 赛道
    if args.modalities in ("text", "text+figure"):
        for hit in hits_by_track.get("text_results", []):
            candidates.append(hit_to_candidate(hit, args.query))

    # figure 赛道
    if use_figure:
        for hit in hits_by_track.get("figure_results", []):
            candidates.append(hit_to_candidate(hit, args.query))

    new_ids = store.add_candidates(candidates, source_query=args.query)
    return new_ids, candidates


# ── Tool Registry ─────────────────────────────────────────────────────────────

class ToolRegistry:
    """管理 agent 可调用工具与系统内部动作的注册与执行。"""

    def __init__(
        self,
        service: "QueryService",
        store: EvidenceStore,
        evidence_cap: int,
    ) -> None:
        self._service = service
        self._store = store
        self._evidence_cap = evidence_cap

    def execute(
        self,
        tool_name: str,
        args: dict[str, Any],
        state: "AgentState",
    ) -> tuple[list[str], list[EvidenceCandidate]]:
        """执行 tool_name 指定的工具。

        Args:
            tool_name: 工具名称（当前只有 "search_evidence"）。
            args: 工具参数 dict（由 LLM 或 controller 提供）。
            state: 当前 AgentState（用于 paper_id 决策等）。

        Returns:
            (new_record_ids, candidates)

        Raises:
            ValueError: tool_name 未注册或参数无效。
        """
        if tool_name == "search_evidence":
            parsed = SearchEvidenceArgs.from_dict(args)
            return execute_search_evidence(
                parsed, state, self._service, self._store, self._evidence_cap
            )
        raise ValueError(f"[ToolRegistry] 未注册的工具：{tool_name!r}")

    def registered_tools(self) -> list[str]:
        """返回当前 agent 可调用的工具名称列表（Phase 1：仅 search_evidence）。"""
        return ["search_evidence"]
