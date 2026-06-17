"""Termination Policy：loop 退出策略，独立于 prompt 定义。

Phase 1 兜底只依赖：
- 显式 abort（agent 主动触发）
- max_steps 强制终止

Phase 2 才正式接入：
- no_progress_steps 驱动的提前终止
- 重复 query 抑制

停止条件必须在代码中显式定义，不能仅依赖 LLM prompt 内部的自然语言约束。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent.state import AgentState

# ── 默认配置 ─────────────────────────────────────────────────────────────────

_DEFAULT_MAX_STEPS = 8
_DEFAULT_MAX_CITATION_RECOVERY_ROUNDS = 1

# Phase 2 will use these thresholds
_NO_PROGRESS_ABORT_THRESHOLD = 2   # 连续 N 步无进展则终止（Phase 2）


class TerminationPolicy:
    """Loop 退出策略（Phase 1 骨架，Phase 2 扩展）。"""

    def __init__(
        self,
        max_steps: int = _DEFAULT_MAX_STEPS,
        max_citation_recovery_rounds: int = _DEFAULT_MAX_CITATION_RECOVERY_ROUNDS,
        enable_no_progress: bool = False,    # Phase 2 开关
        enable_repeat_query: bool = False,   # Phase 2 开关
    ) -> None:
        self.max_steps = max_steps
        self.max_citation_recovery_rounds = max_citation_recovery_rounds
        self._enable_no_progress = enable_no_progress
        self._enable_repeat_query = enable_repeat_query

    # ── 规则触发 finish ──────────────────────────────────────────────────────

    def should_rule_finish(self, state: "AgentState") -> tuple[bool, str]:
        """判断是否满足规则触发的 finish 条件。

        Returns:
            (should_finish, reason)
        """
        if state.current_step() >= self.max_steps:
            return True, f"max_steps={self.max_steps} reached"
        # Phase 2: no_progress 驱动 finish
        if self._enable_no_progress and state.no_progress_steps >= _NO_PROGRESS_ABORT_THRESHOLD:
            return True, f"no_progress_steps={state.no_progress_steps} >= {_NO_PROGRESS_ABORT_THRESHOLD}"
        return False, ""

    # ── 规则触发 abort ───────────────────────────────────────────────────────

    def should_rule_abort(self, state: "AgentState") -> tuple[bool, str]:
        """判断是否满足规则触发的 abort 条件。

        Returns:
            (should_abort, reason)
        """
        return False, ""

    # ── 合并判断（供 controller 在每步 observation 后调用）──────────────────

    def should_terminate(self, state: "AgentState") -> tuple[bool, str, str]:
        """综合判断是否需要终止，以及终止类型。

        Returns:
            (should_terminate, termination_type, reason)
            termination_type: "finish" | "abort" | ""
        """
        abort, abort_reason = self.should_rule_abort(state)
        if abort:
            return True, "abort", abort_reason

        finish, finish_reason = self.should_rule_finish(state)
        if finish:
            return True, "finish", finish_reason

        return False, "", ""

    # ── Citation Recovery 限制 ───────────────────────────────────────────────

    def can_do_citation_recovery(self, state: "AgentState") -> bool:
        """Phase 1：citation recovery 最多执行 1 轮。"""
        return not state.citation_recovery_used

    # ── 低证据 warning 检测 ─────────────────────────────────────────────────

    def check_low_evidence_warning(self, state: "AgentState") -> list[str]:
        """检查低证据 warning 条件，返回触发的 warning code 列表。

        任一满足即触发 LOW_EVIDENCE_EARLY_FINISH：
        - evidence 不足 3 条
        - 覆盖的唯一 source location 不足 2 个（按 paper_id 计）
        """
        warnings: list[str] = []
        evidence = state.evidence

        if len(evidence) < 3:
            warnings.append("LOW_EVIDENCE_EARLY_FINISH")
            return warnings

        unique_sources = len({v.paper_id for v in evidence})
        if unique_sources < 2:
            warnings.append("LOW_EVIDENCE_EARLY_FINISH")
            return warnings

        return warnings