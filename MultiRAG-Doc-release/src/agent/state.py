"""Agent State：loop 期间所有状态的中心定义。

职责分层：
- AgentState 持有面向 controller/LLM 的工作视图（证据列表）。
- EvidenceStore（evidence_store.py）持有 canonical 原始证据全文与底层元数据。
- 二者通过 record_id / chunk_id 关联，AgentState 不重复存完整正文。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ── Warning ──────────────────────────────────────────────────────────────────

@dataclass
class AgentWarning:
    """结构化告警（低证据提前结束、citation 补救失败等）。"""

    code: str        # 全大写机器可读枚举，如 LOW_EVIDENCE_EARLY_FINISH
    message: str     # 面向用户的技术提示
    step: int        # 触发该 warning 的 loop step 编号


# ── Tool History ──────────────────────────────────────────────────────────────

@dataclass
class ToolEvent:
    """单次 tool 调用记录。"""

    step: int
    tool_name: str
    args: dict[str, Any]
    success: bool
    error: str = ""


# ── Step Observation ──────────────────────────────────────────────────────────

@dataclass
class StepObservation:
    """每步 tool 执行后生成的结构化观测。

    summary 与 conflicts 由代码侧规则生成，不引入额外 LLM summarizer。
    conflicts 第一版默认为空列表，作为可选 debug hint，不参与核心终止决策。
    """

    step: int
    new_evidence_count: int
    new_ids: list[str]
    papers_hit: list[str]
    modalities_hit: list[str]
    coverage_delta: int        # 本轮减少的 open_questions 数量（整数计数）
    conflicts: list[str]       # 规则可检出的潜在冲突（第一版默认为空）
    summary: str               # 规则拼装的模板摘要（不含自由文本）


# ── Decision Schema ───────────────────────────────────────────────────────────

@dataclass
class StateDelta:
    """agent_policy.decide() 允许更新的问题表述字段。

    不允许越权修改 evidence、tool_history、no_progress_steps、
    terminate_reason 等系统控制字段。
    """

    normalized_question: str | None = None
    current_objective: str | None = None
    open_questions: list[str] | None = None
    hypotheses: list[str] | None = None


@dataclass
class Decision:
    """agent policy 的结构化输出。

    action:
        "search_evidence"  -> 执行检索
        "expand_evidence"  -> 邻域扩展（Phase 1 仅系统内部 citation recovery 使用）
        "finish"           -> route signal：进入回答阶段
        "abort"            -> route signal：证据不足，终止
    args:
        当前 action 的受控参数（如 search_evidence 的 query / modalities）
    state_update:
        可选，用于结构化更新问题表述字段
    """

    action: Literal["search_evidence", "expand_evidence", "finish", "abort"]
    args: dict[str, Any] = field(default_factory=dict)
    state_update: StateDelta | None = None
    reasoning: str = ""  # LLM 决策理由


# ── Evidence Views ────────────────────────────────────────────────────────────

@dataclass
class SelectedView:
    """AgentState 持有的证据视图（引用 EvidenceStore 中的 record_id）。"""

    record_id: str
    chunk_id: str
    paper_id: str
    page: int | list[int] | None
    modality: str
    content: str              # 回答阶段使用全文
    final_score: float
    llm_relevance_score: int  # 固定 -1（未打分）
    source_query: str
    matched_sub_queries: list[str] = field(default_factory=list)
    figure_id: str | None = None
    image_path: str | None = None


# ── Agent State ───────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    """Loop 工作状态，面向 controller 与 LLM。

    不持有证据原文（全文存于 EvidenceStore），只持有工作视图。
    """

    # 问题表述
    original_question: str
    normalized_question: str
    current_objective: str
    open_questions: list[str]
    hypotheses: list[str] = field(default_factory=list)

    # 工具历史
    tool_history: list[ToolEvent] = field(default_factory=list)

    # 证据工作视图（全量，按 retrieval_score 降序）
    evidence: list[SelectedView] = field(default_factory=list)

    # Evidence budget（由 LoopController 初始化后注入）
    evidence_cap: int = 8

    # Loop 控制
    repeated_queries: set[str] = field(default_factory=set)
    no_progress_steps: int = 0
    citation_recovery_used: bool = False
    compression_used: bool = False          # 每次 agent loop 最多触发 1 次 evidence 压缩
    answer_ready: bool = False
    terminate_reason: str = ""

    # 观测历史（LLM 可见：最近 2 轮 + rolling_summary）
    observation_history: list[StepObservation] = field(default_factory=list)
    rolling_summary: str = ""

    # 告警
    warnings: list[AgentWarning] = field(default_factory=list)

    # 用户指定的 paper_id（最高优先级，agent 不可覆盖）
    user_paper_id: str | None = None

    # planner 解析出的问题类型（agent policy 不允许覆盖）
    question_type: str = "unknown"

    @property
    def evidence_budget_remaining(self) -> int:
        text_count = sum(1 for v in self.evidence if v.modality != "figure")
        return max(0, self.evidence_cap - text_count)

    @classmethod
    def from_question(
        cls,
        question: str,
        paper_id: str | None = None,
    ) -> "AgentState":
        """初始化 AgentState。

        normalized_question 与 current_objective 默认等于 original_question；
        open_questions 默认为 [normalized_question]，不能以空列表启动。
        """
        return cls(
            original_question=question,
            normalized_question=question,
            current_objective=question,
            open_questions=[question],
            user_paper_id=paper_id,
        )

    def apply_state_delta(self, delta: StateDelta) -> None:
        """应用 agent 的结构化 state_update（只允许更新问题表述字段）。"""
        if delta.normalized_question is not None:
            self.normalized_question = delta.normalized_question
        if delta.current_objective is not None:
            self.current_objective = delta.current_objective
        if delta.open_questions is not None:
            if delta.open_questions:  # 不允许清空 open_questions
                self.open_questions = delta.open_questions
        if delta.hypotheses is not None:
            self.hypotheses = delta.hypotheses

    def record_tool_event(
        self,
        step: int,
        tool_name: str,
        args: dict[str, Any],
        success: bool,
        error: str = "",
    ) -> None:
        self.tool_history.append(
            ToolEvent(step=step, tool_name=tool_name, args=args, success=success, error=error)
        )

    def add_warning(self, code: str, message: str, step: int) -> None:
        self.warnings.append(AgentWarning(code=code, message=message, step=step))

    def get_recent_observations(self, n: int = 2) -> list[StepObservation]:
        """返回最近 n 轮 StepObservation（供 LLM 上下文压缩使用）。"""
        return self.observation_history[-n:] if self.observation_history else []

    def current_step(self) -> int:
        return len(self.observation_history)
