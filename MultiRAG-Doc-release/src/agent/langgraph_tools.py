"""LangGraph Tools：将仓库高层工具封装为 LangChain tool schema。

只封装 agent 可调用的高层工具（Phase 1：仅 search_evidence）。
finish / abort 是 route signal，不注册为业务 tool。
select_evidence / expand_evidence 是系统内部动作，不注册。

LangChain tool 只包高层动作，不暴露底层 TextRetriever / ImageRetriever 细节。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# LangGraph / LangChain 依赖检查
try:
    from langchain_core.tools import tool as lc_tool
    from langchain_core.messages import ToolMessage
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

if TYPE_CHECKING:
    from src.agent.state import AgentState
    from src.agent.tool_registry import ToolRegistry
    from src.agent.evidence_store import EvidenceStore

from src.agent.tool_schemas import ALLOWED_MODALITIES, DEFAULT_MODALITY


# ── LangChain Tool Schema ──────────────────────────────────────────────────────

def make_search_evidence_tool(
    registry: "ToolRegistry",
    store: "EvidenceStore",
    state: "AgentState",
):
    """动态创建 search_evidence LangChain tool，绑定 registry / store / state。

    返回 LangChain @tool 装饰的函数（若 langchain_core 不可用则返回 None）。
    """
    if not _LANGCHAIN_AVAILABLE:
        return None

    @lc_tool
    def search_evidence(
        query: str,
        modalities: str = DEFAULT_MODALITY,
        paper_id_hint: str = "",
    ) -> str:
        """Search for evidence in the paper database.

        Args:
            query: Search query in English. Be specific and different from prior queries.
            modalities: One of 'text', 'figure', 'text+figure'. Default: 'text+figure'.
            paper_id_hint: Optional paper ID to narrow search scope.

        Returns:
            JSON string with search results summary.
        """
        import json

        args = {
            "query": query,
            "modalities": modalities if modalities in ALLOWED_MODALITIES else DEFAULT_MODALITY,
            "paper_id_hint": paper_id_hint or None,
        }
        new_ids, candidates = registry.execute("search_evidence", args, state)
        return json.dumps({
            "new_evidence_count": len(new_ids),
            "total_candidates": store.count(),
            "papers_hit": list({c.paper_id for c in candidates}),
            "modalities_hit": list({c.modality for c in candidates}),
        })

    return search_evidence


def get_tool_definitions() -> list[dict[str, Any]]:
    """返回 Phase 1 注册给 agent 的工具定义（OpenAI function calling 格式）。

    不依赖 LangChain，可直接用于 llm_client 的 tools 参数。
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "search_evidence",
                "description": (
                    "Search for evidence in the scientific paper database. "
                    "Call this to retrieve relevant text chunks or figures."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query in English. Be specific.",
                        },
                        "modalities": {
                            "type": "string",
                            "enum": list(ALLOWED_MODALITIES),
                            "description": "Which modalities to search. Default: text+figure.",
                        },
                        "paper_id_hint": {
                            "type": "string",
                            "description": "Optional: narrow search to a specific paper ID.",
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    ]
