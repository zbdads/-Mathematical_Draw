"""Prompt 构建模块。

设计原则（Grounded Generation）：
    - System 层明确约束：只基于检索证据回答，禁止扩写。
    - 每条证据标注 chunk_id、paper_id 和页码，便于 LLM 生成可定位的 citation。
    - 注入 valid_keys 列表，约束 LLM 只能引用已提供的证据。
    - 证据不足时要求模型输出固定拒答语，减少幻觉。
    - 固定输出格式（Answer / Citations），便于下游解析。

参考：
    PaperQA2 — citation key 系统、Valid Keys 约束、科学写作风格
    DocsGPT  — token 预算参数化
"""

from __future__ import annotations

from typing import Any

from src.generator.answer_prompt_selector import render_system_prompt


def build_messages(
    question: str,
    evidence: list[dict[str, Any]],
    answer_language: str = "English",
) -> list[dict[str, str]]:
    """根据问题和证据构造 LLM messages 列表（OpenAI-compatible）。

    Args:
        question: 用户问题。
        evidence: TextRetriever.retrieve() 返回的 chunk 列表，每条含
                  ``paper_id``、``page``、``content``、``chunk_id`` 字段。
        answer_language: system prompt 指定的回答语言。

    Returns:
        messages 列表，可直接传入 ``client.chat.completions.create``。
    """
    evidence_blocks: list[str] = []
    included_chunk_ids: list[str] = []

    for chunk in evidence:
        paper_id = chunk.get("paper_id", "unknown")
        raw_page = chunk.get("page", 0)
        page_num = raw_page[0] if isinstance(raw_page, list) and raw_page else raw_page
        if page_num in ([], "", None):
            page_num = "N/A"
        chunk_id = chunk.get("chunk_id", "?")
        modality = chunk.get("modality", "text")
        content = chunk.get("content", "").strip()
        if modality == "figure":
            content = (
                chunk.get("caption_merged")
                or chunk.get("content")
                or chunk.get("caption")
                or ""
            ).strip()
            image_path = chunk.get("image_path", "")
            if image_path:
                content = f"Visual evidence caption/description:\n{content}\nImage file: {image_path}"
        score = chunk.get("rerank_score") or chunk.get("score", 0.0)
        block = (
            f"[{chunk_id}] ({paper_id}, page {page_num}, "
            f"modality={modality}, score={score:.4f})\n{content}"
        )
        evidence_blocks.append(block)
        included_chunk_ids.append(chunk_id)

    valid_keys = ", ".join(f"[{cid}]" for cid in included_chunk_ids)
    system_content = render_system_prompt(
        question_type="unknown",
        valid_keys=valid_keys,
        answer_language=answer_language,
    )

    if evidence_blocks:
        evidence_section = "\n\n".join(evidence_blocks)
    else:
        evidence_section = "(no retrieved evidence)"

    user_content = (
        f"Question: {question}\n\n"
        f"Retrieved evidence:\n{evidence_section}"
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_prompt(question: str, evidence: list[dict[str, Any]]) -> str:
    """根据问题和证据构造 Prompt 字符串（system + user 合并，供调试用）。

    生产调用请使用 :func:`build_messages`。
    """
    messages = build_messages(question, evidence)
    parts = [f"[{m['role'].upper()}]\n{m['content']}" for m in messages]
    return "\n\n---\n\n".join(parts)
