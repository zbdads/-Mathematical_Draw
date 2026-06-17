"""Evidence Compressor：budget 耗尽时调用 LLM 裁剪低质量证据。

职责：
- 接收 EvidenceStore 与原始问题，让 LLM 自主决定丢弃哪些 text chunk。
- 解析 LLM 输出，校验 chunk_id 合法性，clamp 保留数量 ≥ 3。
- LLM 调用失败时 fallback 到按 retrieval_score 裁剪底部 40%。
- 返回 (discard_ids, reasoning)，不直接修改 store（由调用方执行删除）。
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent.evidence_store import EvidenceStore

from src.generator.llm_client import generate

# ── 常量 ─────────────────────────────────────────────────────────────────────

_MIN_KEEP = 3                    # 最少保留的 text chunk 数量（安全下限）
_FALLBACK_REASONING = "fallback: score-based pruning (LLM parse failed)"

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an evidence pruning assistant for a scientific paper Q&A system.
Your task: given a research question and a pool of retrieved text chunks,
decide which chunks are NOT worth keeping — they are redundant, off-topic,
or too vague to contribute to answering the question.

Output ONLY a valid JSON object:
{
  "discard": ["<chunk_id>", ...],
  "reasoning": "<1-2 sentences explaining your pruning decision>"
}

Guidelines:
- You decide how many to discard. Discard as few or as many as needed.
- You MAY output an empty "discard" list if all chunks are genuinely relevant.
- Prefer to KEEP: chunks directly addressing the core question,
  chunks with specific methods/results/numbers, chunks from diverse papers.
- Prefer to DISCARD: off-topic chunks, near-duplicate chunks covering the
  same sub-point, chunks with only generic background content.
- Output chunk_ids exactly as shown in the input. Do not invent chunk_ids.
"""


# ── EvidenceCompressor ────────────────────────────────────────────────────────

class EvidenceCompressor:
    """调用 LLM 压缩 EvidenceStore，丢弃低相关性 chunk。"""

    def compress(
        self,
        original_question: str,
        store: "EvidenceStore",
        model: str,
    ) -> tuple[list[str], str]:
        """让 LLM 自主决定丢弃哪些 chunk，返回 (discard_ids, reasoning)。

        保留数量由代码侧 clamp 到 ≥ _MIN_KEEP，不写入 prompt。
        LLM 调用或解析失败时 fallback 到按 retrieval_score 裁剪底部 40%。
        """
        text_records = [r for r in store.all_records() if r.modality != "figure"]
        if len(text_records) <= _MIN_KEEP:
            return [], "nothing to discard: at or below minimum keep threshold"

        valid_ids = {r.chunk_id for r in text_records}

        try:
            discard_ids, reasoning = self._llm_compress(
                original_question, text_records, model
            )
            # 校验 chunk_id 合法性
            discard_ids = [cid for cid in discard_ids if cid in valid_ids]
            # clamp：保证保留数量 ≥ _MIN_KEEP
            max_discard = len(text_records) - _MIN_KEEP
            if len(discard_ids) > max_discard:
                discard_ids = discard_ids[:max_discard]
            return discard_ids, reasoning
        except Exception:
            return self._fallback_compress(text_records)

    # ── LLM 压缩 ─────────────────────────────────────────────────────────────

    def _llm_compress(
        self,
        original_question: str,
        text_records: list,
        model: str,
    ) -> tuple[list[str], str]:
        user_msg = self._build_user_message(original_question, text_records)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        raw = generate(
            messages=messages,
            model=model,
            temperature=0.0,
            max_tokens=600,
        )
        return self._parse_llm_output(raw)

    def _build_user_message(self, original_question: str, text_records: list) -> str:
        lines = [
            f"Research question: {original_question}",
            "",
            f"Review the following {len(text_records)} evidence chunks and decide which to discard:",
            "",
        ]
        for rec in text_records:
            preview = rec.content[:200].replace("\n", " ")
            if len(rec.content) > 200:
                preview += "..."
            lines.append(
                f"[chunk_id={rec.chunk_id}] paper={rec.paper_id} "
                f"page={rec.page} score={rec.retrieval_score:.4f}"
            )
            lines.append(preview)
            lines.append("---")
        return "\n".join(lines)

    def _parse_llm_output(self, raw: str) -> tuple[list[str], str]:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            ).strip()
        data = json.loads(text)
        discard = data.get("discard", [])
        if not isinstance(discard, list):
            raise ValueError("discard field is not a list")
        discard_ids = [str(cid) for cid in discard]
        reasoning = str(data.get("reasoning", ""))
        return discard_ids, reasoning

    # ── Fallback：score-based 裁底部 40% ─────────────────────────────────────

    def _fallback_compress(
        self, text_records: list
    ) -> tuple[list[str], str]:
        sorted_recs = sorted(text_records, key=lambda r: r.retrieval_score)
        n = len(sorted_recs)
        n_discard = math.floor(n * 0.4)
        # 保证保留数量 ≥ _MIN_KEEP
        max_discard = n - _MIN_KEEP
        n_discard = min(n_discard, max_discard)
        if n_discard <= 0:
            return [], _FALLBACK_REASONING
        discard_ids = [r.chunk_id for r in sorted_recs[:n_discard]]
        return discard_ids, _FALLBACK_REASONING
