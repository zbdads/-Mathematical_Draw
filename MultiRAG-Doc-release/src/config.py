"""全局配置单例。从项目根目录的 config.yml 读取配置，所有模块通过 `from src.config import CFG` 获取配置。"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any

import yaml

# 项目根目录（src/ 的父目录）
_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_FILE = _ROOT / "config.yml"


def _load_yaml() -> dict[str, Any]:
    with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class _Paths:
    root: Path
    pdf_dir: Path
    chunks_dir: Path
    index_dir: Path
    models_dir: Path
    docling_models_dir: Path | None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "_Paths":
        raw = d.get("docling_models_dir", "")
        return _Paths(
            root=_ROOT,
            pdf_dir=_ROOT / d["pdf_dir"],
            chunks_dir=_ROOT / d["chunks_dir"],
            index_dir=_ROOT / d["index_dir"],
            models_dir=_ROOT / d["models_dir"],
            docling_models_dir=_ROOT / raw if raw else None,
        )


@dataclass(frozen=True)
class _Embedder:
    mode: str
    text_model_name: str
    batch_size: int
    embedding_dim: int
    image_mode: str
    image_model_name: str
    image_embedding_dim: int
    image_api_url: str
    image_api_key_env: str
    api_model_name: str
    api_base_url: str
    api_key_env: str
    api_dimensions: int | None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "_Embedder":
        raw_api_dimensions = d.get("api_dimensions", None)
        return _Embedder(
            mode=d.get("mode", "local"),
            text_model_name=d["text_model_name"],
            batch_size=int(d["batch_size"]),
            embedding_dim=int(d["embedding_dim"]),
            image_mode=d.get("image_mode", d.get("mode", "local")),
            image_model_name=d.get("image_model_name", ""),
            image_embedding_dim=int(d.get("image_embedding_dim", 4096)),
            image_api_url=d.get("image_api_url", ""),
            image_api_key_env=d.get("image_api_key_env", d.get("api_key_env", "EMBEDDING_API_KEY")),
            api_model_name=d.get("api_model_name", ""),
            api_base_url=d.get("api_base_url", ""),
            api_key_env=d.get("api_key_env", "EMBEDDING_API_KEY"),
            api_dimensions=int(raw_api_dimensions) if raw_api_dimensions is not None else None,
        )


@dataclass(frozen=True)
class _Chunker:
    chunk_size: int
    overlap_sentences: int
    min_chunk_size: int
    prepend_header: bool

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "_Chunker":
        return _Chunker(
            chunk_size=int(d["chunk_size"]),
            overlap_sentences=int(d["overlap_sentences"]),
            min_chunk_size=int(d.get("min_chunk_size", 100)),
            prepend_header=bool(d.get("prepend_header", True)),
        )


@dataclass(frozen=True)
class _Retriever:
    top_k: int
    top_k_fig: int
    figure_alpha: float
    figure_beta: float
    multi_paper_balanced: bool
    multi_paper_per_doc_k: int

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "_Retriever":
        return _Retriever(
            top_k=int(d["top_k"]),
            top_k_fig=int(d.get("top_k_fig", 2)),
            figure_alpha=float(d.get("figure_alpha", 0.8)),
            figure_beta=float(d.get("figure_beta", 0.5)),
            multi_paper_balanced=bool(d.get("multi_paper_balanced", False)),
            multi_paper_per_doc_k=int(d.get("multi_paper_per_doc_k", 3)),
        )


@dataclass(frozen=True)
class _Generator:
    model_name: str
    base_url: str
    wire_api: str
    api_key_env: str
    max_new_tokens: int
    temperature: float
    token_budget: int
    answer_language: str
    reasoning_effort: str
    disable_response_storage: bool
    timeout_seconds: float

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "_Generator":
        return _Generator(
            model_name=d.get("model_name", ""),
            base_url=d.get("base_url", "https://api.deepseek.com"),
            wire_api=d.get("wire_api", "chat"),
            api_key_env=d.get("api_key_env", "LLM_API_KEY"),
            max_new_tokens=int(d.get("max_new_tokens", 1024)),
            temperature=float(d.get("temperature", 0.0)),
            token_budget=int(d.get("token_budget", 4000)),
            answer_language=d.get("answer_language", "English"),
            reasoning_effort=d.get("reasoning_effort", ""),
            disable_response_storage=bool(d.get("disable_response_storage", False)),
            timeout_seconds=float(d.get("timeout_seconds", 180.0)),
        )


@dataclass(frozen=True)
class _Reranker:
    enabled: bool
    model_name: str
    candidate_k: int
    decompose_top_k: int

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "_Reranker":
        return _Reranker(
            enabled=bool(d.get("enabled", False)),
            model_name=d.get("model_name", "BAAI/bge-reranker-base"),
            candidate_k=int(d.get("candidate_k", 10)),
            decompose_top_k=int(d.get("decompose_top_k", 10)),
        )


@dataclass(frozen=True)
class _Guardrails:
    min_results: int
    min_top1_score: float
    min_total_chars: int

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "_Guardrails":
        return _Guardrails(
            min_results=int(d.get("min_results", 1)),
            min_top1_score=float(d.get("min_top1_score", 0.35)),
            min_total_chars=int(d.get("min_total_chars", 180)),
        )


@dataclass(frozen=True)
class _Agent:
    max_steps: int
    evidence_cap: int

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "_Agent":
        return _Agent(
            max_steps=int(d.get("max_steps", 8)),
            evidence_cap=int(d.get("evidence_cap", 8)),
        )


@dataclass(frozen=True)
class _CaptionModel:
    mode: str
    model_dir: Path | None
    model_name: str
    base_url: str
    api_key_env: str
    adapter_path: Path | None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "_CaptionModel":
        raw = d.get("model_dir", "")
        raw_adapter = d.get("adapter_path", "")
        return _CaptionModel(
            mode=d.get("mode", "local"),
            model_dir=_ROOT / raw if raw else None,
            model_name=d.get("model_name", "Qwen/Qwen3-VL-8B-Instruct"),
            base_url=d.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            api_key_env=d.get("api_key_env", "LLM_API_KEY"),
            adapter_path=_ROOT / raw_adapter if raw_adapter else None,
        )


@dataclass(frozen=True)
class _Config:
    paths: _Paths
    embedder: _Embedder
    chunker: _Chunker
    retriever: _Retriever
    reranker: _Reranker
    generator: _Generator
    guardrails: _Guardrails
    caption_model: _CaptionModel
    agent: _Agent

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "_Config":
        return _Config(
            paths=_Paths.from_dict(d["paths"]),
            embedder=_Embedder.from_dict(d["embedder"]),
            chunker=_Chunker.from_dict(d["chunker"]),
            retriever=_Retriever.from_dict(d["retriever"]),
            reranker=_Reranker.from_dict(d.get("reranker", {})),
            generator=_Generator.from_dict(d["generator"]),
            guardrails=_Guardrails.from_dict(d.get("guardrails", {})),
            caption_model=_CaptionModel.from_dict(d.get("caption_model", {})),
            agent=_Agent.from_dict(d.get("agent", {})),
        )


CFG = _Config.from_dict(_load_yaml())
