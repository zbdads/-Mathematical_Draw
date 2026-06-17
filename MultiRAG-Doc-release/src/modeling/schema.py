"""Data structures for LLMOPT-style optimization model cards."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ModelEvidenceRef:
    """A source chunk used to build or support a model-card field."""

    chunk_id: str
    page: int | list[int]
    modality: str
    role: str
    quote: str = ""


@dataclass
class ModelCard:
    """Five-element representation of an optimization model in a paper."""

    paper_id: str
    title: str = ""
    problem_type: str = ""
    application_domain: str = ""
    model_name: str = ""
    sets: list[str] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)
    objective: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    algorithm: list[str] = field(default_factory=list)
    source_chunk_ids: list[str] = field(default_factory=list)
    evidence_refs: list[ModelEvidenceRef] = field(default_factory=list)
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelCard":
        refs = [
            ModelEvidenceRef(**ref)
            for ref in data.get("evidence_refs", [])
            if isinstance(ref, dict)
        ]
        allowed = {
            "paper_id",
            "title",
            "problem_type",
            "application_domain",
            "model_name",
            "sets",
            "parameters",
            "variables",
            "objective",
            "constraints",
            "assumptions",
            "algorithm",
            "source_chunk_ids",
            "confidence",
            "warnings",
        }
        kwargs = {k: v for k, v in data.items() if k in allowed}
        card = cls(**kwargs)
        card.evidence_refs = refs
        return card


FIVE_ELEMENT_FIELDS = (
    "sets",
    "parameters",
    "variables",
    "objective",
    "constraints",
)
