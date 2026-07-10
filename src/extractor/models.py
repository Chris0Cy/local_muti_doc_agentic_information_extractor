"""Core data models shared across the pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TierConfig(BaseModel):
    name: str
    model_id: str
    context_window_tokens: int
    reserved_output_tokens: int = 512

    @model_validator(mode="after")
    def _check_positive_budget(self) -> TierConfig:
        if self.reserved_output_tokens >= self.context_window_tokens:
            raise ValueError(
                f"Tier '{self.name}': reserved_output_tokens ({self.reserved_output_tokens}) "
                f"must be less than context_window_tokens ({self.context_window_tokens}), "
                "otherwise no budget is left for input text."
            )
        return self


class LMStudioConfig(BaseModel):
    base_url: str = "http://localhost:1234/v1"
    request_timeout_s: float = 120.0


class RelevanceFilterConfig(BaseModel):
    """Opt-in embedding-based pre-filter: only present in AppConfig when enabled."""

    embedding_model: TierConfig
    top_k: int | None = 10
    similarity_floor: float | None = 0.35
    overlap_tokens: int = 100
    cache_path: str = "./.cache/embeddings.json"

    @model_validator(mode="after")
    def _check_thresholds(self) -> RelevanceFilterConfig:
        if self.top_k is None and self.similarity_floor is None:
            raise ValueError(
                "relevance_filter needs at least one of top_k or similarity_floor set, "
                "otherwise it wouldn't filter anything."
            )
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("relevance_filter.top_k must be >= 1")
        return self


class AppConfig(BaseModel):
    lmstudio: LMStudioConfig = Field(default_factory=LMStudioConfig)
    tiers: list[TierConfig]
    judge: TierConfig
    concurrency_limit: int = 4
    chunk_overlap_tokens: int = 200
    safety_margin: float = 0.85
    output_format: Literal["text", "markdown", "json"] = "markdown"
    save_path: str | None = None
    relevance_filter: RelevanceFilterConfig | None = None

    @model_validator(mode="after")
    def _check_at_least_one_tier(self) -> AppConfig:
        if not self.tiers:
            raise ValueError("AppConfig.tiers must contain at least one tier.")
        return self

    @property
    def tiers_ascending(self) -> list[TierConfig]:
        return sorted(self.tiers, key=lambda t: t.context_window_tokens)


class ExtractedDocument(BaseModel):
    path: Path
    text: str
    extraction_method: str
    warnings: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    doc_path: Path
    chunk_index: int
    total_chunks: int
    text: str
    tier: TierConfig


class WorkerResult(BaseModel):
    doc_path: Path
    chunk_index: int
    total_chunks: int
    tier_used: str
    found_relevant_info: bool
    answer_excerpt: str | None = None
    confidence: Literal["high", "medium", "low"] | None = None
    error: str | None = None
    raw_response: str | None = None


class JudgeResult(BaseModel):
    final_answer_markdown: str
    unanswered: bool
    raw_response: str


class DocumentRelevanceScore(BaseModel):
    path: Path
    score: float
    kept: bool


class RelevanceFilterResult(BaseModel):
    enabled: bool = False
    scores: list[DocumentRelevanceScore] = Field(default_factory=list)

    @property
    def dropped(self) -> list[DocumentRelevanceScore]:
        return [s for s in self.scores if not s.kept]


class RunReport(BaseModel):
    question: str
    documents_scanned: int
    documents_failed: list[tuple[Path, str]]
    worker_results: list[WorkerResult]
    judge_result: JudgeResult | None
    duration_s: float
    relevance_filter: RelevanceFilterResult = Field(default_factory=RelevanceFilterResult)
