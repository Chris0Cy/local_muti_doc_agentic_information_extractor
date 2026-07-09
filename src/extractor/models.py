"""Core data models shared across the pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class TierConfig(BaseModel):
    name: str
    model_id: str
    context_window_tokens: int
    reserved_output_tokens: int = 512


class LMStudioConfig(BaseModel):
    base_url: str = "http://localhost:1234/v1"
    request_timeout_s: float = 120.0


class AppConfig(BaseModel):
    lmstudio: LMStudioConfig = Field(default_factory=LMStudioConfig)
    tiers: list[TierConfig]
    judge: TierConfig
    concurrency_limit: int = 4
    chunk_overlap_tokens: int = 200
    safety_margin: float = 0.85
    output_format: Literal["text", "markdown", "json"] = "markdown"
    save_path: str | None = None

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


class RunReport(BaseModel):
    question: str
    documents_scanned: int
    documents_failed: list[tuple[Path, str]]
    worker_results: list[WorkerResult]
    judge_result: JudgeResult | None
    duration_s: float
