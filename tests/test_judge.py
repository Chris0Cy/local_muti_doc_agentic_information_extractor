from __future__ import annotations

from pathlib import Path

import pytest

from extractor.judge import synthesize
from extractor.lmstudio_client import LMStudioRequestError
from extractor.models import TierConfig, WorkerResult

JUDGE_TIER = TierConfig(
    name="judge", model_id="judge-model", context_window_tokens=8192, reserved_output_tokens=512
)


class FakeClient:
    def __init__(self, response: str | None = None, raise_error: Exception | None = None):
        self.response = response
        self.raise_error = raise_error

    async def health_check(self) -> None:
        pass

    async def list_models(self) -> list[str]:
        return [JUDGE_TIER.model_id]

    async def chat_completion(self, model, messages, *, temperature=0.0, max_tokens=None) -> str:
        if self.raise_error:
            raise self.raise_error
        return self.response


def make_result(doc_name: str, found: bool, excerpt: str | None) -> WorkerResult:
    return WorkerResult(
        doc_path=Path(doc_name),
        chunk_index=0,
        total_chunks=1,
        tier_used="tiny",
        found_relevant_info=found,
        answer_excerpt=excerpt,
        confidence="high" if found else None,
    )


@pytest.mark.asyncio
async def test_synthesize_returns_unanswered_shortcut_when_nothing_relevant():
    client = FakeClient(response="should not be used")
    results = [make_result("a.txt", False, None), make_result("b.txt", False, None)]
    result = await synthesize(client, "What is X?", results, JUDGE_TIER)
    assert result.unanswered is True
    assert (
        "could not" in result.final_answer_markdown.lower()
        or "no relevant" in result.final_answer_markdown.lower()
    )


@pytest.mark.asyncio
async def test_synthesize_calls_judge_model_when_relevant_findings_exist():
    client = FakeClient(response="**Answer**: X is 42 (source: a.txt)")
    results = [make_result("a.txt", True, "X is 42"), make_result("b.txt", False, None)]
    result = await synthesize(client, "What is X?", results, JUDGE_TIER)
    assert result.unanswered is False
    assert "42" in result.final_answer_markdown


@pytest.mark.asyncio
async def test_synthesize_handles_judge_request_error_gracefully():
    client = FakeClient(raise_error=LMStudioRequestError(JUDGE_TIER.model_id, "not loaded"))
    results = [make_result("a.txt", True, "X is 42")]
    result = await synthesize(client, "What is X?", results, JUDGE_TIER)
    assert result.unanswered is True
    assert "failed" in result.final_answer_markdown.lower()
