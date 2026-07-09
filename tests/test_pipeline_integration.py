from __future__ import annotations

from pathlib import Path

import pytest

from extractor import pipeline
from extractor.models import AppConfig, LMStudioConfig, TierConfig

TIER = TierConfig(
    name="tiny", model_id="tiny-model", context_window_tokens=4096, reserved_output_tokens=256
)
JUDGE_TIER = TierConfig(
    name="judge", model_id="judge-model", context_window_tokens=4096, reserved_output_tokens=256
)


def make_config() -> AppConfig:
    return AppConfig(
        lmstudio=LMStudioConfig(),
        tiers=[TIER],
        judge=JUDGE_TIER,
        concurrency_limit=2,
        chunk_overlap_tokens=10,
        safety_margin=0.85,
    )


class FakeClient:
    """Routes by model: judge model synthesizes, worker model inspects the
    document text embedded in the user message to decide relevance."""

    def __init__(self):
        self.health_checked = False

    async def health_check(self) -> None:
        self.health_checked = True

    async def list_models(self) -> list[str]:
        return [TIER.model_id, JUDGE_TIER.model_id]

    async def chat_completion(self, model, messages, *, temperature=0.0, max_tokens=None) -> str:
        user_content = messages[-1]["content"]
        if model == JUDGE_TIER.model_id:
            return "Final answer: revenue grew 12% (source: b.txt)"
        if "12%" in user_content:
            return '{"found_relevant_info": true, "answer_excerpt": "revenue grew 12%", "confidence": "high"}'
        return '{"found_relevant_info": false, "answer_excerpt": null, "confidence": null}'


@pytest.mark.asyncio
async def test_run_processes_all_docs_and_synthesizes_answer(tmp_path: Path):
    (tmp_path / "a.txt").write_text("Nothing relevant here, just filler text.")
    (tmp_path / "b.txt").write_text("Q3 update: revenue grew 12% year over year.")

    client = FakeClient()
    result = await pipeline.run("How did revenue change?", tmp_path, make_config(), client)

    assert client.health_checked is True
    assert result.documents_scanned == 2
    assert result.documents_failed == []
    assert len(result.worker_results) == 2
    assert result.judge_result is not None
    assert result.judge_result.unanswered is False
    assert "12%" in result.judge_result.final_answer_markdown


@pytest.mark.asyncio
async def test_run_continues_when_one_document_fails_extraction(tmp_path: Path):
    (tmp_path / "good.txt").write_text("Q3 update: revenue grew 12% year over year.")
    (tmp_path / "bad.pdf").write_bytes(bytes(range(256)) * 4)  # unextractable garbage

    client = FakeClient()
    result = await pipeline.run("How did revenue change?", tmp_path, make_config(), client)

    assert result.documents_scanned == 2
    assert len(result.documents_failed) == 1
    assert result.documents_failed[0][0].name == "bad.pdf"
    assert len(result.worker_results) == 1
    assert result.judge_result is not None
    assert result.judge_result.unanswered is False


@pytest.mark.asyncio
async def test_run_with_no_relevant_documents_returns_unanswered(tmp_path: Path):
    (tmp_path / "a.txt").write_text("Completely unrelated filler content.")

    client = FakeClient()
    result = await pipeline.run("How did revenue change?", tmp_path, make_config(), client)

    assert result.judge_result is not None
    assert result.judge_result.unanswered is True


@pytest.mark.asyncio
async def test_run_chunks_oversized_document_across_multiple_worker_calls(tmp_path: Path):
    from extractor.sizing import usable_tokens

    tiny_usable = usable_tokens(TIER, 0.85)
    paragraphs = [
        f"Filler paragraph number {i} with unique content padding." for i in range(tiny_usable)
    ]
    paragraphs.insert(len(paragraphs) // 2, "Buried fact: revenue grew 12% year over year.")
    (tmp_path / "huge.txt").write_text("\n\n".join(paragraphs))

    client = FakeClient()
    result = await pipeline.run("How did revenue change?", tmp_path, make_config(), client)

    assert result.documents_scanned == 1
    # The oversized doc must have been split into more than one chunk.
    assert len(result.worker_results) > 1
    assert all(r.doc_path.name == "huge.txt" for r in result.worker_results)
    assert all(r.total_chunks == len(result.worker_results) for r in result.worker_results)
    # Exactly one chunk should contain the buried fact and be marked relevant.
    relevant = [r for r in result.worker_results if r.found_relevant_info]
    assert len(relevant) == 1
    assert result.judge_result is not None
    assert result.judge_result.unanswered is False
