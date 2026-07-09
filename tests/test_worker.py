from __future__ import annotations

from pathlib import Path

import pytest

from extractor.lmstudio_client import LMStudioRequestError
from extractor.models import Chunk, TierConfig
from extractor.worker import ask_worker

TIER = TierConfig(
    name="tiny", model_id="m-tiny", context_window_tokens=4096, reserved_output_tokens=256
)


class FakeClient:
    def __init__(self, response: str | None = None, raise_error: Exception | None = None):
        self.response = response
        self.raise_error = raise_error
        self.received_model: str | None = None

    async def health_check(self) -> None:
        pass

    async def list_models(self) -> list[str]:
        return [TIER.model_id]

    async def chat_completion(self, model, messages, *, temperature=0.0, max_tokens=None) -> str:
        self.received_model = model
        if self.raise_error:
            raise self.raise_error
        return self.response


def make_chunk(text: str = "Some document text.") -> Chunk:
    return Chunk(doc_path=Path("doc.txt"), chunk_index=0, total_chunks=1, text=text, tier=TIER)


@pytest.mark.asyncio
async def test_ask_worker_parses_clean_json():
    client = FakeClient(
        response='{"found_relevant_info": true, "answer_excerpt": "the answer", "confidence": "high"}'
    )
    result = await ask_worker(client, "What is the answer?", make_chunk())
    assert result.found_relevant_info is True
    assert result.answer_excerpt == "the answer"
    assert result.confidence == "high"
    assert result.error is None
    assert client.received_model == TIER.model_id


@pytest.mark.asyncio
async def test_ask_worker_parses_json_embedded_in_prose():
    client = FakeClient(
        response='Sure, here is the result:\n{"found_relevant_info": false, "answer_excerpt": null, "confidence": null}\nHope that helps!'
    )
    result = await ask_worker(client, "q", make_chunk())
    assert result.found_relevant_info is False
    assert result.answer_excerpt is None


@pytest.mark.asyncio
async def test_ask_worker_falls_back_to_raw_text_on_malformed_json():
    client = FakeClient(response="I think the answer is 42, based on the document.")
    result = await ask_worker(client, "q", make_chunk())
    assert result.found_relevant_info is True
    assert result.confidence == "low"
    assert "42" in result.answer_excerpt


@pytest.mark.asyncio
async def test_ask_worker_empty_response_treated_as_not_found():
    client = FakeClient(response="   ")
    result = await ask_worker(client, "q", make_chunk())
    assert result.found_relevant_info is False
    assert result.answer_excerpt is None


@pytest.mark.asyncio
async def test_ask_worker_downgrades_found_true_with_missing_excerpt():
    # Small models sometimes say found_relevant_info=true but leave the excerpt
    # empty; that's unusable for the judge, so it should be downgraded to not-found.
    client = FakeClient(
        response='{"found_relevant_info": true, "answer_excerpt": null, "confidence": "high"}'
    )
    result = await ask_worker(client, "q", make_chunk())
    assert result.found_relevant_info is False
    assert result.answer_excerpt is None


@pytest.mark.asyncio
async def test_ask_worker_captures_request_error_without_raising():
    client = FakeClient(raise_error=LMStudioRequestError("m-tiny", "model not loaded"))
    result = await ask_worker(client, "q", make_chunk())
    assert result.error is not None
    assert "model not loaded" in result.error
    assert result.found_relevant_info is False
