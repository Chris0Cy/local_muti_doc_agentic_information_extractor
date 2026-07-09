"""Live integration tests against a real LM Studio instance.

Excluded by default (see `addopts = -m "not live"` in pyproject.toml). Run
explicitly with: pytest -m live

Before running, start LM Studio's local server (Developer tab -> Start
Server) and load the models referenced in config/default.yaml (or point
--config at a file matching whatever you actually have loaded).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from extractor.config import load_config
from extractor.lmstudio_client import LMStudioClient
from extractor.models import Chunk
from extractor.worker import ask_worker

pytestmark = pytest.mark.live


@pytest.fixture
async def client():
    config = load_config()
    c = LMStudioClient(config.lmstudio.base_url, config.lmstudio.request_timeout_s)
    yield c
    await c.aclose()


@pytest.mark.asyncio
async def test_health_check_against_real_server(client):
    await client.health_check()


@pytest.mark.asyncio
async def test_chat_completion_round_trip(client):
    config = load_config()
    tier = config.tiers_ascending[0]
    reply = await client.chat_completion(
        tier.model_id, [{"role": "user", "content": "Reply with exactly the word: pong"}]
    )
    assert reply.strip()


@pytest.mark.asyncio
async def test_ask_worker_against_real_model(client):
    config = load_config()
    tier = config.tiers_ascending[0]
    chunk = Chunk(
        doc_path=Path("sample.txt"),
        chunk_index=0,
        total_chunks=1,
        text="The company's revenue in Q3 2025 grew by 12% year over year, driven by strong cloud sales.",
        tier=tier,
    )
    result = await ask_worker(client, "How did revenue change?", chunk)
    assert result.error is None
    assert result.found_relevant_info is True
