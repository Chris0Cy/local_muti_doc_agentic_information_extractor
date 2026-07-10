from __future__ import annotations

import httpx
import pytest
import respx

from extractor.lmstudio_client import LMStudioClient, LMStudioRequestError, LMStudioUnavailable

BASE_URL = "http://localhost:1234/v1"


@pytest.mark.asyncio
async def test_health_check_success():
    with respx.mock(base_url=BASE_URL) as mock:
        mock.get("/models").mock(return_value=httpx.Response(200, json={"data": []}))
        client = LMStudioClient(BASE_URL)
        await client.health_check()
        await client.aclose()


@pytest.mark.asyncio
async def test_health_check_connection_refused_raises_unavailable():
    with respx.mock(base_url=BASE_URL) as mock:
        mock.get("/models").mock(side_effect=httpx.ConnectError("refused"))
        client = LMStudioClient(BASE_URL)
        with pytest.raises(LMStudioUnavailable):
            await client.health_check()
        await client.aclose()


@pytest.mark.asyncio
async def test_list_models_returns_ids():
    with respx.mock(base_url=BASE_URL) as mock:
        mock.get("/models").mock(
            return_value=httpx.Response(
                200, json={"data": [{"id": "qwen2.5-0.5b-instruct"}, {"id": "qwen2.5-7b-instruct"}]}
            )
        )
        client = LMStudioClient(BASE_URL)
        models = await client.list_models()
        await client.aclose()
    assert models == ["qwen2.5-0.5b-instruct", "qwen2.5-7b-instruct"]


@pytest.mark.asyncio
async def test_list_models_server_error_raises_unavailable():
    with respx.mock(base_url=BASE_URL) as mock:
        mock.get("/models").mock(return_value=httpx.Response(500, text="internal error"))
        client = LMStudioClient(BASE_URL)
        with pytest.raises(LMStudioUnavailable):
            await client.list_models()
        await client.aclose()


@pytest.mark.asyncio
async def test_chat_completion_sends_expected_payload_and_parses_content():
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        import json as jsonlib

        captured["payload"] = jsonlib.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello back"}}]})

    with respx.mock(base_url=BASE_URL) as mock:
        mock.post("/chat/completions").mock(side_effect=_handler)
        client = LMStudioClient(BASE_URL)
        result = await client.chat_completion(
            "qwen2.5-0.5b-instruct", [{"role": "user", "content": "hi"}], temperature=0.0
        )
        await client.aclose()

    assert result == "hello back"
    assert captured["payload"]["model"] == "qwen2.5-0.5b-instruct"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["payload"]["temperature"] == 0.0


@pytest.mark.asyncio
async def test_chat_completion_model_not_loaded_raises_request_error():
    with respx.mock(base_url=BASE_URL) as mock:
        mock.post("/chat/completions").mock(
            return_value=httpx.Response(404, text="model not found")
        )
        client = LMStudioClient(BASE_URL)
        with pytest.raises(LMStudioRequestError):
            await client.chat_completion("missing-model", [{"role": "user", "content": "hi"}])
        await client.aclose()


@pytest.mark.asyncio
async def test_chat_completion_connection_refused_raises_unavailable():
    with respx.mock(base_url=BASE_URL) as mock:
        mock.post("/chat/completions").mock(side_effect=httpx.ConnectError("refused"))
        client = LMStudioClient(BASE_URL)
        with pytest.raises(LMStudioUnavailable):
            await client.chat_completion("m", [{"role": "user", "content": "hi"}])
        await client.aclose()
