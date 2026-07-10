"""Async client for LM Studio's OpenAI-compatible REST API."""

from __future__ import annotations

from typing import Protocol

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class LMStudioUnavailable(Exception):
    """Raised when LM Studio cannot be reached at all (connection refused/timeout)."""


class LMStudioRequestError(Exception):
    """Raised when a specific request fails (e.g. the requested model isn't loaded)."""

    def __init__(self, model: str, detail: str):
        self.model = model
        self.detail = detail
        super().__init__(f"Request to model '{model}' failed: {detail}")


class LMStudioClientProtocol(Protocol):
    async def health_check(self) -> None: ...

    async def list_models(self) -> list[str]: ...

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str: ...

    async def embeddings(self, model: str, inputs: list[str]) -> list[list[float]]: ...


class LMStudioClient:
    """Thin wrapper around LM Studio's `/v1` OpenAI-compatible endpoints."""

    def __init__(self, base_url: str, timeout: float = 120.0):
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health_check(self) -> None:
        try:
            response = await self._client.get("/models")
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise LMStudioUnavailable(
                f"Could not reach LM Studio at {self._client.base_url}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise LMStudioUnavailable(f"LM Studio returned an error: {e}") from e

    async def list_models(self) -> list[str]:
        try:
            response = await self._client.get("/models")
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise LMStudioUnavailable(
                f"Could not reach LM Studio at {self._client.base_url}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise LMStudioUnavailable(f"LM Studio returned an error: {e}") from e
        data = response.json()
        return [item["id"] for item in data.get("data", [])]

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    async def _post_chat_completion(self, payload: dict) -> httpx.Response:
        # Left undecorated of any try/except so tenacity's retry_if_exception_type
        # actually sees the raw httpx.TransportError and can retry it; converting
        # to our own exception types happens only after retries are exhausted.
        return await self._client.post("/chat/completions", json=payload)

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        payload: dict = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            response = await self._post_chat_completion(payload)
        except httpx.TransportError as e:
            raise LMStudioUnavailable(
                f"Could not reach LM Studio at {self._client.base_url}"
            ) from e

        if response.status_code >= 400:
            raise LMStudioRequestError(model, f"HTTP {response.status_code}: {response.text[:500]}")

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LMStudioRequestError(model, f"Unexpected response shape: {data}") from e

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    async def _post_embeddings(self, payload: dict) -> httpx.Response:
        return await self._client.post("/embeddings", json=payload)

    async def embeddings(self, model: str, inputs: list[str]) -> list[list[float]]:
        payload = {"model": model, "input": inputs}

        try:
            response = await self._post_embeddings(payload)
        except httpx.TransportError as e:
            raise LMStudioUnavailable(
                f"Could not reach LM Studio at {self._client.base_url}"
            ) from e

        if response.status_code >= 400:
            raise LMStudioRequestError(model, f"HTTP {response.status_code}: {response.text[:500]}")

        data = response.json()
        try:
            return [item["embedding"] for item in data["data"]]
        except (KeyError, TypeError) as e:
            raise LMStudioRequestError(model, f"Unexpected response shape: {data}") from e
