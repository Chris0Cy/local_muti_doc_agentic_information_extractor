"""Per-document/chunk worker: asks the assigned SLM the user's question."""

from __future__ import annotations

import json
import re

from extractor.lmstudio_client import (
    LMStudioClientProtocol,
    LMStudioRequestError,
    LMStudioUnavailable,
)
from extractor.models import Chunk, WorkerResult
from extractor.prompts import WORKER_SYSTEM_PROMPT, worker_user_prompt

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_NEGATIVE_SIGNALS = (
    "not relevant",
    "no relevant",
    "n't see any relevant",
    "not applicable",
    "does not contain",
    "doesn't contain",
    "no information",
    "no mention",
    "not mentioned",
    "cannot find",
    "can't find",
    "could not find",
    "not found",
    "not related",
    "irrelevant",
)


def _parse_worker_response(raw: str) -> tuple[bool, str | None, str | None]:
    """Tolerant parsing chain for small-model JSON output.

    Returns (found_relevant_info, answer_excerpt, confidence). Falls back to
    treating the raw text as a low-confidence excerpt if no valid JSON can be
    recovered at all — a known limitation of small instruct models that don't
    always emit clean JSON.
    """
    candidates = [raw]
    match = _JSON_OBJECT_RE.search(raw)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "found_relevant_info" in data:
            found = bool(data.get("found_relevant_info"))
            excerpt = data.get("answer_excerpt")
            # Small models occasionally say found_relevant_info=true but leave
            # answer_excerpt empty/null; without an excerpt there's nothing for
            # the judge to cite, so treat it as not-found rather than a silent
            # drop at the judge-filtering stage.
            if found and not (isinstance(excerpt, str) and excerpt.strip()):
                found, excerpt = False, None
            return found, excerpt, data.get("confidence")

    stripped = raw.strip()
    if stripped:
        # No valid JSON could be recovered at all. Rather than blindly treating
        # any prose as a positive finding, check for an explicit negative
        # signal first -- otherwise a plain-language refusal like "I don't see
        # any relevant information" would get surfaced to the judge as a
        # "relevant" excerpt.
        lowered = stripped.lower()
        if any(signal in lowered for signal in _NEGATIVE_SIGNALS):
            return False, None, None
        return True, stripped, "low"
    return False, None, None


async def ask_worker(client: LMStudioClientProtocol, question: str, chunk: Chunk) -> WorkerResult:
    """Ask the tier-assigned model about one document/chunk. Never raises —
    failures are captured on WorkerResult.error so one bad call doesn't abort
    the rest of the run."""
    messages = [
        {"role": "system", "content": WORKER_SYSTEM_PROMPT},
        {"role": "user", "content": worker_user_prompt(question, chunk)},
    ]

    try:
        raw = await client.chat_completion(chunk.tier.model_id, messages, temperature=0.0)
    except (LMStudioUnavailable, LMStudioRequestError) as e:
        return WorkerResult(
            doc_path=chunk.doc_path,
            chunk_index=chunk.chunk_index,
            total_chunks=chunk.total_chunks,
            tier_used=chunk.tier.name,
            found_relevant_info=False,
            error=str(e),
        )

    found, excerpt, confidence = _parse_worker_response(raw)
    return WorkerResult(
        doc_path=chunk.doc_path,
        chunk_index=chunk.chunk_index,
        total_chunks=chunk.total_chunks,
        tier_used=chunk.tier.name,
        found_relevant_info=found,
        answer_excerpt=excerpt,
        confidence=confidence if confidence in ("high", "medium", "low") else None,
        raw_response=raw,
    )
