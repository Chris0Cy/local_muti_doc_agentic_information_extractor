"""Judge: synthesizes all worker findings into one final, cited answer."""

from __future__ import annotations

from extractor.lmstudio_client import (
    LMStudioClientProtocol,
    LMStudioRequestError,
    LMStudioUnavailable,
)
from extractor.models import JudgeResult, TierConfig, WorkerResult
from extractor.prompts import JUDGE_SYSTEM_PROMPT, judge_user_prompt


async def synthesize(
    client: LMStudioClientProtocol,
    question: str,
    results: list[WorkerResult],
    judge_tier: TierConfig,
) -> JudgeResult:
    relevant = [r for r in results if r.found_relevant_info and r.answer_excerpt]
    if not relevant:
        return JudgeResult(
            final_answer_markdown="No relevant information was found in any of the provided documents.",
            unanswered=True,
            raw_response="",
        )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": judge_user_prompt(question, results)},
    ]

    try:
        raw = await client.chat_completion(
            judge_tier.model_id,
            messages,
            temperature=0.0,
            max_tokens=judge_tier.reserved_output_tokens,
        )
    except (LMStudioUnavailable, LMStudioRequestError) as e:
        return JudgeResult(
            final_answer_markdown=f"Judge synthesis failed: {e}", unanswered=True, raw_response=""
        )

    return JudgeResult(final_answer_markdown=raw, unanswered=False, raw_response=raw)
