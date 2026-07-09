"""Token estimation and tier assignment."""

from __future__ import annotations

from extractor.models import TierConfig

_encoding = None


def get_encoding():
    """Lazily-loaded shared tiktoken encoding, reused by chunking.py."""
    global _encoding
    if _encoding is None:
        import tiktoken

        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def estimate_tokens(text: str) -> int:
    """Approximate token count. tiktoken's cl100k_base is a reasonable proxy for
    most instruct SLMs even though it isn't the model's exact tokenizer; falls
    back to a chars/4 heuristic if tiktoken can't be used."""
    try:
        return len(get_encoding().encode(text))
    except Exception:
        return len(text) // 4


def usable_tokens(tier: TierConfig, safety_margin: float) -> int:
    return int((tier.context_window_tokens - tier.reserved_output_tokens) * safety_margin)


def assign_tier(
    text: str, tiers_sorted_ascending: list[TierConfig], safety_margin: float
) -> TierConfig | None:
    """Return the smallest tier whose usable window fits `text`, or None if the
    text exceeds every tier's usable window (caller must chunk it)."""
    needed = estimate_tokens(text)
    for tier in tiers_sorted_ascending:
        if needed <= usable_tokens(tier, safety_margin):
            return tier
    return None
