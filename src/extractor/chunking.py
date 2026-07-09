"""Paragraph-aware sliding-window chunking for documents too large for any tier."""

from __future__ import annotations

from extractor.models import TierConfig
from extractor.sizing import estimate_tokens, get_encoding, usable_tokens


def _hard_slice(text: str, budget_tokens: int) -> list[str]:
    """Slice a single oversized paragraph into token-budget-sized pieces."""
    try:
        enc = get_encoding()
        tokens = enc.encode(text)
        return [
            enc.decode(tokens[i : i + budget_tokens]) for i in range(0, len(tokens), budget_tokens)
        ]
    except Exception:
        chars_per_piece = budget_tokens * 4
        return [text[i : i + chars_per_piece] for i in range(0, len(text), chars_per_piece)]


def _tail_tokens(text: str, n: int) -> str:
    """Return the last `n` tokens' worth of text, for carrying overlap forward."""
    if n <= 0:
        return ""
    try:
        enc = get_encoding()
        tokens = enc.encode(text)
        return enc.decode(tokens[-n:])
    except Exception:
        return text[-(n * 4) :]


def split_into_chunks(
    text: str, tier: TierConfig, overlap_tokens: int, safety_margin: float = 0.85
) -> list[str]:
    """Split `text` into chunks that each fit within `tier`'s usable context window,
    with `overlap_tokens` of trailing context carried into the next chunk."""
    budget = usable_tokens(tier, safety_margin)
    paragraphs = [p for p in text.split("\n\n") if p.strip()] or ([text] if text.strip() else [])

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        if para_tokens > budget:
            if current:
                chunks.append("\n\n".join(current))
                current, current_tokens = [], 0
            chunks.extend(_hard_slice(para, budget))
            continue

        if current and current_tokens + para_tokens > budget:
            chunks.append("\n\n".join(current))
            overlap_text = _tail_tokens(chunks[-1], overlap_tokens)
            current = [overlap_text] if overlap_text.strip() else []
            current_tokens = estimate_tokens(overlap_text) if overlap_text.strip() else 0

        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks
