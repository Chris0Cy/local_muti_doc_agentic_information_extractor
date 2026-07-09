from __future__ import annotations

from extractor.chunking import split_into_chunks
from extractor.models import TierConfig
from extractor.sizing import estimate_tokens, usable_tokens

TIER = TierConfig(name="tiny", model_id="m", context_window_tokens=500, reserved_output_tokens=100)


def test_split_returns_single_chunk_when_text_fits():
    text = "This is a short paragraph.\n\nAnd another one."
    chunks = split_into_chunks(text, TIER, overlap_tokens=10)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_produces_multiple_chunks_when_oversized():
    budget = usable_tokens(TIER, 0.85)
    # Build enough distinct paragraphs to exceed the tier's usable budget several times over.
    paragraphs = [f"Paragraph number {i} with some unique filler content here." for i in range(200)]
    text = "\n\n".join(paragraphs)
    assert estimate_tokens(text) > budget

    chunks = split_into_chunks(text, TIER, overlap_tokens=10)
    assert len(chunks) > 1
    for chunk in chunks:
        assert estimate_tokens(chunk) <= budget + 10  # small tolerance for overlap carry-over


def test_split_no_content_lost_ignoring_overlap_duplication():
    paragraphs = [f"Paragraph {i}" for i in range(50)]
    text = "\n\n".join(paragraphs)
    chunks = split_into_chunks(text, TIER, overlap_tokens=0)
    reassembled = "\n\n".join(chunks)
    for p in paragraphs:
        assert p in reassembled


def test_split_empty_text_returns_empty_list():
    assert split_into_chunks("", TIER, overlap_tokens=10) == []


def test_split_hard_slices_a_single_oversized_paragraph():
    budget = usable_tokens(TIER, 0.85)
    huge_paragraph = "word " * (budget * 3)
    chunks = split_into_chunks(huge_paragraph, TIER, overlap_tokens=0)
    assert len(chunks) > 1
    for chunk in chunks:
        assert estimate_tokens(chunk) <= budget + 5
