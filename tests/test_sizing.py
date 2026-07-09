from __future__ import annotations

from extractor.models import TierConfig
from extractor.sizing import assign_tier, estimate_tokens, usable_tokens

TIERS = [
    TierConfig(
        name="tiny", model_id="m-tiny", context_window_tokens=1000, reserved_output_tokens=200
    ),
    TierConfig(
        name="big", model_id="m-big", context_window_tokens=10000, reserved_output_tokens=1000
    ),
]


def test_estimate_tokens_nonzero_for_text():
    assert estimate_tokens("hello world, this is a test") > 0


def test_estimate_tokens_zero_for_empty_string():
    assert estimate_tokens("") == 0


def test_usable_tokens_applies_reserved_and_margin():
    tier = TIERS[0]
    assert usable_tokens(tier, 0.85) == int((1000 - 200) * 0.85)


def test_assign_tier_picks_smallest_that_fits():
    small_text = "word " * 10  # well under the tiny tier's usable budget
    tier = assign_tier(small_text, TIERS, safety_margin=0.85)
    assert tier is not None
    assert tier.name == "tiny"


def test_assign_tier_escalates_to_bigger_tier():
    # Build text that exceeds the tiny tier's usable window but fits the big one.
    tiny_usable = usable_tokens(TIERS[0], 0.85)
    text = "word " * (tiny_usable + 50)
    tier = assign_tier(text, TIERS, safety_margin=0.85)
    assert tier is not None
    assert tier.name == "big"


def test_assign_tier_returns_none_when_oversized():
    big_usable = usable_tokens(TIERS[1], 0.85)
    text = "word " * (big_usable + 500)
    tier = assign_tier(text, TIERS, safety_margin=0.85)
    assert tier is None
