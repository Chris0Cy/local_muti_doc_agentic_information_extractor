from __future__ import annotations

import pytest
from pydantic import ValidationError

from extractor.models import AppConfig, TierConfig


def make_tier(**overrides) -> dict:
    base = dict(name="tiny", model_id="m", context_window_tokens=4096, reserved_output_tokens=256)
    base.update(overrides)
    return base


def test_tier_config_rejects_reserved_output_equal_to_context_window():
    with pytest.raises(ValidationError):
        TierConfig(**make_tier(context_window_tokens=1000, reserved_output_tokens=1000))


def test_tier_config_rejects_reserved_output_exceeding_context_window():
    with pytest.raises(ValidationError):
        TierConfig(**make_tier(context_window_tokens=1000, reserved_output_tokens=2000))


def test_tier_config_accepts_valid_budget():
    tier = TierConfig(**make_tier(context_window_tokens=4096, reserved_output_tokens=512))
    assert tier.context_window_tokens == 4096


def test_app_config_rejects_empty_tiers_list():
    with pytest.raises(ValidationError):
        AppConfig(tiers=[], judge=TierConfig(**make_tier()))


def test_app_config_accepts_at_least_one_tier():
    config = AppConfig(tiers=[TierConfig(**make_tier())], judge=TierConfig(**make_tier()))
    assert len(config.tiers) == 1
