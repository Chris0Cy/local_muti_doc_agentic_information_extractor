from __future__ import annotations

import pytest

from extractor.extraction import extract_text
from extractor.extraction.base import ExtractionError


def test_unknown_suffix_falls_back_to_markitdown(tmp_path):
    p = tmp_path / "sample.xyz"
    p.write_text("Plain content in an unrecognized-suffix file.", encoding="utf-8")

    doc = extract_text(p)
    assert doc.extraction_method == "markitdown"
    assert "Plain content" in doc.text


def test_unreadable_file_raises_extraction_error(tmp_path):
    p = tmp_path / "empty.xyz"
    p.write_bytes(b"")

    with pytest.raises(ExtractionError):
        extract_text(p)
