from __future__ import annotations

from extractor.extraction import extract_text


def test_extracts_xlsx(make_xlsx):
    path = make_xlsx([["name", "value"], ["alpha", "1"], ["beta", "2"]])
    doc = extract_text(path)
    assert doc.extraction_method == "xlsx"
    assert "alpha" in doc.text
    assert "beta" in doc.text


def test_extracts_csv(make_csv):
    path = make_csv([["name", "value"], ["alpha", "1"], ["beta", "2"]])
    doc = extract_text(path)
    assert doc.extraction_method == "csv"
    assert "alpha" in doc.text
    assert "beta" in doc.text
