from __future__ import annotations

from extractor.extraction import extract_text


def test_extracts_docx(make_docx):
    path = make_docx(["First paragraph.", "Second paragraph with detail."])
    doc = extract_text(path)
    assert doc.extraction_method == "docx"
    assert "First paragraph." in doc.text
    assert "Second paragraph with detail." in doc.text
