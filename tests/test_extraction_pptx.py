from __future__ import annotations

from extractor.extraction import extract_text


def test_extracts_pptx(make_pptx):
    path = make_pptx(["Slide one title.", "Slide two title."])
    doc = extract_text(path)
    assert doc.extraction_method == "pptx"
    assert "Slide one title." in doc.text
    assert "Slide two title." in doc.text
