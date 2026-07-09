from __future__ import annotations

from extractor.extraction import extract_text


def test_extracts_eml(make_eml):
    path = make_eml(subject="Quarterly update", body="Revenue grew 12% this quarter.")
    doc = extract_text(path)
    assert doc.extraction_method == "eml"
    assert "Quarterly update" in doc.text
    assert "Revenue grew 12%" in doc.text
