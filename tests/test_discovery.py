from __future__ import annotations

from pathlib import Path

from extractor.discovery import scan_folder


def test_scan_folder_finds_files_recursively(tmp_path: Path):
    (tmp_path / "a.txt").write_text("a")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.md").write_text("b")

    paths = scan_folder(tmp_path)
    names = {p.name for p in paths}
    assert names == {"a.txt", "b.md"}


def test_scan_folder_skips_hidden_files_and_dirs(tmp_path: Path):
    (tmp_path / ".hidden.txt").write_text("h")
    hidden_dir = tmp_path / ".git"
    hidden_dir.mkdir()
    (hidden_dir / "config").write_text("c")
    (tmp_path / "visible.txt").write_text("v")

    paths = scan_folder(tmp_path)
    names = {p.name for p in paths}
    assert names == {"visible.txt"}


def test_scan_folder_skips_known_non_document_suffixes(tmp_path: Path):
    (tmp_path / "module.pyc").write_bytes(b"\x00")
    (tmp_path / "doc.txt").write_text("d")

    paths = scan_folder(tmp_path)
    names = {p.name for p in paths}
    assert names == {"doc.txt"}


def test_scan_folder_non_recursive(tmp_path: Path):
    (tmp_path / "top.txt").write_text("t")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.txt").write_text("n")

    paths = scan_folder(tmp_path, recursive=False)
    names = {p.name for p in paths}
    assert names == {"top.txt"}


def test_scan_folder_deterministic_order(tmp_path: Path):
    for name in ["c.txt", "a.txt", "b.txt"]:
        (tmp_path / name).write_text(name)

    paths = scan_folder(tmp_path)
    assert [p.name for p in paths] == ["a.txt", "b.txt", "c.txt"]
