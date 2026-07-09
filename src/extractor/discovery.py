"""Folder scanning for candidate document files."""

from __future__ import annotations

from pathlib import Path

SKIP_SUFFIXES = {".pyc", ".exe", ".dll", ".so", ".zip", ".tmp"}


def scan_folder(folder: Path, recursive: bool = True) -> list[Path]:
    """Return a deterministically ordered list of candidate document files.

    Hidden files/directories (leading dot) and a small set of obviously
    non-document suffixes are skipped; everything else is handed to the
    extraction layer, which has its own fallback for unrecognized formats.
    """
    pattern = "**/*" if recursive else "*"
    paths = [
        p
        for p in folder.glob(pattern)
        if p.is_file()
        and p.suffix.lower() not in SKIP_SUFFIXES
        and not any(part.startswith(".") for part in p.relative_to(folder).parts)
    ]
    return sorted(paths)
