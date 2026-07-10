"""On-disk cache of document chunk embeddings, keyed by path/mtime/size.

Embeddings are query-independent, so caching them is what makes repeated
`ask` runs against a mostly-static corpus cheap -- only the question needs
re-embedding every time. A change in embedding_model_id invalidates the whole
cache, since embeddings from different models aren't comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class _CacheEntry(BaseModel):
    mtime_ns: int
    size: int
    chunk_embeddings: list[list[float]]


class EmbeddingCache(BaseModel):
    embedding_model_id: str | None = None
    entries: dict[str, _CacheEntry] = Field(default_factory=dict)

    def get(self, path: Path, mtime_ns: int, size: int, model_id: str) -> list[list[float]] | None:
        if self.embedding_model_id != model_id:
            return None
        entry = self.entries.get(str(path.resolve()))
        if entry is None or entry.mtime_ns != mtime_ns or entry.size != size:
            return None
        return entry.chunk_embeddings

    def put(
        self,
        path: Path,
        mtime_ns: int,
        size: int,
        model_id: str,
        chunk_embeddings: list[list[float]],
    ) -> None:
        if self.embedding_model_id != model_id:
            # Different embedding space -- the rest of the cache is meaningless now.
            self.entries.clear()
            self.embedding_model_id = model_id
        self.entries[str(path.resolve())] = _CacheEntry(
            mtime_ns=mtime_ns, size=size, chunk_embeddings=chunk_embeddings
        )


def load_cache(path: Path) -> EmbeddingCache:
    """Load the cache from disk. A missing or corrupt file is never fatal --
    just starts from an empty cache."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return EmbeddingCache.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError):
        return EmbeddingCache()


def save_cache(path: Path, cache: EmbeddingCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cache.model_dump_json(indent=2), encoding="utf-8")
