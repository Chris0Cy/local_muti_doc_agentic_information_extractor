from __future__ import annotations

from pathlib import Path

from extractor.embedding_cache import EmbeddingCache, load_cache, save_cache


def test_get_returns_none_for_unknown_path(tmp_path: Path):
    cache = EmbeddingCache()
    assert cache.get(tmp_path / "a.txt", mtime_ns=1, size=1, model_id="m") is None


def test_put_then_get_round_trips(tmp_path: Path):
    cache = EmbeddingCache()
    path = tmp_path / "a.txt"
    cache.put(path, mtime_ns=100, size=10, model_id="m", chunk_embeddings=[[0.1, 0.2]])
    assert cache.get(path, mtime_ns=100, size=10, model_id="m") == [[0.1, 0.2]]


def test_get_returns_none_when_mtime_changed(tmp_path: Path):
    cache = EmbeddingCache()
    path = tmp_path / "a.txt"
    cache.put(path, mtime_ns=100, size=10, model_id="m", chunk_embeddings=[[0.1, 0.2]])
    assert cache.get(path, mtime_ns=999, size=10, model_id="m") is None


def test_get_returns_none_when_size_changed(tmp_path: Path):
    cache = EmbeddingCache()
    path = tmp_path / "a.txt"
    cache.put(path, mtime_ns=100, size=10, model_id="m", chunk_embeddings=[[0.1, 0.2]])
    assert cache.get(path, mtime_ns=100, size=999, model_id="m") is None


def test_get_returns_none_when_model_id_differs(tmp_path: Path):
    cache = EmbeddingCache()
    path = tmp_path / "a.txt"
    cache.put(path, mtime_ns=100, size=10, model_id="model-a", chunk_embeddings=[[0.1, 0.2]])
    assert cache.get(path, mtime_ns=100, size=10, model_id="model-b") is None


def test_put_with_new_model_id_wholesale_invalidates_other_entries(tmp_path: Path):
    cache = EmbeddingCache()
    path_a = tmp_path / "a.txt"
    path_b = tmp_path / "b.txt"
    cache.put(path_a, mtime_ns=100, size=10, model_id="model-a", chunk_embeddings=[[0.1]])
    cache.put(path_b, mtime_ns=200, size=20, model_id="model-b", chunk_embeddings=[[0.2]])
    # path_a's entry was wiped when the model changed, even though path_a itself wasn't re-put.
    assert cache.get(path_a, mtime_ns=100, size=10, model_id="model-b") is None
    assert cache.get(path_b, mtime_ns=200, size=20, model_id="model-b") == [[0.2]]


def test_load_cache_missing_file_returns_empty_cache(tmp_path: Path):
    cache = load_cache(tmp_path / "does_not_exist.json")
    assert cache.embedding_model_id is None
    assert cache.entries == {}


def test_load_cache_corrupt_file_returns_empty_cache(tmp_path: Path):
    cache_path = tmp_path / "corrupt.json"
    cache_path.write_text("{not valid json", encoding="utf-8")
    cache = load_cache(cache_path)
    assert cache.embedding_model_id is None
    assert cache.entries == {}


def test_save_then_load_round_trips(tmp_path: Path):
    cache_path = tmp_path / "subdir" / "embeddings.json"
    cache = EmbeddingCache()
    path = tmp_path / "a.txt"
    cache.put(path, mtime_ns=100, size=10, model_id="m", chunk_embeddings=[[0.1, 0.2]])

    save_cache(cache_path, cache)
    assert cache_path.exists()

    loaded = load_cache(cache_path)
    assert loaded.get(path, mtime_ns=100, size=10, model_id="m") == [[0.1, 0.2]]
