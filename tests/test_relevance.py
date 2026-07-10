from __future__ import annotations

from pathlib import Path

import pytest

from extractor import chunking
from extractor.lmstudio_client import LMStudioUnavailable
from extractor.models import ExtractedDocument, RelevanceFilterConfig, TierConfig
from extractor.relevance import _cosine_similarity, _embed_batched, filter_by_relevance

EMBED_TIER = TierConfig(
    name="embed", model_id="embed-model", context_window_tokens=50, reserved_output_tokens=0
)


class FakeEmbeddingClient:
    def __init__(self, vectors: dict[str, list[float]], default_vector: list[float] | None = None):
        self.vectors = vectors
        self.default_vector = default_vector or [0.0, 0.0, 1.0]
        self.embed_calls: list[list[str]] = []
        self.raise_error: Exception | None = None
        self.drop_last_from_batches: bool = False

    async def health_check(self) -> None:
        pass

    async def list_models(self) -> list[str]:
        return []

    async def chat_completion(self, *args, **kwargs):
        raise NotImplementedError

    async def embeddings(self, model: str, inputs: list[str]) -> list[list[float]]:
        if self.raise_error:
            raise self.raise_error
        self.embed_calls.append(list(inputs))
        result = [self.vectors.get(text, self.default_vector) for text in inputs]
        if self.drop_last_from_batches and len(inputs) > 1:
            # Simulate a misbehaving server returning fewer vectors than requested.
            result = result[:-1]
        return result


def make_doc(path: str, text: str) -> ExtractedDocument:
    return ExtractedDocument(path=Path(path), text=text, extraction_method="text")


def make_config(tmp_path: Path, **overrides) -> RelevanceFilterConfig:
    base = dict(embedding_model=EMBED_TIER, cache_path=str(tmp_path / "cache.json"))
    base.update(overrides)
    return RelevanceFilterConfig(**base)


def test_cosine_similarity_identical_vectors_is_one():
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_returns_zero_not_nan():
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


@pytest.mark.asyncio
async def test_embed_batched_splits_into_batches_of_32():
    client = FakeEmbeddingClient(vectors={}, default_vector=[1.0])
    texts = [f"text {i}" for i in range(40)]
    result = await _embed_batched(client, "m", texts)
    assert len(result) == 40
    assert len(client.embed_calls) == 2
    assert len(client.embed_calls[0]) == 32
    assert len(client.embed_calls[1]) == 8


@pytest.mark.asyncio
async def test_whole_doc_fits_in_one_chunk_and_is_scored(tmp_path: Path):
    doc = make_doc("a.txt", "hello world")
    client = FakeEmbeddingClient(vectors={"the question": [1.0, 0.0], "hello world": [1.0, 0.0]})
    config = make_config(tmp_path, top_k=None, similarity_floor=0.0)

    kept, result = await filter_by_relevance(
        client, "the question", [doc], config, safety_margin=0.85
    )

    assert len(kept) == 1
    assert result.enabled is True
    assert result.scores[0].score == pytest.approx(1.0)
    assert result.scores[0].kept is True


@pytest.mark.asyncio
async def test_oversized_document_scored_by_max_not_average_across_chunks(tmp_path: Path):
    text = (
        "Cats are wonderful pets and very relevant to the question at hand today.\n\n"
        "Rocks are heavy and mostly irrelevant filler content that goes on for quite a while."
    )
    tiny_embed_tier = TierConfig(
        name="embed", model_id="embed-model", context_window_tokens=15, reserved_output_tokens=0
    )
    chunks = chunking.split_into_chunks(text, tiny_embed_tier, overlap_tokens=0, safety_margin=0.85)
    assert len(chunks) >= 2, "fixture must actually produce multiple chunks to test max-vs-average"

    vectors = {"the question": [1.0, 0.0], chunks[0]: [1.0, 0.0]}
    for c in chunks[1:]:
        vectors[c] = [0.0, 1.0]  # orthogonal -> similarity 0.0, would drag an average down

    doc = make_doc("a.txt", text)
    client = FakeEmbeddingClient(vectors)
    config = make_config(
        tmp_path, embedding_model=tiny_embed_tier, top_k=None, similarity_floor=0.0
    )

    _, result = await filter_by_relevance(client, "the question", [doc], config, safety_margin=0.85)

    assert result.scores[0].score == pytest.approx(1.0)  # max, not (1.0 + 0.0*n) / (n+1)


@pytest.mark.asyncio
async def test_top_k_keeps_highest_scoring_regardless_of_absolute_score(tmp_path: Path):
    docs = [make_doc(f"{i}.txt", f"doc {i}") for i in range(3)]
    vectors = {
        "q": [1.0, 0.0],
        "doc 0": [1.0, 0.0],  # score 1.0
        "doc 1": [0.7, 0.3],  # score ~0.92
        "doc 2": [0.0, 1.0],  # score 0.0
    }
    client = FakeEmbeddingClient(vectors)
    config = make_config(tmp_path, top_k=2, similarity_floor=None)

    kept, result = await filter_by_relevance(client, "q", docs, config, safety_margin=0.85)

    kept_names = {d.path.name for d in kept}
    assert kept_names == {"0.txt", "1.txt"}


@pytest.mark.asyncio
async def test_similarity_floor_keeps_all_above_threshold_regardless_of_count(tmp_path: Path):
    docs = [make_doc(f"{i}.txt", f"doc {i}") for i in range(3)]
    vectors = {
        "q": [1.0, 0.0],
        "doc 0": [1.0, 0.0],  # score 1.0
        "doc 1": [0.9, 0.1],  # score ~0.99
        "doc 2": [0.0, 1.0],  # score 0.0
    }
    client = FakeEmbeddingClient(vectors)
    config = make_config(tmp_path, top_k=None, similarity_floor=0.5)

    kept, result = await filter_by_relevance(client, "q", docs, config, safety_margin=0.85)

    kept_names = {d.path.name for d in kept}
    assert kept_names == {"0.txt", "1.txt"}


@pytest.mark.asyncio
async def test_top_k_and_floor_combined_both_must_pass(tmp_path: Path):
    docs = [make_doc(f"{i}.txt", f"doc {i}") for i in range(2)]
    vectors = {
        "q": [1.0, 0.0],
        "doc 0": [1.0, 0.0],  # score 1.0, clears floor, rank 0
        "doc 1": [0.95, 0.05],  # clears floor too, but rank 1 -- excluded by top_k=1
    }
    client = FakeEmbeddingClient(vectors)
    config = make_config(tmp_path, top_k=1, similarity_floor=0.5)

    kept, _ = await filter_by_relevance(client, "q", docs, config, safety_margin=0.85)

    assert [d.path.name for d in kept] == ["0.txt"]


@pytest.mark.asyncio
async def test_cache_hit_skips_reembedding_document_chunks(tmp_path: Path):
    doc = make_doc("a.txt", "hello world")
    vectors = {"q": [1.0, 0.0], "hello world": [1.0, 0.0]}
    client = FakeEmbeddingClient(vectors)
    config = make_config(tmp_path, top_k=None, similarity_floor=0.0)

    await filter_by_relevance(client, "q", [doc], config, safety_margin=0.85)
    calls_after_first_run = len(client.embed_calls)

    # Second run against the same unchanged file: only the question should be
    # re-embedded, the document's chunk embedding should come from cache.
    await filter_by_relevance(client, "q", [doc], config, safety_margin=0.85)
    calls_after_second_run = len(client.embed_calls)

    assert calls_after_second_run == calls_after_first_run + 1  # +1 for the question only


@pytest.mark.asyncio
async def test_embedding_failure_fails_safe_and_keeps_all_documents(tmp_path: Path):
    docs = [make_doc("a.txt", "hello"), make_doc("b.txt", "world")]
    client = FakeEmbeddingClient(vectors={})
    client.raise_error = LMStudioUnavailable("LM Studio is down")
    config = make_config(tmp_path, top_k=1, similarity_floor=0.9)  # would normally drop a lot

    kept, result = await filter_by_relevance(client, "q", docs, config, safety_margin=0.85)

    assert len(kept) == 2  # nothing dropped despite the strict thresholds
    assert result.enabled is True
    assert result.scores == []


@pytest.mark.asyncio
async def test_empty_document_list_returns_empty_unchanged(tmp_path: Path):
    client = FakeEmbeddingClient(vectors={})
    config = make_config(tmp_path)

    kept, result = await filter_by_relevance(client, "q", [], config, safety_margin=0.85)

    assert kept == []
    assert result.enabled is True


@pytest.mark.asyncio
async def test_embedding_count_mismatch_fails_safe_and_keeps_all_documents(tmp_path: Path):
    # Two docs -> two chunk texts in one batch call. A misbehaving server
    # returns only one vector back; slicing by cursor position would
    # otherwise silently misassign the remaining embeddings to the wrong docs.
    docs = [make_doc("a.txt", "hello"), make_doc("b.txt", "world")]
    client = FakeEmbeddingClient(vectors={"hello": [1.0, 0.0], "world": [0.0, 1.0]})
    client.drop_last_from_batches = True
    config = make_config(tmp_path, top_k=1, similarity_floor=0.9)  # would normally drop a lot

    kept, result = await filter_by_relevance(client, "q", docs, config, safety_margin=0.85)

    assert len(kept) == 2  # nothing dropped -- fails safe instead of misassigning scores
    assert result.scores == []


@pytest.mark.asyncio
async def test_cache_invalidated_when_overlap_tokens_changes(tmp_path: Path):
    doc = make_doc("a.txt", "hello world")
    vectors = {"q": [1.0, 0.0], "hello world": [1.0, 0.0]}
    client = FakeEmbeddingClient(vectors)

    config_a = make_config(tmp_path, top_k=None, similarity_floor=0.0, overlap_tokens=10)
    await filter_by_relevance(client, "q", [doc], config_a, safety_margin=0.85)
    calls_after_first_run = len(client.embed_calls)

    # Same model_id, but a different overlap_tokens changes chunk boundaries --
    # the cache must not silently reuse embeddings computed under the old config.
    config_b = make_config(tmp_path, top_k=None, similarity_floor=0.0, overlap_tokens=20)
    await filter_by_relevance(client, "q", [doc], config_b, safety_margin=0.85)
    calls_after_second_run = len(client.embed_calls)

    assert calls_after_second_run == calls_after_first_run + 2  # question + doc chunk re-embedded
