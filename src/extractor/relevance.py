"""Opt-in embedding-based relevance pre-filter.

Ranks documents by cosine similarity between the question and each
document's chunk embeddings (max across chunks), keeping only documents that
are both within the configured top_k and clear the similarity_floor. Fails
safe: if embedding calls error out, every document is kept unchanged rather
than the run silently looking like nothing was relevant.
"""

from __future__ import annotations

import math
from pathlib import Path

from extractor import chunking
from extractor.embedding_cache import load_cache, save_cache
from extractor.lmstudio_client import (
    LMStudioClientProtocol,
    LMStudioRequestError,
    LMStudioUnavailable,
)
from extractor.models import (
    DocumentRelevanceScore,
    ExtractedDocument,
    RelevanceFilterConfig,
    RelevanceFilterResult,
)

_BATCH_SIZE = 32


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _embed_batched(
    client: LMStudioClientProtocol, model_id: str, texts: list[str]
) -> list[list[float]]:
    results: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        results.extend(await client.embeddings(model_id, batch))
    return results


async def filter_by_relevance(
    client: LMStudioClientProtocol,
    question: str,
    extracted: list[ExtractedDocument],
    config: RelevanceFilterConfig,
    safety_margin: float,
) -> tuple[list[ExtractedDocument], RelevanceFilterResult]:
    if not extracted:
        return extracted, RelevanceFilterResult(enabled=True, scores=[])

    cache_path = Path(config.cache_path)
    cache = load_cache(cache_path)
    model_id = config.embedding_model.model_id
    # The cache is invalidated wholesale whenever this fingerprint changes, not
    # just on a raw model_id change -- chunk boundaries (and therefore chunk
    # embeddings) also depend on the embedding tier's budget and overlap, so a
    # tuning change to those without touching model_id must not silently reuse
    # stale chunk embeddings computed with different chunk boundaries.
    cache_key = (
        f"{model_id}:{config.embedding_model.context_window_tokens}:"
        f"{config.embedding_model.reserved_output_tokens}:{config.overlap_tokens}:{safety_margin}"
    )

    doc_stats: dict[Path, tuple[int, int]] = {}
    doc_chunk_texts: dict[Path, list[str]] = {}
    doc_embeddings: dict[Path, list[list[float]]] = {}
    miss_texts: list[str] = []

    for doc in extracted:
        try:
            stat = doc.path.stat()
            mtime_ns, size = stat.st_mtime_ns, stat.st_size
        except OSError:
            mtime_ns, size = 0, 0
        doc_stats[doc.path] = (mtime_ns, size)

        cached = cache.get(doc.path, mtime_ns, size, cache_key)
        if cached is not None:
            doc_embeddings[doc.path] = cached
            continue

        chunks = chunking.split_into_chunks(
            doc.text, config.embedding_model, config.overlap_tokens, safety_margin
        ) or [doc.text]
        doc_chunk_texts[doc.path] = chunks
        miss_texts.extend(chunks)

    try:
        question_embedding = (await client.embeddings(model_id, [question]))[0]
        flat_embeddings = await _embed_batched(client, model_id, miss_texts) if miss_texts else []
    except (LMStudioUnavailable, LMStudioRequestError):
        return extracted, RelevanceFilterResult(enabled=True, scores=[])

    if len(flat_embeddings) != len(miss_texts):
        # The API returned a different number of vectors than requested --
        # slicing by cursor position below would silently misassign
        # embeddings to the wrong documents. Fail safe instead, same as an
        # outright request error.
        return extracted, RelevanceFilterResult(enabled=True, scores=[])

    cursor = 0
    for path, chunks in doc_chunk_texts.items():
        n = len(chunks)
        embeddings_for_doc = flat_embeddings[cursor : cursor + n]
        cursor += n
        doc_embeddings[path] = embeddings_for_doc
        mtime_ns, size = doc_stats[path]
        cache.put(path, mtime_ns, size, cache_key, embeddings_for_doc)

    save_cache(cache_path, cache)

    doc_scores = {
        path: max(_cosine_similarity(question_embedding, emb) for emb in embeddings)
        for path, embeddings in doc_embeddings.items()
    }

    ranked = sorted(extracted, key=lambda d: doc_scores[d.path], reverse=True)
    kept_paths: set[Path] = set()
    for rank, doc in enumerate(ranked):
        score = doc_scores[doc.path]
        within_top_k = config.top_k is None or rank < config.top_k
        clears_floor = config.similarity_floor is None or score >= config.similarity_floor
        if within_top_k and clears_floor:
            kept_paths.add(doc.path)

    scores = [
        DocumentRelevanceScore(
            path=doc.path, score=doc_scores[doc.path], kept=doc.path in kept_paths
        )
        for doc in extracted
    ]
    kept_docs = [doc for doc in extracted if doc.path in kept_paths]

    return kept_docs, RelevanceFilterResult(enabled=True, scores=scores)
