"""Full fan-out/fan-in pipeline: scan -> extract -> size/chunk -> ask workers -> judge."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from extractor import chunking, discovery, sizing
from extractor import extraction as extraction_mod
from extractor import judge as judge_mod
from extractor import relevance as relevance_mod
from extractor.extraction.base import ExtractionError
from extractor.lmstudio_client import LMStudioClientProtocol
from extractor.models import AppConfig, Chunk, ExtractedDocument, RelevanceFilterResult, RunReport
from extractor.worker import ask_worker


def _build_work_items(extracted: list[ExtractedDocument], config: AppConfig) -> list[Chunk]:
    tiers = config.tiers_ascending
    biggest = tiers[-1]
    work_items: list[Chunk] = []

    for doc in extracted:
        tier = sizing.assign_tier(doc.text, tiers, config.safety_margin)
        if tier is None:
            pieces = chunking.split_into_chunks(
                doc.text, biggest, config.chunk_overlap_tokens, config.safety_margin
            )
            total = len(pieces)
            work_items.extend(
                Chunk(
                    doc_path=doc.path, chunk_index=i, total_chunks=total, text=piece, tier=biggest
                )
                for i, piece in enumerate(pieces)
            )
        else:
            work_items.append(
                Chunk(doc_path=doc.path, chunk_index=0, total_chunks=1, text=doc.text, tier=tier)
            )

    return work_items


async def _extract_one(path: Path) -> tuple[ExtractedDocument | None, tuple[Path, str] | None]:
    try:
        doc = await asyncio.to_thread(extraction_mod.extract_text, path)
        return doc, None
    except ExtractionError as e:
        return None, (path, str(e))


async def run(
    question: str, folder: Path, config: AppConfig, client: LMStudioClientProtocol
) -> RunReport:
    t0 = time.monotonic()

    paths = discovery.scan_folder(folder)

    # Extraction is CPU/IO-bound (PDF parsing, OCR-adjacent work, etc.) and
    # each file is independent, so run them concurrently via a thread pool
    # rather than serially blocking the event loop one file at a time.
    extraction_results = await asyncio.gather(*(_extract_one(p) for p in paths))
    extracted = [doc for doc, _ in extraction_results if doc is not None]
    failed = [failure for _, failure in extraction_results if failure is not None]

    await client.health_check()

    if config.relevance_filter is not None:
        extracted, relevance_result = await relevance_mod.filter_by_relevance(
            client, question, extracted, config.relevance_filter, config.safety_margin
        )
    else:
        relevance_result = RelevanceFilterResult()

    work_items = _build_work_items(extracted, config)

    semaphore = asyncio.Semaphore(config.concurrency_limit)

    async def _bounded(item: Chunk):
        async with semaphore:
            return await ask_worker(client, question, item)

    worker_results = list(await asyncio.gather(*(_bounded(item) for item in work_items)))

    judge_result = await judge_mod.synthesize(client, question, worker_results, config.judge)

    return RunReport(
        question=question,
        documents_scanned=len(paths),
        documents_failed=failed,
        worker_results=worker_results,
        judge_result=judge_result,
        duration_s=time.monotonic() - t0,
        relevance_filter=relevance_result,
    )
