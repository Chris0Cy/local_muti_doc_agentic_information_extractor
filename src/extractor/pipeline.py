"""Full fan-out/fan-in pipeline: scan -> extract -> size/chunk -> ask workers -> judge."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from extractor import chunking, discovery, sizing
from extractor import extraction as extraction_mod
from extractor import judge as judge_mod
from extractor.extraction.base import ExtractionError
from extractor.lmstudio_client import LMStudioClientProtocol
from extractor.models import AppConfig, Chunk, ExtractedDocument, RunReport
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


async def run(
    question: str, folder: Path, config: AppConfig, client: LMStudioClientProtocol
) -> RunReport:
    t0 = time.monotonic()

    paths = discovery.scan_folder(folder)

    extracted: list[ExtractedDocument] = []
    failed: list[tuple[Path, str]] = []
    for path in paths:
        try:
            extracted.append(extraction_mod.extract_text(path))
        except ExtractionError as e:
            failed.append((path, str(e)))

    await client.health_check()

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
    )
