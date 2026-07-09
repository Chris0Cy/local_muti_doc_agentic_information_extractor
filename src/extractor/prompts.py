"""Prompt templates for worker and judge models."""

from __future__ import annotations

from extractor.models import Chunk, WorkerResult

WORKER_SYSTEM_PROMPT = """\
You are a document analysis assistant. You are given the text of ONE document \
(or one chunk of a larger document) and a user's question.

Rules:
- Use ONLY the document text below. Do not use outside knowledge.
- If the document does NOT contain information relevant to the question, respond with \
found_relevant_info=false and answer_excerpt=null.
- If it DOES contain relevant information, you MUST set found_relevant_info=true AND \
you MUST fill answer_excerpt with the actual relevant sentence(s) copied from the document. \
Never set found_relevant_info to true while leaving answer_excerpt empty or null. Never invent \
an answer that isn't in the document text.
- Respond with ONLY a single JSON object, no prose before or after, matching exactly this shape:
{"found_relevant_info": true or false, "answer_excerpt": string or null, "confidence": "high" or "medium" or "low"}
"""


def worker_user_prompt(question: str, chunk: Chunk) -> str:
    return (
        f"Document: {chunk.doc_path.name} (chunk {chunk.chunk_index + 1}/{chunk.total_chunks})\n\n"
        f"Question: {question}\n\n"
        "---DOCUMENT TEXT START---\n"
        f"{chunk.text}\n"
        "---DOCUMENT TEXT END---"
    )


JUDGE_SYSTEM_PROMPT = """\
You are a synthesis judge. You receive a user's question and independent findings \
extracted from multiple documents/chunks by other models.

Rules:
- Ignore any finding where found_relevant_info is false.
- Combine the remaining findings into ONE coherent final answer in markdown.
- For every claim, cite the source filename(s) it came from, e.g. "(source: report_q3.pdf)". \
If multiple documents corroborate a fact, cite all of them.
- If NO finding was relevant, state clearly that the answer could not be found in the \
provided documents. Do not fabricate an answer.
"""


def judge_user_prompt(question: str, results: list[WorkerResult]) -> str:
    relevant = [r for r in results if r.found_relevant_info and r.answer_excerpt]
    if not relevant:
        findings_block = "(no worker found any relevant information)"
    else:
        findings_block = "\n".join(
            f"[{r.doc_path.name}#chunk{r.chunk_index + 1}/{r.total_chunks}] "
            f"(confidence: {r.confidence or 'unknown'}): {r.answer_excerpt}"
            for r in relevant
        )
    return f"Question: {question}\n\nFindings:\n{findings_block}"
