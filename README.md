# local_muti_doc_agentic_information_extractor

A local, agentic, multi-document information extractor. Ask a question, point it at a
folder of mixed-format documents, and get one synthesized answer with source citations
— entirely offline, backed by local Small Language Models served through
[LM Studio](https://lmstudio.ai/).

## How it works

1. Each document is scanned and its text extracted (PDF, Word, PowerPoint, Excel, CSV,
   HTML, email, Markdown/text, and a generic fallback for anything else).
2. Each document is sized (token estimate) and assigned to the smallest configured
   model tier whose context window it fits in. Oversized documents are chunked and
   pushed through the largest tier instead.
3. Every document/chunk is sent **in parallel** to its assigned model with the user's
   question — a "worker" call that returns a candidate answer or "not found."
4. A **judge** model reads all worker findings and produces one final answer, citing
   which source file(s) each part came from.

## Setup

1. Install [LM Studio](https://lmstudio.ai/), download a model family in a few sizes
   (e.g. the `qwen2.5-instruct` family at 0.5b/1.5b/3b/7b), and load them.
2. In LM Studio's **Developer** tab, click **Start Server** (default:
   `http://localhost:1234`).
3. Edit [`config/default.yaml`](config/default.yaml) so each tier's `model_id` matches
   the exact model identifiers you have loaded (check via `GET /v1/models` or the
   `list-models` command below) and each tier's `context_window_tokens` matches the
   context length you configured when loading that model.
4. Create a Python 3.11+ environment and install the project:

   ```
   pip install -e ".[dev]"
   ```

## Usage

```
# Confirm LM Studio is reachable and see what's loaded
python -m extractor list-models

# Sanity-check document sizing/tier assignment with no LLM calls
python -m extractor inspect --folder ./docs

# Ask a question against a single document (quick smoke test)
python -m extractor ask-one --file ./docs/report.pdf --question "What was Q3 revenue?"

# Ask a question across an entire folder
python -m extractor ask "How did revenue change in Q3?" --folder ./docs
```

## Demo

[`demo/`](demo/) contains a ready-to-run demo for showing this off to someone
non-technical (e.g. a manager): a fictional company's Q3 business review spread
across 6 documents in 6 different formats (PDF, Word, Excel, email, Markdown,
plain text), with facts deliberately scattered and corroborated across formats,
plus one irrelevant document to prove the tool doesn't force in noise.

Run the whole thing with one command:

```
python demo/run_demo.py          # paced: press Enter between each step
python demo/run_demo.py --auto   # runs straight through, no pauses
```

It confirms LM Studio is reachable, runs `inspect`, then asks 5 questions —
each demonstrating a different capability (cross-format corroboration,
pulling numbers from a PDF and a spreadsheet, reading an email, finding a
detail buried mid-document) — finishing with a "trap" question that has no
answer anywhere in the documents, to show the tool reports "not found" rather
than fabricating one.

Other files in `demo/`:
- [`demo/DEMO_SCRIPT.md`](demo/DEMO_SCRIPT.md) — the same walkthrough with talking
  points for each step, worth reading before presenting live.
- [`demo/demo_config.yaml`](demo/demo_config.yaml) — a config with a higher
  quality floor (3b/7b tiers only) than `config/default.yaml`; testing showed
  the smallest tiers are noticeably less reliable at judging relevance on
  open-ended questions, so the demo uses bigger models even though the demo
  documents are small enough to fit the tiniest tier.
- [`demo/generate_demo_files.py`](demo/generate_demo_files.py) — regenerates
  the documents in `demo/docs/` from scratch if you want to tweak the story.

Requires LM Studio running with `qwen2.5-3b-instruct` and `qwen2.5-7b-instruct`
loaded (see [Setup](#setup) above).

## Development

```
pytest                # unit tests (mocked LM Studio client, no server required)
pytest -m live        # live tests against a real, running LM Studio instance
ruff check src tests
ruff format src tests
```
