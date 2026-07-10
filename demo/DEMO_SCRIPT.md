# Demo Script: Local Agentic Multi-Document Information Extractor

A ~5 minute walkthrough for demoing this to a manager. The scenario: a fictional
company, **Acme Robotics**, has its Q3 2026 business review scattered across six
files in six different formats. You ask plain-English questions; the tool finds
the answer even when it's split across documents — and honestly says "not found"
when it isn't there.

## Setup (do this before the meeting)

1. Start LM Studio's local server (Developer tab → Start Server) with the
   `qwen2.5-3b-instruct` and `qwen2.5-7b-instruct` models loaded.
2. From the project root, in the `doc-extractor` environment:
   ```
   python demo/generate_demo_files.py   # only needed once, files are already in demo/docs
   ```
3. Confirm connectivity: `python -m extractor list-models`

All commands below use `--config demo/demo_config.yaml`, which sets a higher
quality floor (3B minimum, no 0.5B/1.5B tiers) — see "One honest caveat" below
for why.

## The documents (`demo/docs/`)

| File | Format | Contains |
|---|---|---|
| `financial_summary.pdf` | PDF | Revenue, margins, the SensTech sensor supply risk |
| `engineering_status.docx` | Word | Project status, headcount, supply-chain mitigation plan |
| `sales_pipeline.xlsx` | Excel | Pipeline value by region and stage |
| `board_email.eml` | Email | CEO's note to the board on customer churn |
| `allhands_notes.md` | Markdown | Hiring freeze + one approved exception |
| `office_snack_survey.txt` | Text | Completely unrelated — a snack preference survey |

Open `python -m extractor inspect --folder demo/docs` first to show the manager
each file's format is recognized and sized automatically, with no LLM calls yet.

## The five questions to run live

Run each with:
```
python -m extractor ask "<question>" --folder demo/docs --config demo/demo_config.yaml
```

**1. Cross-format corroboration** — same fact, two file formats independently confirm it:
> What supply chain issue is affecting the Atlas arm, and how is the team mitigating it?

Answer cites both `financial_summary.pdf` (PDF) and `engineering_status.docx` (Word) —
the finance report flags the risk, the engineering memo explains the mitigation.
**Talking point:** each document was read by its own model instance in parallel;
neither model saw the other's content, yet the judge correctly merged them.

**2. Cross-format numeric synthesis:**
> How did revenue and the sales pipeline look this quarter?

Pulls the revenue figure from the PDF and the pipeline totals from the Excel
spreadsheet into one answer.

**3. Single-document extraction from an email:**
> What did the CEO tell the board about customer churn this quarter?

Correctly answers from `board_email.eml` alone — demonstrates email support.

**4. A detail buried in the middle of a document:**
> What exception to the hiring freeze was approved this quarter?

Answers from `allhands_notes.md`, correctly surfacing the one exception buried
in an otherwise unremarkable set of meeting notes.

**5. The "trap" question — nothing in any document answers this:**
> What did the board decide about opening a new office in Berlin?

Answer: *"No relevant information was found in any of the provided documents."*
**This is the most important moment in the demo.** Point out that it did not
guess or make something up — it checked every document and said so honestly.

Across all five questions, `office_snack_survey.txt` is never cited — the
system correctly recognizes it's irrelevant to every business question asked.

## Closing pitch points

- **Fully local / private**: nothing in these documents ever left the machine —
  no cloud API calls. Real fit for financial reports, HR data, legal documents.
- **Multi-format out of the box**: PDF, Word, Excel, email, Markdown, plus a
  generic fallback for anything else.
- **Scales by document size, not a fixed model**: small documents run on
  smaller/faster local models; large documents automatically get chunked and
  routed to a bigger-context model — tunable in `config/default.yaml`.
- **Won't fabricate answers**: question 5 is the proof — it reports "not found"
  rather than guessing.

## One honest caveat (say this if asked "does it always work this well?")

The very smallest models (0.5B/1.5B parameters) are fast but noticeably less
reliable at judging what's relevant on open-ended questions — in testing they
occasionally missed real answers or flagged the snack survey as relevant. That's
why `demo/demo_config.yaml` sets a 3B-parameter floor rather than using the
absolute smallest tier for everything. It's a config change, not a code change —
this is a real, tunable quality/speed tradeoff, not a hidden limitation.
