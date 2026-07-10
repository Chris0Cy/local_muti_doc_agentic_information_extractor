"""Runs the full manager demo end-to-end as a single script.

Requires LM Studio's local server running with qwen2.5-3b-instruct and
qwen2.5-7b-instruct loaded (see demo/DEMO_SCRIPT.md for setup).

Usage:
    python demo/run_demo.py            # paced: press Enter between each step
    python demo/run_demo.py --auto     # runs straight through, no pauses
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).parent
DOCS_DIR = DEMO_DIR / "docs"
CONFIG_PATH = DEMO_DIR / "demo_config.yaml"

STEPS = [
    {
        "title": "Step 1: Confirm LM Studio is reachable",
        "talk": "Quick sanity check before we start -- confirms LM Studio's server is "
        "up and the right models are loaded.",
        "cmd": ["list-models"],
    },
    {
        "title": "Step 2: Inspect the folder (no LLM calls yet)",
        "talk": "It just read a PDF, an email, a Word doc, and an Excel file -- "
        "automatically, by content, not by guessing from the file extension -- and "
        "sized each one to see which model tier it needs. No AI model has been "
        "called yet; this is pure parsing.",
        "cmd": ["inspect", "--folder", str(DOCS_DIR), "--config", str(CONFIG_PATH)],
    },
    {
        "title": "Step 3: Cross-format corroboration",
        "talk": "The finance report flagged this risk; a completely separate document "
        "-- the engineering memo -- explained the mitigation plan. Neither model saw "
        "the other's content. The judge merged them and cited both sources.",
        "cmd": [
            "ask",
            "What supply chain issue is affecting the Atlas arm, and how is the team mitigating it?",
            "--folder",
            str(DOCS_DIR),
            "--config",
            str(CONFIG_PATH),
        ],
    },
    {
        "title": "Step 4: Numbers from two different formats",
        "talk": "One number came from a PDF, the other from an Excel spreadsheet -- "
        "format doesn't matter.",
        "cmd": [
            "ask",
            "How did revenue and the sales pipeline look this quarter?",
            "--folder",
            str(DOCS_DIR),
            "--config",
            str(CONFIG_PATH),
        ],
    },
    {
        "title": "Step 5: Reading an email",
        "talk": "Straight out of an email thread, no special handling needed.",
        "cmd": [
            "ask",
            "What did the CEO tell the board about customer churn this quarter?",
            "--folder",
            str(DOCS_DIR),
            "--config",
            str(CONFIG_PATH),
        ],
    },
    {
        "title": "Step 6: A detail buried mid-document",
        "talk": "That exception was one sentence buried in the middle of an otherwise "
        "unremarkable set of meeting notes -- the kind of detail a person skimming "
        "six documents might actually miss.",
        "cmd": [
            "ask",
            "What exception to the hiring freeze was approved this quarter?",
            "--folder",
            str(DOCS_DIR),
            "--config",
            str(CONFIG_PATH),
        ],
    },
    {
        "title": "Step 7: The trap question (this is the one that matters most)",
        "talk": "Nothing in any of the 6 documents mentions Berlin or a new office. "
        "Watch: instead of inventing a plausible-sounding answer, it checks everything "
        "and says so. That's the single biggest thing to point out to a skeptical "
        "audience -- it does not guess.",
        "cmd": [
            "ask",
            "What did the board decide about opening a new office in Berlin?",
            "--folder",
            str(DOCS_DIR),
            "--config",
            str(CONFIG_PATH),
        ],
    },
]


def run_step(step: dict, python_exe: str, auto: bool) -> None:
    # flush=True: without it, this script's own print() output can get
    # buffered behind the subprocess's output (which flushes independently),
    # so headers/talking points appear in the wrong order -- especially when
    # stdout isn't a live terminal, e.g. piped to `tee` for a saved log.
    print("\n" + "=" * 70, flush=True)
    print(step["title"], flush=True)
    print("=" * 70, flush=True)
    print(f"\n{step['talk']}\n", flush=True)
    if not auto:
        input("Press Enter to run this step...")
    result = subprocess.run([python_exe, "-m", "extractor", *step["cmd"]])
    if result.returncode != 0:
        print(f"\n[!] Step failed (exit code {result.returncode}). Is LM Studio running?")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full manager demo end-to-end.")
    parser.add_argument(
        "--auto", action="store_true", help="Run straight through without pausing between steps."
    )
    args = parser.parse_args()

    print("Local Agentic Multi-Document Information Extractor -- Manager Demo", flush=True)
    print(f"Demo documents: {DOCS_DIR}", flush=True)
    print(f"Demo config:    {CONFIG_PATH}", flush=True)

    if not args.auto:
        input("\nPress Enter to begin...")

    for step in STEPS:
        run_step(step, sys.executable, args.auto)

    print("\n" + "=" * 70, flush=True)
    print("Demo complete.", flush=True)
    print("=" * 70)


if __name__ == "__main__":
    main()
