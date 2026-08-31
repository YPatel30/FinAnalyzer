"""Checkpointing for `uv run eval`.

A full run spends real, quota-scarce generation calls (~35 of them across
recall/groundedness/refusal) — and a quota wall hit mid-run previously
meant losing every result already computed, since eval.py built its result
lists purely in memory. This persists each question's result to disk the
moment it's computed, so a rerun skips whatever's already there instead of
re-spending quota recomputing it.

Scoped narrowly to eval.py's own resumability across runs of this one
script — not a general caching layer for ask() in normal use, which stays
explicitly out of scope for Phase 3. Plain JSON, not pickle: the checkpoint
holds only built-in types (str/float/int/bool/list/dict) via each result's
own to-dict/from-dict, so the file stays human-inspectable and doesn't
depend on today's class definitions still matching tomorrow's.
"""

import json
from pathlib import Path

CHECKPOINT_PATH = Path(".cache/eval_checkpoint.json")


def load_checkpoint() -> dict:
    if not CHECKPOINT_PATH.exists():
        return {}
    return json.loads(CHECKPOINT_PATH.read_text())


def save_checkpoint(data: dict) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps(data, indent=2))


class CallCounter:
    """A running total of real (non-cached) API calls made this run,
    printed as each one happens — so progress, and exactly how far a run
    got before hitting a quota wall, is visible live rather than only
    inferred after the fact.
    """

    def __init__(self):
        self.count = 0

    def tick(self, label: str) -> None:
        self.count += 1
        print(f"  [call #{self.count}] {label}")
