#!/usr/bin/env python3
"""Backfill the study ledger from the timekeeper's own focus-block rows.

`focus_history.jsonl` (jarvis/focus.py) only started being written when the
ledger shipped, but every block a focus session ever ran left a durable
trace before that: a silent timer labelled "focus: block N" that reached
state='done' with a `fired_at`, in `timekeeper.db`, which has no prune or
DELETE path. This script reads those rows and writes the ones the ledger
does not already cover as one-block sessions, so "how much did I study
this week" is honest about the weeks before the feature existed.

Idempotent. A block that falls inside an existing ledger session's window
is the same block seen twice and is skipped, so running it again -- or
after the ledger has grown -- adds nothing. The block's length comes from
(due - created) on the row itself, which is exactly what the session asked
for, so no block_min is guessed.

Nothing in timekeeper.db is written, opened for write, or deleted.

Usage:
    ~/vss_env/bin/python scripts/backfill_focus_history.py --dry-run
    ~/vss_env/bin/python scripts/backfill_focus_history.py
    ~/vss_env/bin/python scripts/backfill_focus_history.py \\
        --memory-dir /tmp/somewhere      # a copy, not the live one
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis import focus as focus_mod          # noqa: E402
from jarvis.config import PATHS                # noqa: E402

STATE_NAME = "focus_session.json"
DB_NAME = "timekeeper.db"


def orphan_blocks(state_path: Path, db_path: Path) -> list[dict]:
    """Timekeeper blocks no ledger row already covers, oldest first."""
    ledger = focus_mod.read_history(focus_mod.history_path(state_path))
    windows = []
    for row in ledger:
        try:
            started = float(row.get("started") or 0.0)
        except (TypeError, ValueError):
            continue
        if started > 0:
            ended = float(row.get("ended") or started)
            windows.append((started, max(ended, started)))
    out = [b for b in focus_mod.timekeeper_blocks(db_path)
           if not any(a <= float(b["when"]) <= z for a, z in windows)]
    out.sort(key=lambda b: float(b["when"]))
    return out


def as_rows(blocks: list[dict]) -> list[dict]:
    """One ledger row per orphan block. `started` is the block's start
    (fired_at minus its length), which is what keys the dedupe, and
    `label` says where it came from -- the original session's label is not
    recoverable from a timer row."""
    rows = []
    for blk in blocks:
        when = float(blk["when"])
        minutes = int(blk.get("minutes") or 0)
        started = when - minutes * 60
        rows.append({"date": datetime.fromtimestamp(started).date().isoformat(),
                     "started": started, "ended": when, "label": "",
                     "blocks": 1, "block_min": minutes, "source": "backfill"})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--memory-dir", default=str(PATHS.MEMORY_DIR),
                    help="where focus_session.json and timekeeper.db live")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    memory = Path(args.memory_dir).expanduser()
    state, db = memory / STATE_NAME, memory / DB_NAME
    if not db.exists():
        print(f"no timekeeper store at {db}; nothing to backfill")
        return 0

    blocks = orphan_blocks(state, db)
    rows = as_rows(blocks)
    if not rows:
        print("the ledger already covers every focus block on file")
        return 0

    days = sorted({r["date"] for r in rows})
    minutes = sum(int(r["block_min"]) for r in rows)
    print(f"{len(rows)} block(s), {minutes} minute(s), "
          f"{days[0]} to {days[-1]} across {len(days)} day(s)")
    path = focus_mod.history_path(state)
    if args.dry_run:
        print(f"dry run: would append to {path}")
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"appended to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
