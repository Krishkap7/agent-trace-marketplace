"""Rule-based failure-detection pass.

Walks every trace in the DB that has a parsed ATIF blob but no
``failure_signals`` yet, runs each signal in
:mod:`trace_marketplace.enrich.failure_rules`, and writes the result
back as JSON. ``has_error`` is filled from ``any_signal_fired`` only
when it was previously NULL -- benchmark ground truth (SWE-agent
``target``) is never overwritten thanks to ``COALESCE``.

Idempotent: re-running picks up only newly-ingested or previously-NULL
rows, so it's safe to schedule.

Usage:
    uv run python scripts/detect_failures.py [--db PATH] [--force]
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from trace_marketplace.enrich.failure_rules import (
    any_signal_fired,
    persist_signals_for_trace,
)
from trace_marketplace.storage.db import DEFAULT_DB_PATH, connect, init_schema

log = logging.getLogger("detect_failures")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recompute signals for rows that already have failure_signals. "
            "has_error is still preserved via COALESCE so this is safe to "
            "re-run after tuning thresholds."
        ),
    )
    return parser.parse_args(argv)


def _select_targets(conn: sqlite3.Connection, *, force: bool) -> list[tuple[str, str]]:
    """Return ``(id, atif)`` tuples to process.

    With ``--force`` we re-process every row that has an ATIF blob,
    regardless of whether ``failure_signals`` already exists. Default
    behaviour skips rows that have already been annotated.
    """
    if force:
        sql = "SELECT id, atif FROM traces WHERE atif IS NOT NULL"
    else:
        sql = (
            "SELECT id, atif FROM traces "
            "WHERE atif IS NOT NULL AND failure_signals IS NULL"
        )
    return [(row["id"], row["atif"]) for row in conn.execute(sql)]


def _process_one(
    conn: sqlite3.Connection, trace_id: str, atif_blob: str
) -> dict[str, bool] | None:
    """Compute signals for one trace; persist them.

    Thin wrapper over :func:`persist_signals_for_trace` -- kept as a
    module-local function so the per-row commit cadence + counter
    bookkeeping in :func:`main` reads cleanly. The shared helper lives
    in :mod:`trace_marketplace.enrich.failure_rules` so the upload view
    (slice 5) and this batch script can't drift on the UPDATE SQL.
    """
    return persist_signals_for_trace(conn, trace_id, atif_blob)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if not args.db.exists():
        log.error("Database not found at %s", args.db)
        return 1

    with connect(args.db) as conn:
        init_schema(conn)
        targets = _select_targets(conn, force=args.force)
        if not targets:
            print("No traces need failure detection (use --force to recompute).")
            return 0
        print(f"Running rule-based detection on {len(targets)} traces...")

        per_signal: Counter[str] = Counter()
        any_fired = 0
        flipped_has_error = 0
        processed = 0

        for trace_id, atif_blob in targets:
            # Probe has_error BEFORE the update so we can count flips.
            row = conn.execute(
                "SELECT has_error FROM traces WHERE id = ?", (trace_id,)
            ).fetchone()
            prior_has_error = row["has_error"] if row else None

            signals = _process_one(conn, trace_id, atif_blob)
            if signals is None:
                continue
            processed += 1
            for name, fired in signals.items():
                if fired:
                    per_signal[name] += 1
            if any_signal_fired(signals):
                any_fired += 1
                if prior_has_error is None:
                    flipped_has_error += 1
            conn.commit()

        print()
        print("=" * 60)
        print(f"Processed:                       {processed}")
        print(f"Any signal fired:                {any_fired}")
        print(f"has_error NULL -> 1 (new flips): {flipped_has_error}")
        print("-" * 60)
        for name in (
            "hit_max_turns",
            "repeated_tool_call_loop",
            "ended_on_error",
            "abandonment",
            "incomplete_session",
        ):
            print(f"  {name:<28} {per_signal[name]}")
        print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
