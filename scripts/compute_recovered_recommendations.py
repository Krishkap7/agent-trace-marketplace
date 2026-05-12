"""Batch-compute "Recoveries from similar failures" for every failed trace.

For every trace with ``has_error = 1`` AND a cached embedding, populate
the ``recovered_recommendations`` column with the top-10 similar traces
that recovered within their own session, plus each match's recovered
events (label + summary + step). See
:mod:`trace_marketplace.enrich.recovered_recommendations` for the
underlying logic.

No API calls. Pure-local vector math + SQLite I/O; the whole batch
finishes in seconds on the slice-10 corpus.

**Ordering matters**: must run AFTER ``scripts/extract_failure_events.py``
because the candidate pool depends on ``failure_events`` being
populated. The script logs a hard warning + exits non-zero when the
candidate pool is empty so an operator doesn't waste time wondering
why every recommendation list is ``[]``.

Usage:

    uv run python scripts/compute_recovered_recommendations.py
    uv run python scripts/compute_recovered_recommendations.py --db /path/to/marketplace.db
    uv run python scripts/compute_recovered_recommendations.py --limit 30   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

from trace_marketplace.enrich.recovered_recommendations import (
    DEFAULT_TOP_K,
    compute_recommendations,
    count_qualifying_candidates,
    select_failed_target_ids,
)
from trace_marketplace.storage.db import DEFAULT_DB_PATH, connect, init_schema

log = logging.getLogger("compute_recovered_recommendations")

PROGRESS_EVERY: int = 50
"""Print a progress line every N traces. Set high because the batch
finishes in seconds; we just want a heartbeat in case something hangs."""


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional ceiling on number of failed traces processed. "
            "Useful for a quick smoke test before running the full corpus."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            f"How many recommendations to surface per failed trace. "
            f"Defaults to {DEFAULT_TOP_K}; the detail view panel was "
            f"designed around this number."
        ),
    )
    return parser.parse_args(argv)


def _persist(conn: sqlite3.Connection, trace_id: str, payload: list[dict]) -> None:
    conn.execute(
        "UPDATE traces SET recovered_recommendations = ? WHERE id = ?",
        (json.dumps(payload), trace_id),
    )


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

        # Pre-flight: if no candidate trace has any recovered events,
        # every recommendation list will be empty. Warn loudly + exit
        # non-zero so an operator gets a clear signal that
        # ``extract_failure_events.py`` hasn't been run yet.
        qualifying = count_qualifying_candidates(conn)
        if qualifying == 0:
            log.error(
                "No traces in the corpus have a recovered failure event. "
                "Run scripts/extract_failure_events.py first; "
                "without that, every recommendation list will be empty."
            )
            return 2
        log.info(
            "Candidate pool size: %d traces with at least one recovered event",
            qualifying,
        )

        targets = select_failed_target_ids(conn)
        if args.limit is not None:
            targets = targets[: max(0, int(args.limit))]
        if not targets:
            log.info("No has_error=1 traces with embeddings; nothing to do.")
            return 0

        log.info(
            "Computing recommendations for %d failed traces (top_k=%d)...",
            len(targets),
            args.top_k,
        )

        start = time.perf_counter()
        completed = 0
        non_empty = 0
        for trace_id in targets:
            payload = compute_recommendations(conn, trace_id, top_k=args.top_k)
            _persist(conn, trace_id, payload)
            completed += 1
            if payload:
                non_empty += 1
            if completed % PROGRESS_EVERY == 0:
                elapsed = time.perf_counter() - start
                rate = completed / elapsed if elapsed > 0 else 0
                log.info(
                    "  [%d/%d] %.1f traces/s (%d non-empty so far)",
                    completed,
                    len(targets),
                    rate,
                    non_empty,
                )
        conn.commit()
        elapsed = time.perf_counter() - start
        log.info(
            "Done. %d traces processed in %.2fs (%.1f traces/s); "
            "%d non-empty / %d empty recommendation lists.",
            completed,
            elapsed,
            (completed / elapsed) if elapsed > 0 else 0,
            non_empty,
            completed - non_empty,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
