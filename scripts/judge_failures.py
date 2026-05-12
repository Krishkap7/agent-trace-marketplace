"""LLM-judge pass: assigns a ``failure_label`` to every flagged trace
plus a deterministic 10% sample of clean successes.

Selection set (run on traces where ``failure_label IS NULL``):

* All ``has_error = 1`` rows -- the high-precision yield from the
  rule-based pass + the SWE-agent ground-truth failures.
* A reproducible 10% sample of ``has_error = 0`` rows (seed 42), so
  the validation set has SOME positive-class coverage and the UI's
  ``failure_label`` filter isn't a dead control for success traces.

The seed-and-sample-down approach trades exhaustive coverage for cost
discipline: at ~$5 total spend on Sonnet 4.6 we can't afford to judge
every clean SWE-agent trace, and randomly skipping is the cleanest way
to put a confidence interval on the success-class label distribution.

Usage:

    # Show what would be called without spending anything:
    uv run python scripts/judge_failures.py --dry-run

    # Actually call the API:
    export ANTHROPIC_API_KEY=sk-ant-...
    uv run python scripts/judge_failures.py
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sqlite3
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from trace_marketplace.enrich.failure_judge import (
    INPUT_PRICE_PER_MTOK,
    JUDGE_MODEL,
    OUTPUT_PRICE_PER_MTOK,
    JudgeError,
    JudgeResult,
    estimate_cost_usd,
    judge_trace,
    require_api_key,
)
from trace_marketplace.storage.db import DEFAULT_DB_PATH, connect, init_schema

log = logging.getLogger("judge_failures")

SUCCESS_SAMPLE_RATE: float = 0.10
"""10% of clean-success traces also get judged. Numbers in 0.0-1.0."""

SUCCESS_SAMPLE_SEED: int = 42
"""Pinned PRNG seed so re-running the script picks the same sample.
Tunable but ideally never changed -- the comparison-across-runs
invariant is the whole reason for sampling deterministically."""

PROGRESS_EVERY: int = 25
HARD_CAP_TRACES: int = 500

DEFAULT_WORKERS: int = 8
"""How many traces to judge in parallel by default.

Eight is the empirically-derived sweet spot for the current corpus
size: large enough to hide a single 529 retry burst (~6-8 s of dead
time) behind other in-flight requests, small enough that we stay
under Anthropic Tier-2 throughput caps with comfortable headroom.
Power users can crank to 16-32 via ``--workers`` when the provider
is healthy; the SDK and our own ``judge_trace`` are thread-safe."""

MAX_WORKERS: int = 32
"""Hard ceiling on ``--workers``. The judge tops out around this
many threads even on the largest tier because each call already
holds an HTTP connection -- past 32 we just buy contention without
buying throughput."""
"""When the selection set exceeds this we make the user confirm
interactively. Keeps a bug that 10x's the call count from spending
$50 silently."""

# Token estimates for the dry-run cost projection.
# v1 prompt used ~80 output tokens (one sentence).
# v2 (2-3 sentences with evidence) was projected at ~200.
# v3 (rich 4-6 sentence shop-window paragraph) targets ~500 output
# tokens: 150-220 words * ~1.3 tokens/word + small slack, with a
# JUDGE_MAX_TOKENS cap of 900 so any single outlier stays bounded.
DRY_RUN_INPUT_TOKEN_ESTIMATE: int = 5_000
DRY_RUN_OUTPUT_TOKEN_ESTIMATE: int = 500


@dataclass(frozen=True)
class _Row:
    trace_id: str
    atif: str
    has_error: int | None
    failure_signals: str | None


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Count what would be judged + estimate cost; no API calls. "
            "Always run this first to confirm the call count is sane."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional ceiling on number of API calls (after selection). "
            "Useful for a small 'tip-of-the-iceberg' real-spend run."
        ),
    )
    parser.add_argument(
        "--max-spend-usd",
        type=float,
        default=10.0,
        help=(
            "Hard kill-switch: abort the loop once running spend exceeds "
            "this many dollars. Defaults to the slice 4 plan's $10 cap."
        ),
    )
    parser.add_argument(
        "--re-judge",
        action="store_true",
        help=(
            "Also include traces that already have a failure_label. "
            "Default behaviour skips them so reruns are resumable; use "
            "this flag when the judge prompt or schema has changed and "
            "you want to refresh the persisted reasonings against the "
            "*same* set of traces that were judged on the previous run."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            f"Number of parallel judge calls. Defaults to "
            f"{DEFAULT_WORKERS}; bumping to 16-32 cuts wall-clock "
            f"~linearly when Anthropic is healthy. Capped at "
            f"{MAX_WORKERS}."
        ),
    )
    return parser.parse_args(argv)


def _select_targets(conn: sqlite3.Connection, *, re_judge: bool = False) -> list[_Row]:
    """Build the dry-run-stable selection set (deterministic order).

    When ``re_judge`` is False (default) we only consider traces whose
    ``failure_label`` column is still NULL -- normal resumable runs.
    When True, we relax the gate and re-evaluate every trace that has
    parseable ATIF; this is used to refresh persisted reasonings after
    a prompt-schema change. The success-sample logic still applies in
    both modes so the selection set is reproducible across calls.
    """
    failures: list[_Row] = []
    successes: list[_Row] = []
    sql = """
        SELECT id, atif, has_error, failure_signals
          FROM traces
         WHERE atif IS NOT NULL
    """
    if not re_judge:
        sql += " AND failure_label IS NULL"
    sql += " ORDER BY id ASC"
    cur = conn.execute(sql)
    for row in cur:
        rec = _Row(
            trace_id=row["id"],
            atif=row["atif"],
            has_error=row["has_error"],
            failure_signals=row["failure_signals"],
        )
        if rec.has_error == 1:
            failures.append(rec)
        elif rec.has_error == 0:
            successes.append(rec)
        # has_error IS NULL: skip. Slice 4's detect pass should always
        # set has_error in {0, 1} for any trace with parsed ATIF; if a
        # trace got here with NULL it's a data-quality issue, not the
        # judge's job to backfill.
    # Deterministic 10% sample of successes via a freshly-seeded RNG.
    rng = random.Random(SUCCESS_SAMPLE_SEED)
    sample_size = int(round(len(successes) * SUCCESS_SAMPLE_RATE))
    sampled_successes = rng.sample(successes, sample_size) if sample_size else []
    # Stable final ordering: failures first (sorted by id), then sampled
    # successes (sorted by id) -- the rng sample doesn't preserve order.
    sampled_successes.sort(key=lambda r: r.trace_id)
    return [*failures, *sampled_successes]


def _signals_dict(row: _Row) -> dict[str, bool]:
    if not row.failure_signals:
        return {}
    try:
        decoded = json.loads(row.failure_signals)
    except json.JSONDecodeError:
        return {}
    return {k: bool(v) for k, v in decoded.items()} if isinstance(decoded, dict) else {}


def _confirm_or_abort(selection_size: int) -> None:
    """Hard cap with interactive ack. Keeps a bad query from quietly
    becoming a $50+ run."""
    if selection_size <= HARD_CAP_TRACES:
        return
    print(
        f"\nWARNING: selection set is {selection_size} traces, above the "
        f"safety cap of {HARD_CAP_TRACES}."
    )
    response = input("Type 'yes' to continue, anything else to abort: ").strip().lower()
    if response != "yes":
        raise SystemExit("Aborted: selection set too large; review the query.")


def _run_dry(targets: list[_Row]) -> int:
    """Print a dry-run summary including an order-of-magnitude cost
    estimate; never touch the API."""
    n_failures = sum(1 for r in targets if r.has_error == 1)
    n_successes = sum(1 for r in targets if r.has_error == 0)
    total_in = len(targets) * DRY_RUN_INPUT_TOKEN_ESTIMATE
    total_out = len(targets) * DRY_RUN_OUTPUT_TOKEN_ESTIMATE
    est_cost = estimate_cost_usd(total_in, total_out)
    print("=" * 60)
    print("DRY RUN (no API calls)")
    print(f"Model:                    {JUDGE_MODEL}")
    print(f"Total traces to judge:    {len(targets)}")
    print(f"  has_error = 1:          {n_failures}")
    print(f"  has_error = 0 sampled:  {n_successes}")
    print(
        f"Per-call estimate:        {DRY_RUN_INPUT_TOKEN_ESTIMATE} in / "
        f"{DRY_RUN_OUTPUT_TOKEN_ESTIMATE} out tokens"
    )
    print(f"Aggregate token estimate: {total_in:>10,} in / {total_out:>10,} out")
    print(
        f"Cost estimate:            ${est_cost:.2f}  "
        f"(input ${INPUT_PRICE_PER_MTOK}/Mtok, output ${OUTPUT_PRICE_PER_MTOK}/Mtok)"
    )
    print("=" * 60)
    return 0


def _persist_one(
    conn: sqlite3.Connection, trace_id: str, label: str, reasoning: str
) -> None:
    conn.execute(
        "UPDATE traces SET failure_label = ?, failure_label_reasoning = ? WHERE id = ?",
        (label, reasoning, trace_id),
    )


@dataclass(frozen=True)
class _WorkerOutcome:
    """Result of one ``_judge_one`` call. Either ``result`` is set
    (happy path) or one of the error fields is. We classify the
    error type in the worker so the main thread can preserve the
    serial version's log-level distinction (``JudgeError`` -> warning,
    everything else -> error)."""

    row: _Row
    result: JudgeResult | None = None
    judge_error: JudgeError | None = None
    bad_atif: bool = False
    unexpected_error: Exception | None = None


def _judge_one(row: _Row, *, client: Any) -> _WorkerOutcome:
    """Worker body: pure compute + Anthropic I/O, no DB access.

    Running in a ThreadPoolExecutor. SQLite would happily corrupt
    itself across threads on a shared connection, so we deliberately
    keep this side of the boundary read-only: parse ATIF, call the
    judge, return everything the main thread needs to persist.
    """
    try:
        traj = json.loads(row.atif)
    except json.JSONDecodeError:
        return _WorkerOutcome(row=row, bad_atif=True)
    signals = _signals_dict(row)
    try:
        result = judge_trace(traj, signals, client=client)
        return _WorkerOutcome(row=row, result=result)
    except JudgeError as exc:
        return _WorkerOutcome(row=row, judge_error=exc)
    except Exception as exc:  # noqa: BLE001 -- network / SDK errors
        return _WorkerOutcome(row=row, unexpected_error=exc)


def _run_live(
    conn: sqlite3.Connection,
    targets: list[_Row],
    *,
    max_spend_usd: float,
    limit: int | None,
    workers: int,
) -> int:
    """Judge ``targets`` in parallel, persist results on the main thread.

    Concurrency model:

    * Workers (``ThreadPoolExecutor``) call :func:`judge_trace` only.
      The Anthropic SDK + httpx are thread-safe so we share one
      ``Anthropic`` client across all workers (saves TLS handshakes).
    * Main thread drains ``as_completed`` and is the *only* thread
      that touches the SQLite connection. SQLite connections are not
      thread-safe by default and we don't want to babysit a lock.
    * Cost cap is enforced after each completed task. When we cross
      the threshold we cancel still-pending futures and stop
      collecting; already-running workers finish (Python can't
      interrupt threads mid-flight), but their results are still
      persisted to avoid wasting the API spend that already happened.
    """
    require_api_key()
    if limit is not None:
        targets = targets[: max(0, int(limit))]
    workers = max(1, min(int(workers), MAX_WORKERS, len(targets) or 1))
    print(
        f"Judging {len(targets)} traces with {JUDGE_MODEL} "
        f"(workers={workers})..."
    )
    print(f"Hard spend cap: ${max_spend_usd:.2f}")

    total_in = 0
    total_out = 0
    successes = 0
    failures = 0
    aborted_for_cost = False
    completed = 0

    shared_client = Anthropic()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures: dict[Future[_WorkerOutcome], _Row] = {
            pool.submit(_judge_one, row, client=shared_client): row
            for row in targets
        }
        try:
            for fut in as_completed(futures):
                completed += 1
                outcome = fut.result()
                row = outcome.row

                if outcome.bad_atif:
                    log.warning(
                        "Trace %s has malformed ATIF; skipping", row.trace_id
                    )
                    failures += 1
                elif outcome.judge_error is not None:
                    log.warning(
                        "Judge failed for %s: %s",
                        row.trace_id,
                        outcome.judge_error,
                    )
                    failures += 1
                elif outcome.unexpected_error is not None:
                    log.error(
                        "Unexpected error judging %s: %s",
                        row.trace_id,
                        outcome.unexpected_error,
                    )
                    failures += 1
                else:
                    assert outcome.result is not None  # noqa: S101
                    _persist_one(
                        conn,
                        row.trace_id,
                        outcome.result.label.value,
                        outcome.result.reasoning,
                    )
                    conn.commit()
                    total_in += outcome.result.usage.get("input_tokens", 0)
                    total_out += outcome.result.usage.get("output_tokens", 0)
                    successes += 1

                if (
                    completed % PROGRESS_EVERY == 0
                    or completed == len(targets)
                ):
                    cost_so_far = estimate_cost_usd(total_in, total_out)
                    print(
                        f"  [{completed:>4}/{len(targets)}] "
                        f"tokens={total_in:>7}/{total_out:>5}  "
                        f"cost=${cost_so_far:.2f}"
                    )

                running_cost = estimate_cost_usd(total_in, total_out)
                if running_cost >= max_spend_usd:
                    print(
                        f"\nABORTING: running spend ${running_cost:.2f} "
                        f"hit hard cap ${max_spend_usd:.2f}. "
                        f"{completed}/{len(targets)} traces processed."
                    )
                    aborted_for_cost = True
                    # Cancel pending futures so we don't keep spending.
                    # Already-running workers finish (Python threads
                    # can't be killed); their outcomes are silently
                    # discarded by exiting the as_completed loop.
                    for pending in futures:
                        if not pending.done():
                            pending.cancel()
                    break
        except KeyboardInterrupt:
            # Ctrl-C: cancel queued futures, let in-flight ones drain
            # quietly so the DB doesn't end up with half-written rows.
            print("\nInterrupted; cancelling pending judge calls...")
            for pending in futures:
                if not pending.done():
                    pending.cancel()
            raise

    final_cost = estimate_cost_usd(total_in, total_out)
    print("=" * 60)
    print(f"Judged successfully:  {successes}")
    print(f"Skipped (errors):     {failures}")
    print(f"Total input tokens:   {total_in:,}")
    print(f"Total output tokens:  {total_out:,}")
    print(f"Estimated cost:       ${final_cost:.4f}")
    if aborted_for_cost:
        print("Run was aborted by max-spend cap.")
    print("=" * 60)
    return 0 if not aborted_for_cost else 2


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
        targets = _select_targets(conn, re_judge=args.re_judge)
        if not targets:
            msg = (
                "No traces matched the selection set."
                if args.re_judge
                else "No traces need judging (failure_label already populated)."
            )
            print(msg)
            return 0
        if args.re_judge:
            print(
                f"--re-judge: ignoring failure_label gate; will overwrite "
                f"reasoning + label for {len(targets)} already-judged traces."
            )
        if args.dry_run:
            return _run_dry(targets)
        _confirm_or_abort(len(targets))
        return _run_live(
            conn,
            targets,
            max_spend_usd=args.max_spend_usd,
            limit=args.limit,
            workers=args.workers,
        )


if __name__ == "__main__":
    raise SystemExit(main())
