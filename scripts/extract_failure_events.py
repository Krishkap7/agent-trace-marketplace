"""Batch Opus 4.7 failure-event extraction.

Selects every trace that has parseable ATIF and is *potentially* a
failure (``has_error=1`` OR ``failure_label IS NOT NULL``), skips rows
that already have ``failure_events`` populated (idempotent re-runs), and
calls :func:`trace_marketplace.enrich.failure_events.extract_events` on
each in a ``ThreadPoolExecutor`` -- the Anthropic SDK / httpx pair is
thread-safe and our judge module shares one client across workers, which
saves the TLS handshake cost on every call.

Cost discipline:

* The default invocation is ``--dry-run``: it counts the selection set,
  projects token usage + USD cost (expected + worst case = expected*1.5),
  and exits without touching the API. Every line is prefixed ``[DRY RUN]``
  so a quick scroll can't be mistaken for a live execution.
* The live path requires the explicit ``--confirm`` flag. We print a
  framed "STARTING LIVE EXECUTION" banner before the first API call so
  the operator can confirm at a glance. Per-trace logs in this mode are
  prefixed ``[LIVE]`` to mirror the dry-run convention.

After each successful extraction we persist:

* ``failure_events``     -- JSON-serialised list[FailureEvent]
* ``failure_label``      -- summary label derived via :func:`summarize_label`
* ``failure_label_source = 'opus_events_derived'``
* ``ultimately_succeeded`` -- via :func:`derive_ultimately_succeeded`,
  preserving any prior SWE-agent backfill.

Usage:

    # Show what would be called without spending anything:
    uv run python scripts/extract_failure_events.py
    uv run python scripts/extract_failure_events.py --dry-run

    # Actually call the API (requires explicit --confirm):
    export ANTHROPIC_API_KEY=sk-ant-...
    uv run python scripts/extract_failure_events.py --confirm
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sqlite3
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anthropic import Anthropic, APIStatusError, RateLimitError

from trace_marketplace.enrich.failure_events import (
    EVENTS_MAX_TOKENS,
    EVENTS_MODEL,
    INPUT_PRICE_PER_MTOK,
    OUTPUT_PRICE_PER_MTOK,
    EventsExtractionError,
    EventsResult,
    derive_ultimately_succeeded,
    estimate_cost_usd,
    extract_events,
    require_api_key,
    summarize_label,
)
from trace_marketplace.storage.db import DEFAULT_DB_PATH, connect, init_schema

log = logging.getLogger("extract_failure_events")

DEFAULT_WORKERS: int = 24
"""Default thread count. Sized for an Anthropic Tier-4 budget where the
RPM ceiling is high enough that 24 in-flight Opus calls don't get
rate-limited as long as we stay polite during retries. The user said
"do parallel process everything and increase your limits a bit higher
than expected", so the default is a bit above the prior slice-4 judge
default (8). Operators can push to ``MAX_WORKERS`` (32) when fleet
health is good."""

MAX_WORKERS: int = 32
"""Hard ceiling on ``--max-workers``. Past 32 each call already holds
an HTTP connection and we start buying contention without buying
throughput; this is the same ceiling the slice-4 judge uses."""

PROGRESS_EVERY: int = 10
"""How often to print a live progress line. The events extractor is
slower per-call than the judge (multi-event reasoning) so a tighter
interval keeps the user informed during the ~5-10 min wall."""

RATE_LIMIT_MAX_RETRIES: int = 6
"""Per-call retry budget for transient errors (429, 5xx). Exponential
backoff means total wait is up to ~2 minutes across 6 attempts."""

# Token estimates for the dry-run cost projection.
# Opus 4.7 events extraction prompts:
# - Input: compressed trajectory (~6-8 K tokens) + label list + framing.
#   Conservative: 8000 tokens / trace.
# - Output: multi-event JSON with recovery summaries. Empirically a
#   trace with 2-3 events writes ~600-900 tokens; we provision 1000 for
#   the headroom.
DRY_RUN_INPUT_TOKEN_ESTIMATE: int = 8_000
DRY_RUN_OUTPUT_TOKEN_ESTIMATE: int = 1_000

WORST_CASE_MULTIPLIER: float = 1.5
"""Multiplier applied to the per-call estimate for the worst-case USD
projection. Padding (rather than a sharp upper bound) lets a single
re-run or a longer-prompt outlier stay within budget; ratio = 1.5 was
the user's "increase limits a bit higher than expected" guidance."""


@dataclass(frozen=True)
class _Row:
    trace_id: str
    atif: str
    existing_succeeded: int | None  # ultimately_succeeded value pre-extraction


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Count what would be extracted + project cost; no API calls. "
            "This is also the default behaviour when --confirm is omitted."
        ),
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help=(
            "Explicitly opt in to live API calls. Without this flag the "
            "script always behaves like --dry-run, so a misclick can't "
            "spend $50+ silently."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional ceiling on number of traces processed (after "
            "selection). Useful for a small 'sample' run before the "
            "full corpus."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            f"Number of parallel extraction calls. Defaults to "
            f"{DEFAULT_WORKERS}; capped at {MAX_WORKERS}. The SDK + our "
            f"extract_events helper are thread-safe; one Anthropic "
            f"client is shared across workers."
        ),
    )
    parser.add_argument(
        "--max-spend-usd",
        type=float,
        default=200.0,
        help=(
            "Hard kill-switch: abort the loop once running spend exceeds "
            "this many dollars. Defaults to $200 -- comfortably above "
            "the expected ~$70 worst case, but a real ceiling so a "
            "runaway can't silently spend $1000."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Selection.
# ---------------------------------------------------------------------------


def _select_targets(conn: sqlite3.Connection) -> list[_Row]:
    """Build the dry-run-stable selection set.

    Targets every trace that:
    * has parseable ATIF (``atif IS NOT NULL``),
    * looks like it might be a failure (``has_error = 1`` OR
      ``failure_label IS NOT NULL`` -- the latter catches the 10%
      success sample from the slice-4 judge, which is useful coverage
      for verifying the extractor doesn't hallucinate events on clean
      traces),
    * doesn't already have ``failure_events`` (idempotency).

    Order is deterministic so dry-run output matches the live run.
    """
    cur = conn.execute(
        """
        SELECT id, atif, ultimately_succeeded
          FROM traces
         WHERE atif IS NOT NULL
           AND (has_error = 1 OR failure_label IS NOT NULL)
           AND failure_events IS NULL
         ORDER BY id ASC
        """
    )
    out: list[_Row] = []
    for row in cur:
        out.append(
            _Row(
                trace_id=row["id"],
                atif=row["atif"],
                existing_succeeded=row["ultimately_succeeded"],
            )
        )
    return out


# ---------------------------------------------------------------------------
# Backoff helper (429 / 5xx-aware sleep).
# ---------------------------------------------------------------------------


def _sleep_for_rate_limit(
    attempt: int, retry_after: float | None, base: float = 2.0, jitter: float = 0.5
) -> None:
    """Sleep before retrying a transient error.

    When the provider tells us how long to wait (``retry-after`` header,
    surfaced through the SDK) we honour it; otherwise we fall back to
    exponential backoff with jitter so a worker pool doesn't re-fire in
    lockstep after a coordinated 429 wave.
    """
    if retry_after is not None and retry_after > 0:
        delay = float(retry_after)
    else:
        delay = base * (2 ** (attempt - 1))
        delay += random.uniform(0, jitter * delay)
    time.sleep(delay)


def _extract_retry_after(exc: Exception) -> float | None:
    """Pull the ``retry-after`` value (seconds) out of an Anthropic error.

    The SDK exposes the raw response headers via ``exc.response.headers``.
    We tolerate the field being missing / non-numeric -- the caller
    falls back to exponential backoff."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after") if hasattr(headers, "get") else None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Worker body.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _WorkerOutcome:
    row: _Row
    result: EventsResult | None = None
    extraction_error: EventsExtractionError | None = None
    bad_atif: bool = False
    unexpected_error: Exception | None = None
    rate_limited_giveup: bool = False


def _call_with_backoff(client: Any, atif: dict[str, Any]) -> EventsResult:
    """Wrap :func:`extract_events` with 429/5xx backoff.

    The events extractor's own one-shot retry (for schema-validation
    failures) lives inside ``extract_events``; this layer handles
    transient HTTP errors instead. They're orthogonal -- a 429 won't
    cause a schema failure and vice versa -- so nesting them is fine.
    """
    last_transient: Exception | None = None
    for attempt in range(1, RATE_LIMIT_MAX_RETRIES + 1):
        try:
            return extract_events(atif, client=client)
        except RateLimitError as exc:
            last_transient = exc
            if attempt == RATE_LIMIT_MAX_RETRIES:
                break
            retry_after = _extract_retry_after(exc)
            log.warning(
                "Anthropic 429 (attempt %d/%d); sleeping %s",
                attempt,
                RATE_LIMIT_MAX_RETRIES,
                f"{retry_after:.1f}s" if retry_after else "exp-backoff",
            )
            _sleep_for_rate_limit(attempt, retry_after)
        except APIStatusError as exc:
            status = getattr(exc, "status_code", None)
            if status is None or not (500 <= int(status) < 600):
                raise
            last_transient = exc
            if attempt == RATE_LIMIT_MAX_RETRIES:
                break
            retry_after = _extract_retry_after(exc)
            log.warning(
                "Anthropic %s (attempt %d/%d); sleeping %s",
                status,
                attempt,
                RATE_LIMIT_MAX_RETRIES,
                f"{retry_after:.1f}s" if retry_after else "exp-backoff",
            )
            _sleep_for_rate_limit(attempt, retry_after)
    assert last_transient is not None  # noqa: S101
    raise last_transient


def _extract_one(row: _Row, *, client: Any) -> _WorkerOutcome:
    """Worker body: parse ATIF, call Opus, return everything the main
    thread needs to persist. No DB access from worker threads."""
    try:
        atif = json.loads(row.atif)
    except json.JSONDecodeError:
        return _WorkerOutcome(row=row, bad_atif=True)
    if not isinstance(atif, dict):
        return _WorkerOutcome(row=row, bad_atif=True)
    try:
        result = _call_with_backoff(client, atif)
        return _WorkerOutcome(row=row, result=result)
    except EventsExtractionError as exc:
        return _WorkerOutcome(row=row, extraction_error=exc)
    except (RateLimitError, APIStatusError) as exc:
        # Exhausted backoff -- we treat this as a per-trace failure and
        # keep going so a transient wave doesn't tank the whole run.
        return _WorkerOutcome(row=row, rate_limited_giveup=True, unexpected_error=exc)
    except Exception as exc:  # noqa: BLE001 -- network / SDK errors
        return _WorkerOutcome(row=row, unexpected_error=exc)


# ---------------------------------------------------------------------------
# Persistence (main-thread only).
# ---------------------------------------------------------------------------


def _persist(conn: sqlite3.Connection, outcome: _WorkerOutcome) -> None:
    assert outcome.result is not None  # noqa: S101
    events = outcome.result.events
    summary_label = summarize_label(events).value
    new_succeeded = derive_ultimately_succeeded(events, outcome.row.existing_succeeded)
    payload = json.dumps([e.to_dict() for e in events])
    conn.execute(
        """
        UPDATE traces
           SET failure_events = ?,
               failure_label = ?,
               failure_label_source = 'opus_events_derived',
               ultimately_succeeded = ?
         WHERE id = ?
        """,
        (payload, summary_label, new_succeeded, outcome.row.trace_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Dry-run + live-run renderers.
# ---------------------------------------------------------------------------


def _project_costs(target_count: int) -> tuple[int, int, float, float]:
    """Return ``(input_tokens, output_tokens, expected_usd, worst_case_usd)``
    for a selection set of ``target_count`` traces."""
    total_in = target_count * DRY_RUN_INPUT_TOKEN_ESTIMATE
    total_out = target_count * DRY_RUN_OUTPUT_TOKEN_ESTIMATE
    expected = estimate_cost_usd(total_in, total_out)
    worst = expected * WORST_CASE_MULTIPLIER
    return total_in, total_out, expected, worst


def _print_dry_run(targets: list[_Row]) -> int:
    """Pretty-print the cost projection. Every line is prefixed
    ``[DRY RUN]`` so this can't be mistaken for a live execution."""
    in_tok, out_tok, expected_usd, worst_usd = _project_costs(len(targets))
    bar = "=" * 60
    lines = [
        bar,
        "DRY RUN (no API calls)",
        f"Model:                    {EVENTS_MODEL}",
        f"Total traces to extract:  {len(targets)}",
        (
            f"Per-call estimate:        {DRY_RUN_INPUT_TOKEN_ESTIMATE:,} in / "
            f"{DRY_RUN_OUTPUT_TOKEN_ESTIMATE:,} out tokens"
        ),
        f"Aggregate input tokens:   {in_tok:>12,}",
        f"Aggregate output tokens:  {out_tok:>12,}",
        f"Expected USD:             ${expected_usd:.2f}",
        (
            f"Worst-case USD:           ${worst_usd:.2f}  "
            f"(= expected x {WORST_CASE_MULTIPLIER})"
        ),
        (
            f"Pricing:                  input ${INPUT_PRICE_PER_MTOK}/Mtok, "
            f"output ${OUTPUT_PRICE_PER_MTOK}/Mtok"
        ),
        f"max_tokens cap per call:  {EVENTS_MAX_TOKENS}",
        "",
        "To run live, re-invoke with --confirm:",
        "    uv run python scripts/extract_failure_events.py --confirm",
        bar,
    ]
    for line in lines:
        print(f"[DRY RUN] {line}")
    return 0


def _print_live_banner(
    target_count: int, workers: int, expected_usd: float, worst_usd: float
) -> None:
    """Frame the start of a live run so the operator sees a banner
    distinct from any dry-run output."""
    bar = "=" * 60
    print(bar)
    print("STARTING LIVE EXECUTION")
    print(f"  traces:   {target_count}")
    print(f"  workers:  {workers}")
    print(f"  model:    {EVENTS_MODEL}")
    print(f"  expected: ${expected_usd:.2f}  (worst-case ${worst_usd:.2f})")
    print(bar)


def _run_live(
    conn: sqlite3.Connection,
    targets: list[_Row],
    *,
    max_spend_usd: float,
    workers: int,
) -> int:
    """Extract events for ``targets`` in parallel, persist on the main thread.

    ``targets`` is expected to already have ``--limit`` applied by the
    caller (see :func:`main`); we no longer trim here so dry-run and
    live paths see the same list and ``_print_dry_run`` can be a
    faithful preview of what ``--confirm`` would burn.
    """
    require_api_key()
    workers = max(1, min(int(workers), MAX_WORKERS, len(targets) or 1))

    _, _, expected_usd, worst_usd = _project_costs(len(targets))
    _print_live_banner(len(targets), workers, expected_usd, worst_usd)

    total_in = 0
    total_out = 0
    successes = 0
    failures = 0
    aborted_for_cost = False
    completed = 0

    shared_client = Anthropic()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures: dict[Future[_WorkerOutcome], _Row] = {
            pool.submit(_extract_one, row, client=shared_client): row for row in targets
        }
        try:
            for fut in as_completed(futures):
                completed += 1
                outcome = fut.result()
                row = outcome.row

                if outcome.bad_atif:
                    log.warning(
                        "[LIVE] Trace %s has malformed ATIF; skipping",
                        row.trace_id,
                    )
                    failures += 1
                elif outcome.extraction_error is not None:
                    log.warning(
                        "[LIVE] Extraction failed for %s: %s",
                        row.trace_id,
                        outcome.extraction_error,
                    )
                    failures += 1
                elif outcome.rate_limited_giveup:
                    log.error(
                        "[LIVE] Gave up on %s after exhausting backoff: %s",
                        row.trace_id,
                        outcome.unexpected_error,
                    )
                    failures += 1
                elif outcome.unexpected_error is not None:
                    log.error(
                        "[LIVE] Unexpected error on %s: %s",
                        row.trace_id,
                        outcome.unexpected_error,
                    )
                    failures += 1
                else:
                    assert outcome.result is not None  # noqa: S101
                    _persist(conn, outcome)
                    total_in += outcome.result.input_tokens
                    total_out += outcome.result.output_tokens
                    successes += 1

                if completed % PROGRESS_EVERY == 0 or completed == len(targets):
                    cost_so_far = estimate_cost_usd(total_in, total_out)
                    print(
                        f"[LIVE]   [{completed:>4}/{len(targets)}] "
                        f"ok={successes} fail={failures}  "
                        f"tokens={total_in:>8,}/{total_out:>6,}  "
                        f"cost=${cost_so_far:.2f}"
                    )

                running_cost = estimate_cost_usd(total_in, total_out)
                if running_cost >= max_spend_usd:
                    print(
                        f"\n[LIVE] ABORTING: running spend ${running_cost:.2f} "
                        f"hit hard cap ${max_spend_usd:.2f}. "
                        f"{completed}/{len(targets)} traces processed."
                    )
                    aborted_for_cost = True
                    for pending in futures:
                        if not pending.done():
                            pending.cancel()
                    break
        except KeyboardInterrupt:
            print("\n[LIVE] Interrupted; cancelling pending calls...")
            for pending in futures:
                if not pending.done():
                    pending.cancel()
            raise

    final_cost = estimate_cost_usd(total_in, total_out)
    bar = "=" * 60
    print(bar)
    print(f"Extracted successfully:  {successes}")
    print(f"Skipped (errors):        {failures}")
    print(f"Total input tokens:      {total_in:,}")
    print(f"Total output tokens:     {total_out:,}")
    print(f"Estimated cost:          ${final_cost:.4f}")
    if aborted_for_cost:
        print("Run was aborted by max-spend cap.")
    print(bar)
    return 0 if not aborted_for_cost else 2


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


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
        targets = _select_targets(conn)
        if not targets:
            print(
                "No traces need extraction "
                "(failure_events already populated for every target row)."
            )
            return 0

        # Apply --limit BEFORE branching so dry-run and live see the
        # exact same target list. Previously --limit was honoured only
        # inside _run_live, which made ``--dry-run --limit 5`` project
        # the cost for the full corpus -- misleading and defeats the
        # purpose of dry-run as a preview.
        if args.limit is not None:
            targets = targets[: max(0, int(args.limit))]
            if not targets:
                print(
                    f"--limit {args.limit} reduces the selection to zero "
                    "traces; nothing to do."
                )
                return 0

        # Default and --dry-run both go to the dry-run path; live calls
        # require --confirm. Mixed flags (--dry-run + --confirm) treat
        # --dry-run as the operator opting back out at the last second.
        if args.dry_run or not args.confirm:
            return _print_dry_run(targets)

        return _run_live(
            conn,
            targets,
            max_spend_usd=args.max_spend_usd,
            workers=args.max_workers,
        )


if __name__ == "__main__":
    raise SystemExit(main())
