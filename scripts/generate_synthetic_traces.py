"""Sonnet-authored synthetic trace orchestrator (Slice 7).

Generates a corpus of N synthetic agent traces split across Claude
Code / Codex / Cursor formats, with a configurable failure-mode
distribution. Each trace is written to ``data/raw/<format>/`` so the
existing ``scripts/ingest_*.py`` pipeline picks them up unchanged.

Pipeline:

1. ``sample_specs(N, seed)`` -> deterministic list of TraceSpecs.
2. ``ThreadPoolExecutor`` fans them out to the Anthropic API
   (default 30 workers). Each worker:
     a. Builds the format-specific prompt + tool schema.
     b. Calls ``claude-sonnet-4-6`` with prompt caching enabled.
     c. Runs the response through the matching adapter.
     d. Retries once with the adapter's error if validation fails.
3. Successful traces land on disk; the manifest records every
   spec's outcome (ok/error/cost/path).
4. The orchestrator gates on a cumulative budget cap so a runaway
   experiment can't blow past the user's stated spend.

Flags (the ones worth highlighting):

* ``--dry-run``: sample specs + project cost; no API calls.
* ``--probe-rate-limits``: one tiny call (~$0.0001) reads the
  ``anthropic-ratelimit-*`` response headers and prints RPM / ITPM /
  OTPM caps + a recommended ``--max-workers`` + an ETA for the
  planned ``--count``. Use BEFORE the live run; cheaper and more
  accurate than timing-based inference.
* ``--budget USD``: hard cap. Workers refuse to start a fresh call
  once cumulative cost exceeds this number. Final result may be
  short of ``--count`` -- that's by design.
* ``--limit N``: cap on total traces this invocation will attempt.
  Combined with ``--count`` to support smoke runs (``--count 500
  --limit 6`` samples 500 specs but only runs the first 6).
* ``--keep-failed``: write the last raw bytes from failed
  generations to ``failed/`` for inspection. Off by default.

Usage:
    uv run python scripts/generate_synthetic_traces.py --dry-run
    uv run python scripts/generate_synthetic_traces.py --probe-rate-limits
    uv run python scripts/generate_synthetic_traces.py --limit 6  # smoke
    uv run python scripts/generate_synthetic_traces.py            # full
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from trace_marketplace.synth.llm_generator import (
    CostEstimate,
    DEFAULT_RATE_LIMIT_RETRIES,
    GENERATOR_MODEL,
    GenerationResult,
    generate_trace,
)
from trace_marketplace.synth.spec import (
    ALL_FORMATS,
    FORMAT_EXTENSIONS,
    TraceSpec,
    resolve_label_distribution,
    sample_specs,
)

log = logging.getLogger("generate_synthetic_traces")

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"
DEFAULT_MANIFEST_NAME = "synth_manifest.json"
DEFAULT_COUNT: int = 500
DEFAULT_MAX_WORKERS: int = 30
"""Matches what the user asked for: max-throughput Anthropic
concurrency. Above this the marginal speedup is small and the 429
risk dominates."""

DEFAULT_BUDGET_USD: float = 80.0
"""Generous-but-bounded ceiling. The expected spend for 500 traces
is ~$30-45 even with retries; an 80$ cap gives ~2x headroom while
still capping a runaway experiment well before damage is done."""

# Pricing for cost projection (mirrors llm_generator constants;
# duplicated here so ``--dry-run`` doesn't need to import the live
# generator's pricing table by name).
_DRYRUN_INPUT_TOKENS_PER_TRACE_FIRST = 4_500
_DRYRUN_OUTPUT_TOKENS_PER_TRACE = 7_000
_DRYRUN_CACHE_READ_TOKENS = 3_500  # cached portion of system prompt


# ---------------------------------------------------------------------------
# CLI parsing.
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"Total specs to sample (default: {DEFAULT_COUNT}).",
    )
    parser.add_argument(
        "--failure-ratio",
        type=float,
        default=0.75,
        help=(
            "Legacy knob -- only consulted when --distribution-file is "
            "passed as an empty file. Default Slice 7 distribution is "
            "an explicit per-label dict; this flag is preserved for "
            "back-compat."
        ),
    )
    parser.add_argument(
        "--distribution-file",
        type=Path,
        default=None,
        help=(
            "Path to a JSON file mapping FailureLabel values -> floats "
            "(must sum to ~1.0). Overrides the default Slice 7 demo "
            "distribution (30%% success, 5%% partial, 15%% loop, 10%% "
            "each on gave_up/hallucinated_api/misread_error/"
            "environment_issue/wrong_approach)."
        ),
    )
    parser.add_argument(
        "--target-step-count",
        type=int,
        default=25,
        help="Per-trace step-count hint passed to Sonnet (default: 25).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible spec sampling (default: 42).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Concurrent Anthropic calls (default: {DEFAULT_MAX_WORKERS}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write traces (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=DEFAULT_BUDGET_USD,
        help=(
            f"Hard cap on cumulative spend in USD (default: ${DEFAULT_BUDGET_USD:.0f}). "
            "Workers stop scheduling new calls once exceeded."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Cap on traces this run will actually attempt (after spec "
            "sampling). Useful for smoke tests; e.g. --limit 6."
        ),
    )
    parser.add_argument(
        "--rate-limit-retries",
        type=int,
        default=DEFAULT_RATE_LIMIT_RETRIES,
        help=f"Per-call HTTP 429 retry budget (default: {DEFAULT_RATE_LIMIT_RETRIES}).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Where to write the per-spec outcome manifest "
            f"(default: <output-dir>/{DEFAULT_MANIFEST_NAME})."
        ),
    )
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help=(
            "Persist raw bytes from failed generations to "
            "<output-dir>/failed/ for inspection. Off by default."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sample specs + project cost; do not call the Anthropic API.",
    )
    parser.add_argument(
        "--probe-rate-limits",
        action="store_true",
        help=(
            "Make one tiny call to read Anthropic's RPM / ITPM / OTPM "
            "caps from the response headers, print them + a "
            "recommended --max-workers + a wall-clock ETA for --count, "
            "then exit. ~$0.0001, no corpus files written."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Distribution loading.
# ---------------------------------------------------------------------------


def _resolve_distribution(args: argparse.Namespace) -> dict[str, float] | None:
    """Decide which label distribution to use.

    Priority order:
    1. ``--distribution-file`` path with non-empty JSON dict -> load it.
    2. ``--distribution-file`` path with empty content -> sentinel ``{}``
       so :func:`sample_specs` falls back to ``failure_ratio``.
    3. No flag -> ``None`` so :data:`DEFAULT_LABEL_DISTRIBUTION` wins.

    Loading errors are fatal (we'd rather abort before spending money
    on a corpus shaped by a typo).
    """
    path = args.distribution_file
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise SystemExit(f"Could not read --distribution-file {path}: {exc}") from exc
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--distribution-file {path}: invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(
            f"--distribution-file {path}: top-level must be a JSON object, "
            f"got {type(parsed).__name__}"
        )
    coerced: dict[str, float] = {}
    for label, weight in parsed.items():
        if not isinstance(weight, (int, float)):
            raise SystemExit(
                f"--distribution-file {path}: weight for {label!r} must be a number, "
                f"got {type(weight).__name__}"
            )
        coerced[label] = float(weight)
    return coerced


# ---------------------------------------------------------------------------
# Dry-run cost projection.
# ---------------------------------------------------------------------------


def _project_cost(specs: list[TraceSpec]) -> dict[str, Any]:
    """Heuristic dry-run cost estimate.

    Approximations (calibrated against early smoke runs):
    * First request per format: full system + brief, no cache hit.
    * Subsequent requests per format: ~80% of system tokens cached.
    * Validation failure retry rate: ~10% of specs trigger a 2nd
      call.
    * Output: ~7K tokens per successful generation.

    The actual run reconciles against real ``usage`` -- this is just
    a "should I press go?" sanity check.
    """
    from trace_marketplace.synth.llm_generator import (
        PRICE_PER_MTOK_CACHE_READ,
        PRICE_PER_MTOK_CACHE_WRITE,
        PRICE_PER_MTOK_INPUT,
        PRICE_PER_MTOK_OUTPUT,
    )

    n = len(specs)
    n_first_per_format = len(set(s.output_format for s in specs))
    n_cached_calls = max(0, n - n_first_per_format)

    cache_write = n_first_per_format * _DRYRUN_INPUT_TOKENS_PER_TRACE_FIRST
    cache_read = n_cached_calls * _DRYRUN_CACHE_READ_TOKENS
    base_input = (
        n_first_per_format * (_DRYRUN_INPUT_TOKENS_PER_TRACE_FIRST - _DRYRUN_CACHE_READ_TOKENS)
        + n_cached_calls
        * (_DRYRUN_INPUT_TOKENS_PER_TRACE_FIRST - _DRYRUN_CACHE_READ_TOKENS)
    )
    output_tokens = int(n * _DRYRUN_OUTPUT_TOKENS_PER_TRACE * 1.1)  # +10% retries

    cost_usd = (
        base_input * PRICE_PER_MTOK_INPUT / 1_000_000
        + cache_write * PRICE_PER_MTOK_CACHE_WRITE / 1_000_000
        + cache_read * PRICE_PER_MTOK_CACHE_READ / 1_000_000
        + output_tokens * PRICE_PER_MTOK_OUTPUT / 1_000_000
    )

    by_format: dict[str, int] = {fmt: 0 for fmt in ALL_FORMATS}
    for spec in specs:
        by_format[spec.output_format] += 1

    by_failure: dict[str, int] = {}
    for spec in specs:
        by_failure[spec.primary_failure] = by_failure.get(spec.primary_failure, 0) + 1

    return {
        "n_specs": n,
        "by_format": by_format,
        "by_failure": by_failure,
        "est_input_tokens": base_input,
        "est_cache_write_tokens": cache_write,
        "est_cache_read_tokens": cache_read,
        "est_output_tokens": output_tokens,
        "est_cost_usd": round(cost_usd, 2),
    }


def _print_dry_run(
    projection: dict[str, Any],
    *,
    distribution: dict[str, float] | None,
    failure_ratio: float,
) -> None:
    """Render the dry-run summary including the active distribution.

    Mirrors :func:`sample_specs`'s three-way precedence
    (``None`` -> default, ``{}`` -> legacy ``failure_ratio``, dict ->
    as-is) via :func:`resolve_label_distribution` so the "Active label
    distribution (target)" the user sees here is byte-for-byte what
    the sampler will use on the live run. Previously the legacy
    ``{}`` path silently fell back to the demo default when printed.
    """
    active = resolve_label_distribution(distribution, failure_ratio=failure_ratio)
    print()
    print("=" * 60)
    print("DRY RUN -- no Anthropic API calls")
    print("-" * 60)
    print(f"  Specs sampled:         {projection['n_specs']}")
    print("  Active label distribution (target):")
    for label, weight in sorted(active.items(), key=lambda x: -x[1]):
        print(f"    {label:<22} {weight * 100:>5.1f}%")
    print("  Format distribution (actual):")
    for fmt, count in projection["by_format"].items():
        print(f"    {fmt:<14} {count}")
    print("  Failure-mode distribution (actual primary):")
    for label, count in sorted(
        projection["by_failure"].items(), key=lambda x: -x[1]
    ):
        print(f"    {label:<22} {count}")
    print(f"  Est. input tokens:      {projection['est_input_tokens']:,}")
    print(f"  Est. cache-write toks:  {projection['est_cache_write_tokens']:,}")
    print(f"  Est. cache-read toks:   {projection['est_cache_read_tokens']:,}")
    print(f"  Est. output tokens:     {projection['est_output_tokens']:,}")
    print(f"  Est. cost (USD):        ${projection['est_cost_usd']}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Rate-limit probe.
# ---------------------------------------------------------------------------
#
# We don't have to *infer* the tier from timing -- every Anthropic
# response carries the limits in headers:
#
#   anthropic-ratelimit-requests-limit       (RPM)
#   anthropic-ratelimit-input-tokens-limit   (ITPM)
#   anthropic-ratelimit-output-tokens-limit  (OTPM)
#   anthropic-ratelimit-*-remaining          (current budget)
#   anthropic-ratelimit-*-reset              (UTC ISO timestamp)
#
# Strategy: one tiny call (max_tokens=8) gives us all of the above for
# essentially free (~$0.0001), and we can derive a defensible
# `--max-workers` recommendation from the OTPM ceiling instead of
# guessing.

# Anthropic's published baseline tier limits (claude-sonnet-4-6).
# Custom enterprise tiers go higher; we just use these to print a
# rough tier label for context.
_TIER_TABLE: list[tuple[str, int, int]] = [
    ("Tier 1", 50, 8_000),
    ("Tier 2", 1_000, 16_000),
    ("Tier 3", 2_000, 32_000),
    ("Tier 4", 4_000, 80_000),
]

# Calibration constants for the worker recommendation. Calibrated
# against early smoke runs; the orchestrator's actual usage will
# refine these. Conservative on purpose -- we'd rather under-utilise
# than spend an hour fighting 429s.
_AVG_OUTPUT_TOKENS_PER_TRACE: int = 7_000
_AVG_SECONDS_PER_TRACE: float = 30.0
_HEADROOM_FACTOR: float = 0.75
"""Use only 75% of the published cap. Anthropic's burst behaviour
plus our prompt-caching-write spikes routinely push us over the
exact cap; 75% lets the smooth-throughput stay well under it."""


def _label_for_otpm(otpm: int) -> str:
    """Map an OTPM cap to a tier label for human-readable output.

    Returns "Tier N" for caps within ~20% of the published baselines,
    or "custom" for limits that don't match any baseline (which is
    common -- requested tier increases land at non-baseline numbers).
    """
    for label, _rpm, baseline_otpm in _TIER_TABLE:
        if baseline_otpm * 0.8 <= otpm <= baseline_otpm * 1.25:
            return label
    return "custom (above baseline)" if otpm > _TIER_TABLE[-1][2] else "custom"


def _recommended_workers(otpm: int) -> int:
    """Compute a safe max-workers from the OTPM ceiling.

    Maths:
      throughput_per_worker_per_min = AVG_OUTPUT_TOKENS_PER_TRACE
                                      * (60 / AVG_SECONDS_PER_TRACE)
      max_workers = floor(OTPM * HEADROOM / throughput_per_worker)

    With 7k tokens / 30s per trace, one worker consumes ~14k OTPM
    sustained. Tier 4 (80k OTPM) -> floor(80k * 0.75 / 14k) = 4
    workers. That sounds low, but it matches what the API will
    actually let through without backpressure.

    Floor of 1 (you can always run at least one), ceiling capped at
    50 (above that other bottlenecks dominate).
    """
    per_worker = _AVG_OUTPUT_TOKENS_PER_TRACE * (60.0 / _AVG_SECONDS_PER_TRACE)
    raw = int((otpm * _HEADROOM_FACTOR) / per_worker)
    return max(1, min(raw, 50))


def _read_rate_limit_headers(client: Any) -> dict[str, str] | None:
    """Make one cheap call and pull the rate-limit headers off the response.

    Uses ``client.messages.with_raw_response.create`` so we get the
    full HTTP response object (not just the parsed message). The
    extra ``with_raw_response`` indirection is the documented way to
    access headers in the Anthropic SDK.

    Returns the headers dict on success, ``None`` if the SDK doesn't
    expose ``with_raw_response`` (older versions). On HTTP errors
    other than success we let the SDK raise so the user sees what
    actually broke (e.g. invalid key, no billing set up).
    """
    raw = getattr(client.messages, "with_raw_response", None)
    if raw is None:
        return None
    response = raw.create(
        model=GENERATOR_MODEL,
        max_tokens=8,
        messages=[{"role": "user", "content": "ping"}],
    )
    headers = getattr(response, "headers", None) or {}
    return {k.lower(): str(v) for k, v in dict(headers).items()}


def _eta_for_count(otpm: int, count: int, max_workers: int) -> str:
    """Return a human-readable ETA for ``count`` traces at this OTPM.

    Two regimes:

    * **Token-bound** (low tiers): OTPM caps total output throughput.
      Wall-clock floor = ``count * tokens_per_trace / OTPM``.
    * **Worker-bound** (high tiers): OTPM has plenty of headroom but
      we only have ``max_workers`` threads, each holding a slot for
      ``AVG_SECONDS_PER_TRACE``. Wall-clock floor = ``count *
      seconds_per_trace / max_workers``.

    Real wall-clock is dominated by whichever floor is higher, so we
    take ``max(token_floor, serial_floor)`` as the lower bound. Upper
    bound adds 10% retry slack + 30% variance margin -- enough to
    cover the long tail of slow calls without being so wide it stops
    informing the press-go decision.

    Both bounds are clamped together so the display never inverts.
    """
    if otpm <= 0:
        return "unknown"
    total_output_tokens = count * _AVG_OUTPUT_TOKENS_PER_TRACE
    token_floor_min = total_output_tokens / otpm
    serial_floor_min = (count * _AVG_SECONDS_PER_TRACE) / max(1, max_workers) / 60.0
    lower_min = max(token_floor_min, serial_floor_min)
    # 1.30x covers per-call latency variance; +10% on top for the
    # observed ~10% retry rate from the validator's adapter checks.
    upper_min = lower_min * 1.30 * 1.10
    return f"{lower_min:.0f}-{upper_min:.0f} min"


def _probe_rate_limits(max_workers: int, count_for_eta: int) -> int:
    """Read the Anthropic account's rate-limit headers and report.

    Output (printed):
      * RPM / ITPM / OTPM caps and current "remaining" budget.
      * Inferred tier label.
      * Recommended ``--max-workers`` based on OTPM + per-trace cost.
      * Wall-clock ETA range for the planned corpus size.

    Returns the OTPM (callers don't really use the return value -- it's
    mostly there for tests).
    """
    from anthropic import Anthropic

    client = Anthropic()
    try:
        headers = _read_rate_limit_headers(client)
    except Exception as exc:  # noqa: BLE001
        log.error("Probe call failed: %s", exc)
        print(
            "Probe could not reach Anthropic. Check ANTHROPIC_API_KEY, network, "
            "and that the account has billing enabled."
        )
        return 0

    if headers is None:
        print(
            "Anthropic SDK is too old to expose response headers. "
            "Upgrade with `uv add anthropic@latest`."
        )
        return 0

    def _h_int(name: str) -> int:
        try:
            return int(headers.get(name, "0"))
        except ValueError:
            return 0

    rpm = _h_int("anthropic-ratelimit-requests-limit")
    itpm = _h_int("anthropic-ratelimit-input-tokens-limit")
    otpm = _h_int("anthropic-ratelimit-output-tokens-limit")
    rpm_remaining = _h_int("anthropic-ratelimit-requests-remaining")
    otpm_remaining = _h_int("anthropic-ratelimit-output-tokens-remaining")
    rpm_reset = headers.get("anthropic-ratelimit-requests-reset", "?")
    otpm_reset = headers.get("anthropic-ratelimit-output-tokens-reset", "?")

    tier_label = _label_for_otpm(otpm) if otpm else "unknown"
    suggested = _recommended_workers(otpm) if otpm else 1
    eta = _eta_for_count(otpm, count_for_eta, suggested) if otpm else "unknown"

    print()
    print("=" * 60)
    print("ANTHROPIC RATE-LIMIT PROBE (from response headers)")
    print("-" * 60)
    print(f"  Model:                       {GENERATOR_MODEL}")
    print(f"  Requests/min (RPM):          {rpm or '?'}  "
          f"(remaining: {rpm_remaining}, resets {rpm_reset})")
    print(f"  Input tokens/min (ITPM):     {itpm or '?'}")
    print(f"  Output tokens/min (OTPM):    {otpm or '?'}  "
          f"(remaining: {otpm_remaining}, resets {otpm_reset})")
    print(f"  Inferred tier:               {tier_label}")
    print("-" * 60)
    print(f"  Per-trace est:               {_AVG_OUTPUT_TOKENS_PER_TRACE:,} output toks, "
          f"~{_AVG_SECONDS_PER_TRACE:.0f}s")
    print(f"  Recommended --max-workers:   {suggested}")
    print(f"  ETA for --count {count_for_eta}:   ~{eta}")
    print(f"  (You passed --max-workers {max_workers}; "
          f"using more than {suggested} will likely hit 429s.)")
    print("=" * 60)
    print()
    print("Source of truth for tier:")
    print("  https://console.anthropic.com/settings/limits")
    print()
    return otpm


# ---------------------------------------------------------------------------
# Main orchestration.
# ---------------------------------------------------------------------------


class _BudgetGate:
    """Atomic budget-cap tracker shared across worker threads.

    The orchestrator checks ``should_stop()`` before scheduling each
    new spec. Already-running futures complete (their cost is already
    spent); new ones get short-circuited to a "budget exhausted"
    failure outcome so the manifest still records them.
    """

    def __init__(self, cap_usd: float) -> None:
        self._cap = cap_usd
        self._spent = 0.0
        self._lock = threading.Lock()

    def record(self, cost: CostEstimate) -> None:
        with self._lock:
            self._spent += cost.usd

    def spent(self) -> float:
        with self._lock:
            return self._spent

    def should_stop(self) -> bool:
        with self._lock:
            return self._spent >= self._cap


def _atif_summary(result: GenerationResult) -> dict[str, Any]:
    """Pull a small ATIF summary out of a successful result.

    The manifest records the number of steps and tool calls so a
    quick scan can flag any "all 5-step degenerate" runs without
    re-parsing the raw bytes.
    """
    if not result.ok or result.validation is None or result.validation.trajectory is None:
        return {}
    traj = result.validation.trajectory
    tool_calls = 0
    for step in traj.steps:
        tcs = getattr(step, "tool_calls", None)
        if tcs:
            tool_calls += len(tcs)
    return {
        "n_steps": len(traj.steps),
        "n_tool_calls": tool_calls,
    }


def _filename_for(result: GenerationResult, index: int) -> str:
    """Deterministic filename: ``synth_NNNN_<format>_<label>.<ext>``.

    Numbered first so directory listing is roughly chronological;
    label included so a human eyeballing the dir can pick out
    "find a loop trace" without grepping the manifest.
    """
    ext = FORMAT_EXTENSIONS[result.spec.output_format]
    safe_label = result.spec.primary_failure.replace(" ", "_")
    return f"synth_{index:04d}_{result.spec.output_format}_{safe_label}{ext}"


def _write_trace(
    output_dir: Path,
    result: GenerationResult,
    index: int,
) -> Path:
    """Write a successful generation to its format dir."""
    fmt_dir = output_dir / result.spec.output_format
    fmt_dir.mkdir(parents=True, exist_ok=True)
    path = fmt_dir / _filename_for(result, index)
    assert result.file_bytes is not None  # noqa: S101
    path.write_text(result.file_bytes, encoding="utf-8")
    return path


def _write_failed(
    output_dir: Path,
    result: GenerationResult,
    index: int,
) -> Path | None:
    """Optionally persist a failed generation's raw bytes for triage."""
    if result.file_bytes is None:
        return None
    fail_dir = output_dir / "failed"
    fail_dir.mkdir(parents=True, exist_ok=True)
    path = fail_dir / _filename_for(result, index)
    path.write_text(result.file_bytes, encoding="utf-8")
    return path


def _build_manifest_entry(
    result: GenerationResult,
    output_path: Path | None,
    *,
    budget_skipped: bool = False,
) -> dict[str, Any]:
    """Render one :class:`GenerationResult` as a manifest record."""
    entry: dict[str, Any] = {
        "spec": {
            "output_format": result.spec.output_format,
            "primary_failure": result.spec.primary_failure,
            "secondary_failures": list(result.spec.secondary_failures),
            "task": result.spec.task,
            "target_step_count": result.spec.target_step_count,
            "is_failure": result.spec.is_failure,
        },
        "ok": result.ok,
        "attempts": result.attempts,
        "error": result.error,
        "notes": result.notes,
        "cost": asdict(result.cost),
        "cost_usd": round(result.cost.usd, 6),
        "output_path": str(output_path) if output_path else None,
        "atif_summary": _atif_summary(result),
    }
    if budget_skipped:
        entry["budget_skipped"] = True
    return entry


def _write_manifest(manifest_path: Path, payload: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_live(args: argparse.Namespace, specs: list[TraceSpec]) -> int:
    """Live (non-dry-run) generation. Returns process exit code."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error(
            "ANTHROPIC_API_KEY is not set; cannot generate traces. "
            "Export it and re-run.\n  export ANTHROPIC_API_KEY=sk-ant-..."
        )
        return 2

    budget = _BudgetGate(args.budget)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (output_dir / DEFAULT_MANIFEST_NAME)

    results: list[dict[str, Any]] = []
    counters = {"ok": 0, "failed": 0, "budget_skipped": 0}
    counter_lock = threading.Lock()
    start = time.monotonic()

    def _process(spec: TraceSpec, idx: int) -> dict[str, Any]:
        if budget.should_stop():
            with counter_lock:
                counters["budget_skipped"] += 1
            placeholder = GenerationResult(
                spec=spec,
                ok=False,
                error=f"budget cap ${args.budget:.2f} exceeded before scheduling",
            )
            return _build_manifest_entry(placeholder, None, budget_skipped=True)

        result = generate_trace(spec, max_429_retries=args.rate_limit_retries)
        budget.record(result.cost)

        output_path: Path | None = None
        if result.ok:
            output_path = _write_trace(output_dir, result, idx)
            with counter_lock:
                counters["ok"] += 1
        else:
            with counter_lock:
                counters["failed"] += 1
            if args.keep_failed:
                output_path = _write_failed(output_dir, result, idx)

        entry = _build_manifest_entry(result, output_path)
        with counter_lock:
            done = counters["ok"] + counters["failed"]
            if done % 10 == 0 or done == len(specs):
                elapsed = time.monotonic() - start
                rate = done / elapsed if elapsed > 0 else 0.0
                log.info(
                    "progress: %d/%d (ok=%d failed=%d skipped=%d) "
                    "spent=$%.2f rate=%.1f/min",
                    done,
                    len(specs),
                    counters["ok"],
                    counters["failed"],
                    counters["budget_skipped"],
                    budget.spent(),
                    rate * 60,
                )
        return entry

    log.info(
        "Starting live run: %d specs, max_workers=%d, budget=$%.2f, "
        "model=%s",
        len(specs),
        args.max_workers,
        args.budget,
        GENERATOR_MODEL,
    )

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures: list[Future[dict[str, Any]]] = [
            ex.submit(_process, spec, idx + 1)
            for idx, spec in enumerate(specs)
        ]
        for fut in futures:
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                log.exception("worker raised: %s", exc)
                results.append(
                    {
                        "ok": False,
                        "error": f"worker exception: {exc}",
                        "spec": None,
                    }
                )

    elapsed = time.monotonic() - start
    total_cost = sum(entry.get("cost_usd", 0.0) or 0.0 for entry in results)

    summary = {
        "model": GENERATOR_MODEL,
        "specs_attempted": len(specs),
        "ok": counters["ok"],
        "failed": counters["failed"],
        "budget_skipped": counters["budget_skipped"],
        "total_cost_usd": round(total_cost, 4),
        "wall_clock_seconds": round(elapsed, 1),
        "max_workers": args.max_workers,
        "seed": args.seed,
        "budget_cap_usd": args.budget,
    }
    _write_manifest(
        manifest_path,
        {"summary": summary, "results": results},
    )

    print()
    print("=" * 60)
    print("LIVE RUN SUMMARY")
    print("-" * 60)
    for k, v in summary.items():
        print(f"  {k:<24} {v}")
    print(f"  manifest:                {manifest_path}")
    print("=" * 60)
    return 0 if counters["ok"] > 0 else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if args.probe_rate_limits:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            log.error("ANTHROPIC_API_KEY is required for --probe-rate-limits")
            return 2
        _probe_rate_limits(args.max_workers, args.count)
        return 0

    distribution = _resolve_distribution(args)
    specs = sample_specs(
        args.count,
        label_distribution=distribution,
        failure_ratio=args.failure_ratio,
        target_step_count=args.target_step_count,
        seed=args.seed,
    )
    if args.limit is not None:
        specs = specs[: args.limit]

    if args.dry_run:
        projection = _project_cost(specs)
        _print_dry_run(
            projection,
            distribution=distribution,
            failure_ratio=args.failure_ratio,
        )
        return 0

    return _run_live(args, specs)


if __name__ == "__main__":
    raise SystemExit(main())
