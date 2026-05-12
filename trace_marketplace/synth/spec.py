"""TraceSpec: the pure-Python "recipe" for one synthetic trace.

This module owns the *distribution* side of synthetic generation: how
many traces of each format, what failure modes, what task domains.
The LLM generator consumes one :class:`TraceSpec` at a time and
returns bytes; nothing in here calls an API or touches the network.

Keeping spec sampling pure has three payoffs:

1. **Reproducibility.** ``sample_specs(N, seed=S)`` returns identical
   lists for identical ``(N, S)``. The full Sonnet run becomes
   re-runnable if a generation pass aborts midway.
2. **Cost projection.** Dry-run can iterate the spec list and
   estimate total spend without making any API calls.
3. **Unit-testability.** Distribution invariants (75% failure ratio,
   even format split, every label covered) are checked with pure
   pytest, no mocks needed.

The 11-label failure taxonomy lives in
:mod:`trace_marketplace.enrich.failure_taxonomy`; we don't duplicate
it here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from trace_marketplace.enrich.failure_taxonomy import FailureLabel

# ---------------------------------------------------------------------------
# Output formats.
# ---------------------------------------------------------------------------

FORMAT_CLAUDE_CODE = "claude_code"
FORMAT_CODEX = "codex"
FORMAT_CURSOR = "cursor"

ALL_FORMATS: tuple[str, ...] = (FORMAT_CLAUDE_CODE, FORMAT_CODEX, FORMAT_CURSOR)
"""Single source of truth for which formats the generator targets.
The CLI's ``--per-format`` flag validates against this tuple, the
prompts module dispatches on it, and the validator picks an adapter
based on it. Adding ATIF / SWE-agent later is a one-line addition
here plus a new prompt + validator branch."""

FORMAT_EXTENSIONS: dict[str, str] = {
    FORMAT_CLAUDE_CODE: ".jsonl",
    FORMAT_CODEX: ".jsonl",
    FORMAT_CURSOR: ".cursor.json",
}
"""File extension per format. Mirrors what the existing ingest
scripts expect when scanning ``data/raw/<format>/``."""


# ---------------------------------------------------------------------------
# Failure-mode buckets.
# ---------------------------------------------------------------------------

NON_SUCCESS_LABELS: tuple[str, ...] = tuple(
    m.value for m in FailureLabel if m not in {FailureLabel.SUCCESS, FailureLabel.PARTIAL_SUCCESS}
)
"""All 9 labels in the taxonomy that aren't success/partial_success.
Used as the candidate pool for even-split sampling AND as the
allow-list when validating custom label distributions.

``unclear`` is intentionally included here even though the default
distribution skips it -- callers who want ambiguous traces can
override the distribution to include it."""


# The user-specified default failure-mode distribution for the Slice 7
# demo corpus. Sums to 1.0; values are interpreted as fractions of the
# total trace count. Labels not present in this dict are NEVER sampled
# (e.g. tool_spamming / incomplete / unclear under this default).
#
# Rationale for the shape:
#   * 30% successes for "good agent" examples (NL-search needs both
#     sides of the failure boundary visible).
#   * 5% partial_success to expose the "almost worked" cluster.
#   * 15% loop -- the most visually striking failure mode in a demo,
#     so over-represent slightly.
#   * 10% each on the five "diagnostic" failures (gave_up,
#     hallucinated_api, misread_error, environment_issue,
#     wrong_approach) for an even bench across the categories most
#     reviewers care about.
DEFAULT_LABEL_DISTRIBUTION: dict[str, float] = {
    FailureLabel.SUCCESS.value: 0.30,
    FailureLabel.PARTIAL_SUCCESS.value: 0.05,
    FailureLabel.LOOP.value: 0.15,
    FailureLabel.GAVE_UP.value: 0.10,
    FailureLabel.HALLUCINATED_API.value: 0.10,
    FailureLabel.MISREAD_ERROR.value: 0.10,
    FailureLabel.ENVIRONMENT_ISSUE.value: 0.10,
    FailureLabel.WRONG_APPROACH.value: 0.10,
}


_SUCCESS_LIKE_LABELS: frozenset[str] = frozenset(
    {FailureLabel.SUCCESS.value, FailureLabel.PARTIAL_SUCCESS.value}
)
_ALL_VALID_LABELS: frozenset[str] = frozenset(m.value for m in FailureLabel)


# ---------------------------------------------------------------------------
# Task domain pool.
# ---------------------------------------------------------------------------
#
# The "task" is whatever the user asked the agent to do. We sample
# from a small fixed pool so generated traces span common coding-
# agent scenarios (bug fix, refactor, new feature, etc.) rather than
# the model picking the same "fix this python error" task 500 times
# in a row. Each entry is a one-line topic; the prompt embeds it
# verbatim and the model elaborates.

TASK_TOPICS: tuple[str, ...] = (
    "Fix a failing pytest in a Python web service.",
    "Add a new REST endpoint with input validation and tests.",
    "Investigate a memory leak in a long-running daemon.",
    "Migrate a SQLAlchemy model column without losing data.",
    "Refactor a 300-line function into smaller helpers.",
    "Debug a flaky CI integration test.",
    "Set up a new GitHub Actions workflow for the repo.",
    "Add type hints to an untyped module.",
    "Investigate why production latency spiked overnight.",
    "Implement a small CLI subcommand with argparse.",
    "Wire a new feature flag through a service.",
    "Fix a regex that's matching too greedily.",
    "Convert a class-based view to a function-based one.",
    "Add a Redis cache layer with proper invalidation.",
    "Resolve a circular import between two modules.",
    "Track down which migration introduced a column-not-null bug.",
    "Add retry-with-backoff to an HTTP client call.",
    "Hunt down a race condition in a multi-threaded queue worker.",
    "Update a dependency that just shipped a breaking change.",
    "Improve error messages on a CLI tool for unknown subcommands.",
    "Add structured logging to a service that's using print().",
    "Inline an over-engineered abstraction layer.",
    "Implement OAuth2 PKCE flow in a small client library.",
    "Replace a hand-rolled config parser with pydantic-settings.",
    "Debug why a Docker container exits immediately on start.",
)
"""25 task topics covering the common coding-agent scenarios. The
sample size is intentionally small relative to the corpus -- we
WANT some clustering by task type so the embedding NN has obvious
neighbourhoods to expose."""


# ---------------------------------------------------------------------------
# TraceSpec dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceSpec:
    """One trace's generation recipe.

    Fields:

    * ``output_format``: one of :data:`ALL_FORMATS`. Decides which
      adapter validates and which file extension lands on disk.
    * ``primary_failure``: the headline failure mode (or
      ``"success"`` / ``"partial_success"`` for non-failing traces).
      Surfaced into the trace's ``synthetic_failure_mode`` field.
    * ``secondary_failures``: 0-2 additional failure modes that the
      trace should also exhibit. Allows the generator to produce
      multi-mode complex failures the user asked for. Empty for
      success traces.
    * ``task``: the task topic the trace acts out. Embeds verbatim
      in the prompt.
    * ``target_step_count``: how many trajectory steps the model
      should aim for. Embedded as guidance; we accept anything in
      ``[target * 0.5, target * 2]`` post-hoc.
    * ``is_failure``: convenience flag mirroring whether
      ``primary_failure`` is a non-success label.

    Frozen so specs can hash into a set if the orchestrator ever
    wants to dedupe.
    """

    output_format: str
    primary_failure: str
    task: str
    target_step_count: int
    secondary_failures: tuple[str, ...] = field(default_factory=tuple)
    is_failure: bool = True


# ---------------------------------------------------------------------------
# Sampling.
# ---------------------------------------------------------------------------


def _balanced_format_assignments(count: int, rng: random.Random) -> list[str]:
    """Return a ``count``-long list of formats with the closest-possible
    even split, in shuffled order.

    A naive ``rng.choice(ALL_FORMATS)`` produces noticeable lumps for
    counts in the hundreds (one format ending up ~30 traces ahead of
    another). The "fill quotas then shuffle" approach guarantees the
    final per-format count is within ``+/-1`` of ``count / len(ALL_FORMATS)``.
    """
    base, remainder = divmod(count, len(ALL_FORMATS))
    assignments: list[str] = []
    for idx, fmt in enumerate(ALL_FORMATS):
        # The first ``remainder`` formats get one extra trace; the order
        # is fixed so seeded runs are reproducible.
        quota = base + (1 if idx < remainder else 0)
        assignments.extend([fmt] * quota)
    rng.shuffle(assignments)
    return assignments


def _validate_distribution(distribution: dict[str, float]) -> None:
    """Reject obviously-bad distribution dicts up front.

    Two gates:
    * Every key must be a recognised :class:`FailureLabel` value
      (typos otherwise silently kill that label's sampling).
    * Probabilities must sum to within +/-0.01 of 1.0; below that we
      lose specs, above it we overshoot ``count``.

    Negative or NaN weights also rejected. Allowed for keys to be
    absent (those labels just never get sampled).
    """
    if not distribution:
        raise ValueError("label_distribution must be non-empty")
    unknown = set(distribution) - _ALL_VALID_LABELS
    if unknown:
        raise ValueError(
            f"label_distribution contains unknown labels: {sorted(unknown)}. "
            f"Valid: {sorted(_ALL_VALID_LABELS)}"
        )
    for label, weight in distribution.items():
        if not isinstance(weight, (int, float)) or weight != weight:  # NaN check
            raise ValueError(f"distribution[{label}] must be a real number, got {weight!r}")
        if weight < 0:
            raise ValueError(f"distribution[{label}] cannot be negative: {weight}")
    total = sum(distribution.values())
    if not 0.99 <= total <= 1.01:
        raise ValueError(
            f"label_distribution weights must sum to ~1.0 (got {total:.4f})"
        )


def _distribution_from_failure_ratio(failure_ratio: float) -> dict[str, float]:
    """Build the legacy "even split across all 9 failure labels" distribution.

    Used as the fallback when no explicit :data:`label_distribution`
    is provided. Preserves the original behaviour for callers (and
    tests) that pass only ``failure_ratio``.

    Success and partial_success split 64/36 within the non-failure
    bucket, matching the original code's heuristic.
    """
    fail_each = failure_ratio / len(NON_SUCCESS_LABELS)
    non_fail = 1.0 - failure_ratio
    success_frac = non_fail * 0.64
    partial_frac = non_fail - success_frac
    dist: dict[str, float] = {
        FailureLabel.SUCCESS.value: success_frac,
        FailureLabel.PARTIAL_SUCCESS.value: partial_frac,
    }
    for label in NON_SUCCESS_LABELS:
        dist[label] = fail_each
    return dist


def _allocate_counts(
    distribution: dict[str, float], total: int
) -> dict[str, int]:
    """Largest-remainder method to allocate ``total`` across labels.

    Plain ``round(weight * total)`` accumulates rounding error and
    can produce counts that sum to ``total +/- k`` for k up to the
    number of buckets. The Hare-quota / largest-remainder method
    guarantees ``sum(counts.values()) == total`` exactly: every
    label gets ``floor(weight * total)`` first, then the unallocated
    remainder ``total - sum(floors)`` is distributed one-by-one to
    the labels with the largest fractional remainders.

    Deterministic and seed-free (sorts by fractional remainder
    desc, breaks ties by label name) so the same distribution +
    total always produces the same counts.
    """
    floors: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
    for label, weight in distribution.items():
        exact = weight * total
        floor = int(exact)
        floors[label] = floor
        fractions.append((exact - floor, label))
    assigned = sum(floors.values())
    leftover = total - assigned
    if leftover > 0:
        # Tie-break by label name for determinism.
        fractions.sort(key=lambda pair: (-pair[0], pair[1]))
        for _, label in fractions[:leftover]:
            floors[label] += 1
    elif leftover < 0:
        # Over-allocated by rounding (rare; happens only when
        # distribution sums > 1.0 by exactly 0.005 + buckets). Trim
        # from labels with smallest fractional remainder so the
        # over-rounded buckets give back first.
        fractions.sort(key=lambda pair: (pair[0], pair[1]))
        for _, label in fractions[: -leftover]:
            if floors[label] > 0:
                floors[label] -= 1
    return floors


def _sample_secondary_failures(
    primary: str,
    rng: random.Random,
    *,
    active_labels: tuple[str, ...],
) -> tuple[str, ...]:
    """Decide how many secondary modes a failing trace has + which ones.

    Distribution (unchanged from the slice-7 baseline):
    * 50% of failing traces: no secondary mode (single-mode failure).
    * 30%: exactly 1 secondary mode.
    * 20%: 2 secondary modes (complex multi-mode failures).

    Secondary modes are drawn without replacement from
    ``active_labels`` minus ``primary`` so they never duplicate the
    headline AND never include labels the caller's distribution
    doesn't otherwise sample (a "loop + tool_spamming" trace would
    confuse a corpus whose distribution explicitly drops
    tool_spamming).

    If ``active_labels`` has fewer than 2 entries, secondary failures
    are impossible -- return empty tuple regardless of the roll.
    """
    if len(active_labels) < 2:
        return ()
    roll = rng.random()
    if roll < 0.50:
        return ()
    n_secondary = 1 if roll < 0.80 else 2

    pool = [label for label in active_labels if label != primary]
    rng.shuffle(pool)
    return tuple(pool[:n_secondary])


def sample_specs(
    count: int,
    *,
    label_distribution: dict[str, float] | None = None,
    failure_ratio: float = 0.75,
    target_step_count: int = 25,
    seed: int = 42,
    task_pool: Sequence[str] = TASK_TOPICS,
) -> list[TraceSpec]:
    """Generate a reproducible list of ``count`` :class:`TraceSpec` instances.

    Distribution invariants (verified by the test suite):

    * Total length == ``count`` (largest-remainder rounding
      guarantees exact match, not "+/- a few" from naive ``round``).
    * Format split is even within +/-1 across :data:`ALL_FORMATS`.
    * Per-label counts match ``label_distribution * count`` within
      one (the remainder slot per label).
    * Labels absent from ``label_distribution`` are NEVER sampled.
    * Same ``(count, seed, distribution)`` -> byte-identical output.

    Parameters
    ----------
    count
        Total number of specs to produce.
    label_distribution
        Optional dict mapping :class:`FailureLabel` values to fractions
        of the corpus. Must sum to ~1.0; unknown labels rejected.
        When ``None`` (the bare default), :data:`DEFAULT_LABEL_DISTRIBUTION`
        is used -- the user-specified Slice 7 distribution
        (30% success, 5% partial, 15% loop, 10% each on 5 diagnostic
        failures). Pass an explicit dict to override per-corpus.
    failure_ratio
        Legacy knob: only consulted when ``label_distribution`` is
        explicitly passed as the sentinel ``{}`` (empty dict). In that
        case, an even-split-across-9-labels distribution is built
        from this ratio. Kept for backwards compatibility with any
        old call sites; new code should pass ``label_distribution``
        directly.
    target_step_count
        Per-spec hint for "how long should this trace be?" Passed
        verbatim to the prompt. Most prompts will land within a factor
        of two; the validator's ``>=5 steps`` gate catches degenerate
        cases.
    seed
        RNG seed. The full run uses 42 by default; tests pin alternate
        seeds.
    task_pool
        Source of task topics. Tests substitute a tiny pool to assert
        the sampling logic without depending on the production list.
    """
    if count < 0:
        raise ValueError("count must be non-negative")

    # Resolve which distribution to use:
    # * Explicit dict (including the default) -> use as-is.
    # * Explicit empty dict {} -> backwards-compat: build from
    #   failure_ratio (legacy even-split-across-9 behaviour).
    # * None -> default demo distribution.
    if label_distribution is None:
        distribution = dict(DEFAULT_LABEL_DISTRIBUTION)
    elif not label_distribution:
        if not 0.0 <= failure_ratio <= 1.0:
            raise ValueError("failure_ratio must be in [0, 1]")
        distribution = _distribution_from_failure_ratio(failure_ratio)
    else:
        distribution = dict(label_distribution)
    _validate_distribution(distribution)

    rng = random.Random(seed)

    # Allocate counts deterministically via largest-remainder.
    counts = _allocate_counts(distribution, count)

    # Build the flat label sequence: one entry per spec slot.
    label_sequence: list[str] = []
    for label, n in counts.items():
        label_sequence.extend([label] * n)
    rng.shuffle(label_sequence)

    # Format assignments split EVENLY across the whole population,
    # not per-bucket, so we don't end up with all-codex successes by
    # accident.
    format_assignments = _balanced_format_assignments(count, rng)

    # Secondary failures can only draw from labels actually present in
    # the distribution as non-success modes. Otherwise a loop trace's
    # secondary could land on a label the user explicitly omitted.
    active_failure_labels: tuple[str, ...] = tuple(
        sorted(
            label
            for label, weight in distribution.items()
            if weight > 0 and label not in _SUCCESS_LIKE_LABELS
        )
    )

    specs: list[TraceSpec] = []
    for idx, primary in enumerate(label_sequence):
        is_failure = primary not in _SUCCESS_LIKE_LABELS
        secondary = (
            _sample_secondary_failures(
                primary, rng, active_labels=active_failure_labels
            )
            if is_failure
            else ()
        )
        task = rng.choice(list(task_pool))
        specs.append(
            TraceSpec(
                output_format=format_assignments[idx],
                primary_failure=primary,
                secondary_failures=secondary,
                task=task,
                target_step_count=target_step_count,
                is_failure=is_failure,
            )
        )
    return specs


__all__ = [
    "ALL_FORMATS",
    "DEFAULT_LABEL_DISTRIBUTION",
    "FORMAT_CLAUDE_CODE",
    "FORMAT_CODEX",
    "FORMAT_CURSOR",
    "FORMAT_EXTENSIONS",
    "NON_SUCCESS_LABELS",
    "TASK_TOPICS",
    "TraceSpec",
    "sample_specs",
]
