"""Per-format validation wrapping the production adapters.

The synth pipeline trusts adapters as the source of truth: a trace is
"valid" iff the format's adapter accepts it. This module is a thin
``(ok, error)`` wrapper around ``ClaudeCodeAdapter.to_atif``,
``CodexAdapter.to_atif``, and ``CursorAdapter.to_atif``.

Why not let the generator call adapters directly?

1. The adapters return ``None`` for malformed inputs and *also* log
   the actual reason at WARNING / DEBUG. The generator needs the
   reason as a string so the retry prompt can show the model what
   broke. We capture log records during validation and surface them
   in the error string.
2. The adapters don't enforce business-level constraints (>=5 steps,
   at least one tool call). The synth pipeline rejects degenerate
   traces above that bar even when ATIF would accept them. Putting
   the bar here keeps it shared between dry-run, smoke, and live
   runs.
3. Mocking three adapter classes in tests is more code than mocking
   one validator function. Encapsulating here makes the generator's
   tests simple.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from harbor.models.trajectories import Trajectory
from trace_marketplace.adapters.claude_code import ClaudeCodeAdapter
from trace_marketplace.adapters.codex import CodexAdapter
from trace_marketplace.adapters.cursor import CursorAdapter
from trace_marketplace.synth.spec import (
    FORMAT_CLAUDE_CODE,
    FORMAT_CODEX,
    FORMAT_CURSOR,
)

MIN_STEPS = 5
"""Lower bound for "non-degenerate" trace. Below this the model
basically didn't try (or emitted a stub); we'd rather retry than
keep the row.

5 is a permissive floor -- short successful traces (user prompt +
one read + one edit + one verify + final summary) can hit exactly 5.
Going lower lets degenerate two-step stubs through; going higher
starts rejecting legitimately short success traces."""

MAX_STEPS = 200
"""Upper bound. Sonnet has wandered up to 80+ steps in testing; 200
is well above any realistic ceiling and protects DB blob sizes /
embedding cost from a runaway hallucination."""


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a validator pass.

    Fields:

    * ``ok``: whether the trace cleared all checks (adapter + step
      counts + tool-call presence).
    * ``error``: short reason on failure (the validator's verdict + a
      tail of the adapter's WARNING log records, ~200 chars total).
      Empty on success.
    * ``trajectory``: parsed :class:`Trajectory` on success, ``None``
      on failure. The orchestrator inspects it to extract metadata
      (step count, primary failure mode) for the run manifest.

    Frozen because a validation result is a value, not state to
    mutate.
    """

    ok: bool
    error: str
    trajectory: Trajectory | None


# ---------------------------------------------------------------------------
# Log capture.
# ---------------------------------------------------------------------------


@contextmanager
def _capture_adapter_logs() -> Iterator[list[logging.LogRecord]]:
    """Yield a list that fills with records emitted by adapter loggers.

    The adapters log validation failures at WARNING under their own
    module loggers (``trace_marketplace.adapters.*``). We attach a
    transient ``logging.Handler`` to the parent logger, drain it on
    context exit, and return the records to the caller. We DON'T
    silence the records -- the standard run still surfaces them in
    the CLI output for debugging.
    """
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler(level=logging.WARNING)
    parent = logging.getLogger("trace_marketplace.adapters")
    parent.addHandler(handler)
    try:
        yield records
    finally:
        parent.removeHandler(handler)


def _summarise_records(records: list[logging.LogRecord]) -> str:
    """Concatenate captured adapter log records into a single string.

    Truncated to ~200 chars (we just need enough to know which line
    of the JSONL the adapter rejected). Multiple records joined with
    ``" | "`` so each retry attempt sees a compact reason.
    """
    if not records:
        return ""
    parts: list[str] = []
    for record in records:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            msg = record.msg if isinstance(record.msg, str) else ""
        if msg:
            parts.append(msg)
    joined = " | ".join(parts)
    if len(joined) > 200:
        return joined[:197] + "..."
    return joined


# ---------------------------------------------------------------------------
# Adapter dispatch.
# ---------------------------------------------------------------------------

_ADAPTERS = {
    FORMAT_CLAUDE_CODE: ClaudeCodeAdapter,
    FORMAT_CODEX: CodexAdapter,
    FORMAT_CURSOR: CursorAdapter,
}


def _count_tool_calls(trajectory: Trajectory) -> int:
    """Sum ``len(step.tool_calls or [])`` across all steps."""
    total = 0
    for step in trajectory.steps:
        tool_calls = getattr(step, "tool_calls", None)
        if tool_calls:
            total += len(tool_calls)
    return total


def validate(format_name: str, raw_text: str) -> ValidationResult:
    """Run ``raw_text`` through ``format_name``'s adapter.

    Pipeline:
    1. Dispatch on ``format_name`` to pick an adapter.
    2. Run ``adapter.to_atif`` with adapter-log capture.
    3. If adapter returns ``None``, fail with the captured log tail.
    4. Otherwise enforce ``MIN_STEPS <= len(steps) <= MAX_STEPS`` and
       ``count_tool_calls >= 1``. Failure produces a short reason.

    Parameters
    ----------
    format_name
        One of :data:`ALL_FORMATS`. Unknown formats raise
        :class:`ValueError` -- the orchestrator guarantees this never
        happens, but tests pin the behaviour.
    raw_text
        The bytes Sonnet emitted via the ``submit_trace`` tool.

    Returns
    -------
    :class:`ValidationResult`
        ``ok=True`` only when every gate passes.
    """
    adapter_cls = _ADAPTERS.get(format_name)
    if adapter_cls is None:
        raise ValueError(
            f"Unknown format {format_name!r}; expected one of {sorted(_ADAPTERS)}"
        )

    if not raw_text or not raw_text.strip():
        return ValidationResult(
            ok=False, error="empty raw_text from model", trajectory=None
        )

    adapter = adapter_cls()
    with _capture_adapter_logs() as records:
        try:
            trajectory = adapter.to_atif(raw_text)
        except Exception as exc:  # noqa: BLE001
            # Adapters are meant to never raise; if one does, treat
            # it like a validation failure with the exception text.
            return ValidationResult(
                ok=False,
                error=f"adapter exception: {type(exc).__name__}: {exc}",
                trajectory=None,
            )

    if trajectory is None:
        log_tail = _summarise_records(records)
        reason = "adapter returned None"
        if log_tail:
            reason = f"{reason} ({log_tail})"
        return ValidationResult(ok=False, error=reason, trajectory=None)

    step_count = len(trajectory.steps)
    if step_count < MIN_STEPS:
        return ValidationResult(
            ok=False,
            error=f"too few steps ({step_count} < {MIN_STEPS})",
            trajectory=trajectory,
        )
    if step_count > MAX_STEPS:
        return ValidationResult(
            ok=False,
            error=f"too many steps ({step_count} > {MAX_STEPS})",
            trajectory=trajectory,
        )
    if _count_tool_calls(trajectory) < 1:
        return ValidationResult(
            ok=False,
            error="no tool calls in trajectory (degenerate)",
            trajectory=trajectory,
        )

    return ValidationResult(ok=True, error="", trajectory=trajectory)


__all__ = ["MAX_STEPS", "MIN_STEPS", "ValidationResult", "validate"]
