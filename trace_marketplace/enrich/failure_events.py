"""Multi-event failure-mode extractor backed by Claude Opus 4.7.

A trace can hit several distinct failure modes during one session. The
slice-4 judge collapsed each trace into a single ``FailureLabel``, which
loses information about (a) compound failures (loop -> hallucinate -> give
up) and (b) recoveries (agent hit a wall, then escaped it). This module's
:func:`extract_events` returns a list of :class:`FailureEvent` objects
per trace -- one per distinct wall, with per-event recovery info.

Why a separate module from ``failure_judge.py``?

* The prompt schema is different (list output vs. single-label tool-use).
* The model is different (Opus 4.7 vs. Sonnet 4.6) and prices differ.
* Re-using ``judge_trace`` would require a feature flag deep inside the
  call site, which makes the prompt + retry semantics harder to reason
  about. Keeping them as siblings under ``enrich/`` lets each call site
  pick the right granularity.

The trajectory rendering (compressed conversation with step IDs) is the
ONE thing we share, via :func:`failure_judge._render_conversation` --
that helper is already battle-tested and there's no value in forking it.

Public API:

* :class:`FailureEvent`, :class:`EventsResult` -- typed return shapes.
* :func:`extract_events` -- single trace -> list of events (one Opus call,
  retry once on schema failure, tool-use forced).
* :func:`summarize_label` -- collapse events into the most-severe
  unrecovered label (or ``SUCCESS`` if everything recovered).
* :func:`derive_ultimately_succeeded` -- compute the outcome column from
  the event list, preserving an existing benchmark value.
* :func:`estimate_cost_usd` -- pricing helper for the dry-run script.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

from anthropic import Anthropic

from trace_marketplace.enrich.failure_judge import _render_conversation
from trace_marketplace.enrich.failure_taxonomy import (
    FailureLabel,
    LABEL_DESCRIPTIONS,
    labels_with_descriptions,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model + pricing constants.
# ---------------------------------------------------------------------------

EVENTS_MODEL: str = "claude-opus-4-7"
"""Pinned model slug. Opus 4.7 was the user's choice for slice 10: it
handles the multi-event reasoning -- spotting recoveries, ranking
severity, writing actionable summaries -- noticeably better than Sonnet
on hand-graded spot checks. Cost is acceptable at our corpus size
(~350 calls * ~$0.20/call = ~$70)."""

EVENTS_TEMPERATURE: float | None = None
"""Opus 4.7 is a reasoning model and the API rejects the ``temperature``
parameter with ``'`temperature` is deprecated for this model.'`` (HTTP
400). We pin this to ``None`` so :func:`_call_once` knows to omit it
entirely. The provider-side default is already deterministic enough
for our event-extraction use case -- empirically reruns produce
identical event lists for the same trace.

Left as a module-level constant rather than dropped so the slice-4
judge module + this module read the same way (each pins a sampling
intent at the top of the file)."""

EVENTS_MAX_TOKENS: int = 2500
"""Cap on the model's response. Sized for multi-event output:

* A trace with ~5 events averages ~1.5K tokens (label + step range +
  recovery summary per event).
* Worst case is ~10-15 distinct events with rich recovery summaries,
  which we cap at 2.5K to leave headroom for the rare long-tail trace
  without blowing the per-call cost ceiling."""

INPUT_PRICE_PER_MTOK: float = 15.00
"""Opus 4.7 input token price in USD per million tokens. Updated
manually when Anthropic changes pricing."""

OUTPUT_PRICE_PER_MTOK: float = 75.00
"""Opus 4.7 output token price in USD per million tokens."""


# ---------------------------------------------------------------------------
# Severity ordering for ``summarize_label``.
# ---------------------------------------------------------------------------

SEVERITY_ORDER: tuple[FailureLabel, ...] = (
    # Cognitive failures top the list: an agent that hallucinated a
    # nonexistent API or misread the error has the deepest semantic
    # confusion and that's what we want a one-word summary to flag.
    FailureLabel.HALLUCINATED_API,
    FailureLabel.MISREAD_ERROR,
    FailureLabel.WRONG_APPROACH,
    # Behavioural failures next: the agent could have recovered but
    # didn't -- still useful to surface but less diagnostic than the
    # cognitive ones.
    FailureLabel.GAVE_UP,
    FailureLabel.TOOL_SPAMMING,
    FailureLabel.LOOP,
    # External / setup failures rank below behavioural because they're
    # often not the agent's fault.
    FailureLabel.ENVIRONMENT_ISSUE,
    FailureLabel.INCOMPLETE,
    # Soft-positive / neutral / clean classes go at the bottom; they
    # only win the summary slot when nothing more diagnostic fired.
    FailureLabel.PARTIAL_SUCCESS,
    FailureLabel.UNCLEAR,
    FailureLabel.SUCCESS,
)
"""Severity ranking used by :func:`summarize_label` to pick the
representative label from a multi-event list. First entry = most severe.

Locked by the slice 10 plan; keep this in sync with the plan's
``Locked decisions`` block when extending the taxonomy."""

_SEVERITY_INDEX: dict[FailureLabel, int] = {
    label: idx for idx, label in enumerate(SEVERITY_ORDER)
}
"""Reverse lookup so :func:`summarize_label` is O(n) instead of O(n*m).
Pre-built at module import so callers don't pay the construction cost
per call."""


# ---------------------------------------------------------------------------
# Typed return shapes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureEvent:
    """One distinct failure wall in a trace.

    ``step_start`` / ``step_end`` are 1-based indices into the trace's
    step list (the same numbering the UI shows in step badges).
    ``recovered`` is True when the agent subsequently succeeded at the
    sub-goal that this event blocked; ``recovery_step`` and
    ``recovery_summary`` are populated iff ``recovered=True``.

    ``recovery_summary`` is a single concrete sentence describing what
    the agent did to escape -- this is the actionable advice surfaced
    in the detail-view "Recoveries from similar failures" panel, so the
    prompt asks the model to write it for other agents to read.
    """

    label: FailureLabel
    step_start: int
    step_end: int
    recovered: bool
    recovery_step: int | None
    recovery_summary: str | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form for storage in the ``failure_events`` column."""
        return {
            "label": self.label.value,
            "step_start": int(self.step_start),
            "step_end": int(self.step_end),
            "recovered": bool(self.recovered),
            "recovery_step": (
                int(self.recovery_step) if self.recovery_step is not None else None
            ),
            "recovery_summary": self.recovery_summary,
        }


@dataclass(frozen=True)
class EventsResult:
    """Outcome of one :func:`extract_events` call.

    ``input_tokens`` / ``output_tokens`` mirror what the Anthropic SDK
    returns and are aggregated by the batch script for total-spend
    accounting.
    """

    events: list[FailureEvent]
    input_tokens: int
    output_tokens: int


class EventsExtractionError(RuntimeError):
    """Raised when :func:`extract_events` can't produce a valid event
    list even after the single allowed retry."""


# ---------------------------------------------------------------------------
# Tool-use schema -- enum-constrained list output.
# ---------------------------------------------------------------------------


_LABEL_ENUM_VALUES: list[str] = [m.value for m in FailureLabel]

_EVENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "label",
        "step_start",
        "step_end",
        "recovered",
        "recovery_step",
        "recovery_summary",
    ],
    "properties": {
        "label": {
            "type": "string",
            "enum": _LABEL_ENUM_VALUES,
            "description": (
                "The failure mode for this event. Use 'success' ONLY when "
                "you genuinely want to record a successful sub-goal (rare; "
                "usually you'd just emit an empty events list for a clean "
                "trace)."
            ),
        },
        "step_start": {
            "type": "integer",
            "minimum": 1,
            "description": "1-based step index where this failure mode first appeared.",
        },
        "step_end": {
            "type": "integer",
            "minimum": 1,
            "description": (
                "1-based step index where this failure mode was last "
                "observed (inclusive). If the event spans one step, "
                "step_end == step_start."
            ),
        },
        "recovered": {
            "type": "boolean",
            "description": (
                "True if the agent subsequently succeeded at the sub-goal "
                "that this event blocked. False if the wall stuck."
            ),
        },
        "recovery_step": {
            "type": ["integer", "null"],
            "description": (
                "1-based step index where recovery was achieved. Null when "
                "recovered=false."
            ),
        },
        "recovery_summary": {
            "type": ["string", "null"],
            "description": (
                "Single concrete sentence describing what specific action "
                "or change in approach got the agent past this wall. This "
                "is shown to OTHER agents facing similar failures, so make "
                "it actionable and specific (name tools, files, error "
                "keywords). Null when recovered=false. Max ~200 chars."
            ),
        },
    },
}

_RECORD_EVENTS_TOOL: dict[str, Any] = {
    "name": "record_failure_events",
    "description": (
        "Record the list of distinct failure events observed in this "
        "trace. An empty list is valid and means the trace was clean. "
        "Each event has its own step range, recovery status, and (when "
        "recovered) a one-sentence summary of what the agent did to "
        "escape."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["events"],
        "properties": {
            "events": {
                "type": "array",
                "items": _EVENT_SCHEMA,
                "description": (
                    "Ordered list of failure events. Empty list means "
                    "the agent had no notable failures."
                ),
            }
        },
    },
}


# ---------------------------------------------------------------------------
# Prompt construction.
# ---------------------------------------------------------------------------


def _build_user_prompt(traj: dict[str, Any]) -> str:
    """Build the Opus 4.7 user prompt for one trace.

    Same compression strategy as the slice-4 judge (latest 30 steps in
    full, earlier steps replaced by a placeholder). We add explicit
    1-based step numbering so the model can cite step ranges that the
    UI's step badges line up with.
    """
    agent = traj.get("agent") or {}
    agent_name = agent.get("name", "unknown")
    model_name = agent.get("model_name", "unknown")
    steps = traj.get("steps") or []
    num_steps = len(steps) if isinstance(steps, list) else 0
    num_tool_calls = 0
    if isinstance(steps, list):
        for s in steps:
            if isinstance(s, dict):
                num_tool_calls += len(s.get("tool_calls") or [])

    parts = [
        "You are extracting a structured failure-event timeline from a "
        "coding agent's trace. The output will power a search index of "
        "failure modes plus an in-product 'similar failures that "
        "recovered' panel -- so accuracy on both the events themselves "
        "AND the recovery summaries matters.",
        "",
        "Trace summary:",
        f"- Agent: {agent_name}   Model: {model_name}",
        f"- Steps: {num_steps}    Tool calls: {num_tool_calls}",
        "",
        "Conversation (compressed, latest steps last). Step indices are "
        "1-based and match the UI's step badges:",
        _render_conversation(traj),
        "",
        "Use the `record_failure_events` tool to return the timeline.",
        "",
        "Decision rules:",
        (
            "- One event per DISTINCT failure mode. A loop that then "
            "turned into a give-up is two events (loop, then gave_up), "
            "not one."
        ),
        (
            "- step_start / step_end are 1-based inclusive bounds for the "
            "failure window. If the wall lives at a single step, "
            "step_end == step_start."
        ),
        (
            "- recovered=true means the agent subsequently SUCCEEDED at "
            "the sub-goal this failure was blocking, within the same "
            "trace. A new attempt that hit a different wall is NOT a "
            "recovery for the original event."
        ),
        (
            "- recovery_step is the 1-based index of the step where the "
            "agent broke through (the step where it first executed the "
            "action that worked)."
        ),
        (
            "- recovery_summary is a single concrete sentence describing "
            "what specific action or change in approach the agent used "
            "to get past the failure. This summary will be shown to "
            "OTHER agents facing similar failures, so make it actionable "
            "and specific -- name tools, files, or error keywords. NOT "
            "generic phrases like 'the agent fixed it' or 'the agent "
            "tried again'."
        ),
        (
            "- An empty events list is valid and correct for a trace with "
            "no notable failures. Do NOT invent events to fill the list."
        ),
        (
            "- Use 'unclear' only when the trace is genuinely too short or "
            "too truncated to read. Prefer a confident specific label "
            "when possible."
        ),
        "",
        "Allowed labels (definitions):",
        labels_with_descriptions(),
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Anthropic client protocol (tight contract so tests can mock the SDK).
# ---------------------------------------------------------------------------


class _Messages(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _AnthropicLike(Protocol):
    messages: _Messages


# ---------------------------------------------------------------------------
# Response parsing.
# ---------------------------------------------------------------------------


def _normalise_usage(usage: Any) -> dict[str, int]:
    """SDK ``Usage`` objects expose tokens as attributes; tests pass
    plain dicts. Normalise to a stable dict shape."""
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0}
    if isinstance(usage, dict):
        return {
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
        }
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
    }


def _extract_tool_payload(message_obj: Any) -> dict[str, Any] | None:
    """Pull the ``record_failure_events`` tool_use input out of an
    Anthropic response. Returns ``None`` when no usable block is
    present so the retry path can kick in."""
    content = getattr(message_obj, "content", None) or []
    for block in content:
        block_type = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if block_type != "tool_use":
            continue
        name = getattr(block, "name", None) or (
            block.get("name") if isinstance(block, dict) else None
        )
        if name != _RECORD_EVENTS_TOOL["name"]:
            continue
        payload = getattr(block, "input", None) or (
            block.get("input") if isinstance(block, dict) else None
        )
        if isinstance(payload, dict):
            return payload
    return None


def _validate_event(item: Any) -> FailureEvent | None:
    """Coerce one tool-payload event into a :class:`FailureEvent` or
    return ``None`` to signal validation failure."""
    if not isinstance(item, dict):
        return None
    label_str = item.get("label")
    if not isinstance(label_str, str):
        return None
    try:
        label = FailureLabel(label_str)
    except ValueError:
        return None
    step_start = item.get("step_start")
    step_end = item.get("step_end")
    if not isinstance(step_start, int) or not isinstance(step_end, int):
        return None
    if step_start < 1 or step_end < 1 or step_end < step_start:
        return None
    recovered = item.get("recovered")
    if not isinstance(recovered, bool):
        return None
    recovery_step = item.get("recovery_step")
    recovery_summary = item.get("recovery_summary")
    if recovered:
        # Recovery must happen at or after the last failing step --
        # ``step_end`` is the boundary case where the recovery action
        # coincides with the closing step of the failure span; any
        # value inside the span (or before it) is semantically
        # inconsistent because the failure hasn't ended yet. Empirically
        # Opus 4.7 emits ``recovery_step > step_end`` ~95% of the time
        # with the remaining ~5% landing on ``recovery_step == step_end``
        # (recovery action on the last failing step); we accept both.
        # Anything stricter is too inside the window and indicates a
        # misformed event -- reject so the caller retries.
        if not isinstance(recovery_step, int) or recovery_step < step_end:
            return None
        if not isinstance(recovery_summary, str) or not recovery_summary.strip():
            return None
    else:
        # Non-recovered events: both fields must be null per the tool
        # schema contract. We coerce unconditionally instead of
        # rejecting on stray values -- a single off-spec field
        # shouldn't cost a retry. Empty-string recovery_summary is a
        # known Opus 4.7 quirk; a non-empty summary with
        # ``recovered=false`` is a model-confusion case we silently
        # drop because the recovered=false signal is the authoritative
        # one (we'd render a "not recovered" event with summary text
        # otherwise, which is contradictory in the UI).
        recovery_step = None
        recovery_summary = None
    return FailureEvent(
        label=label,
        step_start=int(step_start),
        step_end=int(step_end),
        recovered=bool(recovered),
        recovery_step=recovery_step,
        recovery_summary=recovery_summary,
    )


def _validate_events_payload(payload: dict[str, Any]) -> list[FailureEvent] | None:
    """Coerce the tool input into a list of :class:`FailureEvent`. Returns
    ``None`` on any structural failure so the caller can retry."""
    events = payload.get("events")
    if not isinstance(events, list):
        return None
    out: list[FailureEvent] = []
    for item in events:
        validated = _validate_event(item)
        if validated is None:
            return None
        out.append(validated)
    return out


# ---------------------------------------------------------------------------
# Single-call + retry orchestration.
# ---------------------------------------------------------------------------


def _call_once(
    client: _AnthropicLike, user_prompt: str
) -> tuple[list[FailureEvent], dict[str, int]]:
    """One SDK request. Returns ``(events, usage)`` or raises
    :class:`EventsExtractionError` on parse failure. Caller drives the
    retry.

    We deliberately do NOT pass ``temperature``: Opus 4.7 is a
    reasoning model and the API rejects the parameter with HTTP 400
    (``'`temperature` is deprecated for this model.'``). The locked
    ``EVENTS_TEMPERATURE = None`` constant documents the intent;
    actual sampling determinism is delegated to the provider's
    reasoning-model defaults.
    """
    create_kwargs: dict[str, Any] = {
        "model": EVENTS_MODEL,
        "max_tokens": EVENTS_MAX_TOKENS,
        "tools": [_RECORD_EVENTS_TOOL],
        "tool_choice": {"type": "tool", "name": _RECORD_EVENTS_TOOL["name"]},
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if EVENTS_TEMPERATURE is not None:
        create_kwargs["temperature"] = EVENTS_TEMPERATURE
    response = client.messages.create(**create_kwargs)
    payload = _extract_tool_payload(response)
    if payload is None:
        raise EventsExtractionError(
            "Model response did not include a record_failure_events tool_use block"
        )
    validated = _validate_events_payload(payload)
    if validated is None:
        raise EventsExtractionError(
            "record_failure_events payload failed schema validation"
        )
    return validated, _normalise_usage(getattr(response, "usage", None))


def extract_events(
    atif: dict[str, Any],
    *,
    client: _AnthropicLike | None = None,
) -> EventsResult:
    """Classify ``atif`` into a list of :class:`FailureEvent` objects.

    A custom ``client`` can be injected for tests; production callers
    pass ``None`` and we instantiate a fresh ``Anthropic()`` whose
    ``ANTHROPIC_API_KEY`` lookup happens via the SDK's normal env
    handling.

    Retries:
    * 1st failure (no tool_use block / schema validation error) -> retry once.
    * 2nd failure -> raise :class:`EventsExtractionError`. The batch
      caller logs the trace id and continues (one bad trace shouldn't
      tank a 350-call run).
    """
    if client is None:
        client = Anthropic()

    user_prompt = _build_user_prompt(atif)
    last_exc: EventsExtractionError | None = None
    for attempt in (1, 2):
        try:
            events, usage = _call_once(client, user_prompt)
        except EventsExtractionError as exc:
            last_exc = exc
            if attempt == 2:
                raise
            log.warning(
                "extract_events: parse failure on attempt %d (%s); retrying",
                attempt,
                exc,
            )
            continue
        return EventsResult(
            events=events,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )
    assert last_exc is not None  # noqa: S101 -- unreachable
    raise last_exc


# ---------------------------------------------------------------------------
# Pure helpers (no API, no DB).
# ---------------------------------------------------------------------------


def summarize_label(events: list[FailureEvent]) -> FailureLabel:
    """Collapse an event list into the most representative single label.

    Rules (in order):

    1. Empty list -> :data:`FailureLabel.SUCCESS`.
    2. At least one unrecovered event -> the most-severe label among
       the unrecovered subset (per :data:`SEVERITY_ORDER`).
    3. All events recovered -> :data:`FailureLabel.SUCCESS` (the trace
       hit walls but ultimately succeeded; the timeline is in
       ``failure_events`` for users who want detail).

    The chosen label is what gets written to the legacy ``failure_label``
    column so the slice-4 UI filters still work post-slice-10.
    """
    if not events:
        return FailureLabel.SUCCESS
    unrecovered = [e for e in events if not e.recovered]
    if not unrecovered:
        return FailureLabel.SUCCESS
    # Pick the lowest-index label per ``_SEVERITY_INDEX`` (lower = more
    # severe in our ranking).
    best_idx = len(SEVERITY_ORDER)  # sentinel (worse than any real label)
    best_label = unrecovered[0].label
    for event in unrecovered:
        idx = _SEVERITY_INDEX.get(event.label, len(SEVERITY_ORDER))
        if idx < best_idx:
            best_idx = idx
            best_label = event.label
    return best_label


def derive_ultimately_succeeded(
    events: list[FailureEvent], existing: int | None
) -> int | None:
    """Compute the ``ultimately_succeeded`` column from an event list.

    Priority rules:

    * If ``existing`` is not None (e.g. SWE-agent ground truth
      backfilled from ``target``) we PRESERVE it -- benchmark labels
      are never overwritten by extractor output.
    * Empty events -> ``1`` (clean trace).
    * Any unrecovered event -> ``0`` (trace ended with an open wall).
    * Otherwise (every event recovered) -> ``1``.

    Returns ``None`` only if the inputs say so; the script callers
    never pass ``existing=None`` and ``events=None`` at the same time.
    """
    if existing is not None:
        return int(existing)
    if not events:
        return 1
    if any(not e.recovered for e in events):
        return 0
    return 1


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Return the dollar cost of one or many extraction calls.

    Source of truth for the dry-run estimator in
    ``scripts/extract_failure_events.py``; same pattern as the slice 4
    judge's helper so the two scripts can't drift on pricing math."""
    return (
        input_tokens / 1_000_000 * INPUT_PRICE_PER_MTOK
        + output_tokens / 1_000_000 * OUTPUT_PRICE_PER_MTOK
    )


def require_api_key() -> None:
    """Refuse to start when ``ANTHROPIC_API_KEY`` is unset.

    Mirrors :func:`failure_judge.require_api_key` so the two enrichment
    scripts have a uniform "give the user a clear instruction" failure
    mode rather than dropping a stacktrace from inside the SDK.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Export it before running this script:\n"
            "    export ANTHROPIC_API_KEY=sk-ant-..."
        )


__all__ = [
    "EVENTS_MAX_TOKENS",
    "EVENTS_MODEL",
    "EVENTS_TEMPERATURE",
    "INPUT_PRICE_PER_MTOK",
    "LABEL_DESCRIPTIONS",
    "OUTPUT_PRICE_PER_MTOK",
    "SEVERITY_ORDER",
    "EventsExtractionError",
    "EventsResult",
    "FailureEvent",
    "derive_ultimately_succeeded",
    "estimate_cost_usd",
    "extract_events",
    "require_api_key",
    "summarize_label",
]
