"""Deterministic structural signals over a parsed ATIF dict.

Each ``signal_*`` function is pure: it takes an already-decoded ATIF
trajectory (the kind of ``dict`` you get from ``json.loads`` on the
``atif`` column) and returns a ``bool``. No SQL, no I/O, no network.

The aggregator :func:`compute_signals` produces a flat ``dict[str, bool]``
that the pipeline JSON-encodes into the ``failure_signals`` column.

Thresholds are module-level constants so they're tunable in one place
once :mod:`scripts.validate_detection` tells us which ones are too
strict / too loose against SWE-agent ground truth.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable thresholds.
# ---------------------------------------------------------------------------

MAX_TURNS_THRESHOLD: int = 50
"""``hit_max_turns`` fires once a trajectory has this many steps. 50 is
the prompt's seed value; tune via the validation script."""

LOOP_REPEAT_THRESHOLD: int = 5
"""``repeated_tool_call_loop`` fires when this many consecutive identical
``(function_name, arguments)`` keys appear back-to-back."""

ENDED_ON_ERROR_WINDOW: int = 3
"""How many tail steps :func:`signal_ended_on_error` inspects. The error
must occur within this window and have no recovering agent step after."""

REPETITION_KEY_PREFIX_CHARS: int = 120
"""When a step has no structured ``tool_calls``, we fall back to its
``message`` content as the step signature, taking this many leading
characters. Picked empirically: long enough to be specific (so two
genuinely different system error messages don't collide), short enough
that templated SWE-agent system responses match when only line numbers
differ."""

_EXIT_CONTEXT_PATTERN: re.Pattern[str] = re.compile(
    r"""(?xi)
    exit_context
  | context_limit_exceeded
  | context\ window\ exceeded
  | maximum\ context\ length
    """
)
"""SWE-agent-style ``trajectory.extra["exit_status"]`` markers that mean
the session terminated because it hit the context window, not because
the task completed. Treated as an end-on-error signal."""

# Compiled regexes (module-scoped so they're built once per import).
_ERROR_PATTERN: re.Pattern[str] = re.compile(
    r"""(?xi)
    (?:^|[\s\n])(?:
        Traceback\ \(most\ recent\ call\ last\)
        | Error:
        | ERROR:
        | exit\ code\ [1-9]
        | command\ failed
        | failed\ with\ status\ [1-9]
        | SyntaxError | TypeError | ValueError | KeyError
        | ModuleNotFoundError | AttributeError | ImportError
        | RuntimeError | AssertionError
    )
    """
)
"""Substring patterns that indicate an error in observation content.

Matched case-insensitively. The ``(?:^|[\\s\\n])`` prefix keeps us from
false-positiving inside compound English words like ``"erroneous"``."""

# Single character class that matches both the straight apostrophe
# (U+0027) and the curly right-single-quote (U+2018, U+2019) that real
# Claude / Codex exports normalise to. A bare ``[''']`` looks like three
# different quotes in source but is actually three copies of U+0027,
# which silently drops the smart-quote handling -- caught by
# ``test_abandonment_handles_smart_quote_apostrophe``.
_APOS = "['\u2018\u2019]"

_GIVE_UP_PATTERN: re.Pattern[str] = re.compile(
    rf"""(?xi)
    (
        I\ (?:cannot|can{_APOS}t|am\ unable|am\ not\ able)
      | I\ don{_APOS}t\ have\ (?:access|the\ ability)
      | this\ (?:is\ beyond|isn{_APOS}t\ something\ I\ can)
      | let\ me\ know\ how\ you{_APOS}d\ like\ to\ proceed
      | I{_APOS}ll\ stop\ here
      | unable\ to\ complete
      | I\ (?:apologize|am\ sorry),?\ (?:but\ )?I\ (?:cannot|can{_APOS}t)
    )
    """
)
"""Plain-language ``gave_up`` indicators in an agent message.

The 6 baseline patterns from the slice brief, with explicit smart-quote
variants so we don't miss curly-apostrophe text from real Claude /
Codex exports."""

# ATIF source values that count as the agent talking.
_AGENT_SOURCES: frozenset[str] = frozenset({"agent", "assistant"})

# Sources whose step signatures participate in loop detection. Including
# ``system`` is what catches SWE-agent's pattern of the same templated
# error message ("Your proposed edit has introduced new syntax error(s)
# ...") repeating across attempts: the agent paraphrases each retry, but
# the system response stays identical.
_LOOP_SCAN_SOURCES: frozenset[str] = frozenset({"agent", "assistant", "system"})


# ---------------------------------------------------------------------------
# Helpers (kept private; never imported elsewhere).
# ---------------------------------------------------------------------------


def _steps(traj: dict[str, Any]) -> list[dict[str, Any]]:
    steps = traj.get("steps")
    if not isinstance(steps, list):
        return []
    return [s for s in steps if isinstance(s, dict)]


def _is_agent_step(step: dict[str, Any]) -> bool:
    return (step.get("source") or "").lower() in _AGENT_SOURCES


def _stringify_result_content(content: Any) -> str:
    """Flatten an ``ObservationResult.content`` into one searchable string.

    Mirrors what the detail-view renderer does: strings pass through,
    list-of-blocks gets concatenated by ``text`` field, anything else
    is JSON-dumped so the regex can still find error markers in raw
    structured payloads.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            else:
                parts.append(json.dumps(block, default=str))
        return "\n".join(parts)
    if content is None:
        return ""
    return json.dumps(content, default=str)


def _tool_call_key(call: dict[str, Any]) -> str:
    """Canonicalise a ``ToolCall`` dict into a string key for loop detection.

    Two calls with the same function name + same arguments produce the
    same key, regardless of how the arguments are nested or ordered.
    """
    name = call.get("function_name") or ""
    args = call.get("arguments")
    try:
        args_str = json.dumps(args, sort_keys=True, default=str)
    except TypeError:
        args_str = repr(args)
    return f"tc::{name}::{args_str}"


def _step_signature(step: dict[str, Any]) -> str | None:
    """Return a comparable key for ``step``, or ``None`` if the step
    carries no detectable repetition signal.

    Preference order matches how different adapters surface "what the
    agent is doing":

    1. **Structured tool call** (Codex, Cursor, Claude Code): the
       ``(function_name, arguments)`` key.
    2. **Message-only** (SWE-agent): the first
       :data:`REPETITION_KEY_PREFIX_CHARS` of the message, stripped
       and lowercased. Catches templated system responses + the
       common "The syntax errors persist..." agent paraphrase that
       a real SWE-agent failure repeats verbatim several times.
    """
    tool_calls = step.get("tool_calls") or []
    if tool_calls and isinstance(tool_calls[0], dict):
        return _tool_call_key(tool_calls[0])
    msg = step.get("message")
    if isinstance(msg, str):
        trimmed = msg.strip()[:REPETITION_KEY_PREFIX_CHARS].lower()
        if trimmed:
            return f"msg::{trimmed}"
    return None


def _trajectory_extra(traj: dict[str, Any]) -> dict[str, Any]:
    extra = traj.get("extra")
    return extra if isinstance(extra, dict) else {}


# ---------------------------------------------------------------------------
# Public signal functions.
# ---------------------------------------------------------------------------


def signal_hit_max_turns(
    traj: dict[str, Any], threshold: int = MAX_TURNS_THRESHOLD
) -> bool:
    """True when the trajectory has ``threshold`` or more steps.

    A high step count is the cheapest possible "something went wrong"
    signal: well-resolved tasks usually wrap up in a dozen steps.
    """
    return len(_steps(traj)) >= threshold


def signal_repeated_tool_call_loop(
    traj: dict[str, Any], repeat_threshold: int = LOOP_REPEAT_THRESHOLD
) -> bool:
    """True when the same step signature appears ``repeat_threshold``
    times in immediate succession **within a single source**.

    Streaks are tracked per-source so SWE-agent's interleaved agent /
    system pattern still counts: the agent paraphrases each retry, but
    the system response stays templated, and we want the templated
    system messages' streak to grow even though agent steps appear
    between them. Tracking globally would silently reset on every
    source switch (caught by ``test_repeated_tool_call_loop_catches_
    swe_agent_system_message_repetition``).

    "Step signature" is the structured ``(function_name, arguments)``
    pair when the step carries tool_calls (Codex / Cursor / Claude Code
    shape), or the first ~120 chars of the message otherwise (SWE-agent
    shape, where tool actions are embedded in agent messages and errors
    come back as templated system messages).
    """
    last_key: dict[str, str | None] = {}
    streak: dict[str, int] = {}
    for step in _steps(traj):
        src = (step.get("source") or "").lower()
        if src not in _LOOP_SCAN_SOURCES:
            continue
        key = _step_signature(step)
        if key is None:
            last_key[src] = None
            streak[src] = 0
            continue
        if key == last_key.get(src):
            streak[src] = streak.get(src, 0) + 1
        else:
            last_key[src] = key
            streak[src] = 1
        if streak[src] >= repeat_threshold:
            return True
    return False


def _step_error_text(step: dict[str, Any]) -> str | None:
    """Return error-marker text from ``step`` if any, else ``None``.

    Three places we look, in order:

    1. Observation results (Codex / Cursor / Claude Code shape).
    2. The step's ``message`` field (SWE-agent shape -- both the
       system "Your proposed edit introduced syntax error(s)" template
       and agent messages that quote error output land here).
    3. Tool-call arguments / output buffers -- skipped: ``arguments``
       look like user code that may legitimately contain the word
       ``Error:``.
    """
    observation = step.get("observation")
    if isinstance(observation, dict):
        for result in observation.get("results") or []:
            if not isinstance(result, dict):
                continue
            text = _stringify_result_content(result.get("content"))
            if text and _ERROR_PATTERN.search(text):
                return text
    msg = step.get("message")
    if isinstance(msg, str) and msg and _ERROR_PATTERN.search(msg):
        return msg
    return None


def signal_ended_on_error(
    traj: dict[str, Any], window: int = ENDED_ON_ERROR_WINDOW
) -> bool:
    """True when an error marker appears within the last ``window``
    steps OR the trajectory exit_status indicates a context-window
    blow-up, AND no recovering agent step follows the error.

    Three failure shapes the rule covers:

    * Codex / Cursor / Claude Code -- error in the final observation's
      result content (Traceback, ``Error:``, common exception names).
    * SWE-agent -- templated error in the final system message
      (``"Your proposed edit introduced syntax error(s)..."`` etc.).
    * Hard wall -- ``trajectory.extra["exit_status"]`` contains
      ``exit_context`` / ``context_limit_exceeded`` (agent ran out
      of context, not a clean termination).
    """
    # First, the cheap structural signal: trajectory-level exit_status.
    exit_status = _trajectory_extra(traj).get("exit_status")
    if isinstance(exit_status, str) and _EXIT_CONTEXT_PATTERN.search(exit_status):
        return True

    steps = _steps(traj)
    if not steps:
        return False
    tail = steps[-window:] if window > 0 else steps
    last_error_idx: int | None = None
    for i, step in enumerate(tail):
        if _step_error_text(step) is not None:
            last_error_idx = i
    if last_error_idx is None:
        return False
    # Any agent step strictly after the error in the tail window counts
    # as a recovery attempt; nothing after -> we ended on the error.
    for later in tail[last_error_idx + 1 :]:
        if _is_agent_step(later):
            return False
    return True


def signal_abandonment(traj: dict[str, Any]) -> bool:
    """True when the trajectory's final agent message contains one of
    the give-up patterns.

    Limited to the final agent message on purpose: agents often
    acknowledge limits mid-trace ("I can't read this file, let me try
    another") and then recover. Only the trailing give-up is a real
    abandonment signal.
    """
    last_message: str | None = None
    for step in _steps(traj):
        if not _is_agent_step(step):
            continue
        msg = step.get("message")
        if isinstance(msg, str) and msg.strip():
            last_message = msg
    if last_message is None:
        return False
    return _GIVE_UP_PATTERN.search(last_message) is not None


def signal_incomplete_session(traj: dict[str, Any]) -> bool:
    """True when the last agent step issues a tool call that never gets
    an observation back.

    Indicates a session that died mid-tool-execution: API hang-up,
    crash, user-driven abort. Detecting "no observation after" requires
    finding any step (agent OR observation) following the unmatched
    tool-call step.
    """
    steps = _steps(traj)
    if not steps:
        return False
    # Find the last agent step that issued at least one tool call.
    last_tool_call_idx: int | None = None
    for i in range(len(steps) - 1, -1, -1):
        step = steps[i]
        if not _is_agent_step(step):
            continue
        tool_calls = step.get("tool_calls") or []
        if tool_calls:
            last_tool_call_idx = i
            break
    if last_tool_call_idx is None:
        return False
    # Any step after that with a non-empty observation.results closes
    # the contract; ATIF adapters that pair calls+results into one step
    # (the Codex pattern) also count -- check the originating step's
    # OWN observation.
    originating = steps[last_tool_call_idx]
    own_obs = originating.get("observation") or {}
    if isinstance(own_obs, dict) and (own_obs.get("results") or []):
        return False
    for later in steps[last_tool_call_idx + 1 :]:
        obs = later.get("observation") or {}
        if isinstance(obs, dict) and (obs.get("results") or []):
            return False
    return True


# ---------------------------------------------------------------------------
# Aggregator.
# ---------------------------------------------------------------------------


def compute_signals(traj: dict[str, Any]) -> dict[str, bool]:
    """Run every public signal on a trajectory; return a flat dict.

    Keys mirror the function names (minus the ``signal_`` prefix) so
    the JSON-encoded ``failure_signals`` column reads like a checklist:

        {"hit_max_turns": false, "repeated_tool_call_loop": true, ...}
    """
    return {
        "hit_max_turns": signal_hit_max_turns(traj),
        "repeated_tool_call_loop": signal_repeated_tool_call_loop(traj),
        "ended_on_error": signal_ended_on_error(traj),
        "abandonment": signal_abandonment(traj),
        "incomplete_session": signal_incomplete_session(traj),
    }


def any_signal_fired(signals: dict[str, bool]) -> bool:
    """Aggregate predicate: ``True`` when at least one signal fired."""
    return any(bool(v) for v in signals.values())


# ---------------------------------------------------------------------------
# Persistence helper.
# ---------------------------------------------------------------------------


def persist_signals_for_trace(
    conn: sqlite3.Connection, trace_id: str, atif_text: str | None
) -> dict[str, bool] | None:
    """Compute structural signals for one trace and write them to the DB.

    Shared by :mod:`scripts.detect_failures` (batch pass over every
    unannotated row) and :mod:`trace_marketplace.ui.upload_view` (inline
    pass after a fresh upload). Centralising the SQL + JSON-encode + has_error
    rules in one place stops the two callers from drifting; the older
    pre-extraction version had the same UPDATE statement copy-pasted into
    ``detect_failures._process_one`` and was a foot-gun for the upload path.

    Behaviour:

    * ``atif_text is None`` or fails to ``json.loads`` -> log + return
      ``None``. Caller decides whether that's fatal (it isn't, for the
      batch pass; it's a no-op for upload since unknown-format rows
      already have ``atif`` NULL by design).
    * Decoded payload isn't a dict -> same as above.
    * Otherwise: compute signals, persist them as a sorted JSON blob,
      and fill ``has_error`` from ``any_signal_fired`` ONLY when the
      column was previously NULL (``COALESCE``). Benchmark ground-truth
      ``has_error`` values (e.g. SWE-agent's ``target``-derived 0/1)
      are preserved exactly because they're written by the ingest
      pipeline *before* this helper runs.

    Does NOT call ``conn.commit()``. Callers control transaction scope
    so multi-file uploads can batch commits and the batch script's
    per-row commit-cadence stays explicit.
    """
    if atif_text is None:
        log.debug(
            "persist_signals_for_trace: trace %s has NULL atif; skipping", trace_id
        )
        return None
    try:
        traj = json.loads(atif_text)
    except json.JSONDecodeError:
        log.warning(
            "persist_signals_for_trace: trace %s has malformed ATIF JSON; skipping",
            trace_id,
        )
        return None
    if not isinstance(traj, dict):
        log.warning(
            "persist_signals_for_trace: trace %s ATIF decoded to non-dict; skipping",
            trace_id,
        )
        return None

    signals = compute_signals(traj)
    fired_int = int(any_signal_fired(signals))
    conn.execute(
        """
        UPDATE traces
           SET failure_signals = ?,
               has_error = COALESCE(has_error, ?)
         WHERE id = ?
        """,
        (json.dumps(signals, sort_keys=True), fired_int, trace_id),
    )
    return signals


__all__ = [
    "ENDED_ON_ERROR_WINDOW",
    "LOOP_REPEAT_THRESHOLD",
    "MAX_TURNS_THRESHOLD",
    "any_signal_fired",
    "compute_signals",
    "persist_signals_for_trace",
    "signal_abandonment",
    "signal_ended_on_error",
    "signal_hit_max_turns",
    "signal_incomplete_session",
    "signal_repeated_tool_call_loop",
]
