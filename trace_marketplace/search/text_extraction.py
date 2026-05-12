"""Derive a compact searchable text blob from an ATIF trajectory.

The blob is the input to OpenAI's ``text-embedding-3-small``. The
function below is the single biggest lever on similarity quality:
nearest-neighbour search only finds what the embedding "sees", so
this module's job is to surface the highest-signal slices of a
trace (agent identity, task, failure label, conversational shape)
while staying small enough to be cheap to embed.

Design choices:

* **Deterministic output.** Same ATIF + same enrichment columns must
  produce the same blob, byte-for-byte. This is what makes
  ``source_text_hash`` work as an idempotency key for the backfill
  script. No dict iteration on raw ``atif["agent"]``; we read fields
  explicitly.

* **~4 KB cap (~1 K tokens).** Well under the 8192-token limit for
  ``text-embedding-3-small`` and roughly 1/30th the cost ceiling we
  budgeted. Long traces are tail-biased: failures and label-defining
  behaviour cluster near the end, so we keep the last ``TAIL_STEPS``
  steps in full and add a compact ``(... N earlier steps omitted ...)``
  marker for the head.

* **Tool names get pulled out separately** because they are
  disproportionately information-dense for similarity ("did this
  trace call ``bash`` 12 times?" is a strong cluster signal even
  when the prose drifts).

* **Failure label + reasoning go in the header**, not the body. Two
  reasons: (1) they're concise high-signal labels that should dominate
  the semantic neighbourhood; (2) putting them up front means the
  truncation-to-cap rule below preserves them even on very long traces.

The function is intentionally NOT a method on an ATIF model: keeping
it as a free function operating on plain dicts means we can run it
against the silver-layer ``atif`` TEXT column without re-validating
via Pydantic on every batch.
"""

from __future__ import annotations

from typing import Any

DEFAULT_MAX_CHARS: int = 4096
"""Embedding input cap. ~1024 tokens at 4 chars/token; comfortably
under ``text-embedding-3-small``'s 8192-token context and our
per-trace cost budget."""

TAIL_STEPS: int = 12
"""How many trailing steps to render in full. Failures usually
crystallise in the last 5-15 steps (loops, give-ups, error spirals),
so this is where the embedding gets the most diagnostic signal.
12 steps * ~250 chars = ~3 KB, leaving ~1 KB for header + task."""

STEP_MESSAGE_CAP: int = 220
"""Per-step message truncation. Smaller than the judge's 300 because
embeddings don't need full prose -- they need keywords and shape."""

STEP_RESULT_CAP: int = 140
"""Per-step observation-result truncation. Tool outputs are noisy;
this is enough to keep filenames / error strings without ingesting
whole shell dumps."""

TASK_CHAR_CAP: int = 600
"""How much of the first user message to keep as the "task" line.
600 chars covers the typical SWE-bench problem statement opener
plus a couple of sentences from a Claude-Code session prompt."""


def _truncate(text: str, cap: int) -> str:
    """Strip + cap to ``cap`` chars, appending an ellipsis when cut."""
    text = text.strip()
    if len(text) <= cap:
        return text
    return text[: cap - 1].rstrip() + "\u2026"


def _extract_first_user_message(steps: list[Any]) -> str | None:
    """Find the trajectory's "task" -- the first message from a
    non-agent source (typically ``user`` or ``system``).

    This is where SWE-agent puts the problem statement, where
    Claude-Code traces carry the user's opening prompt, and where
    Codex stores the developer query. None of the adapters lift it
    into a dedicated ``task`` field on ATIF yet, so we scan steps.

    Returns ``None`` if no step has a non-empty message from a
    non-agent source -- the trace will still embed (just without a
    Task line) so the function never raises on incomplete data.
    """
    for step in steps:
        if not isinstance(step, dict):
            continue
        source = step.get("source")
        if source == "agent":
            continue
        message = step.get("message")
        if isinstance(message, str) and message.strip():
            return message
    return None


def _render_step_line(step: dict[str, Any], *, idx: int) -> str:
    """Single-line compressed render of one step.

    Format:
        ``<idx>. [<source>] <message_or_pending> {<tool_names>} -> <result>``

    Mirrors the judge's line shape so the embedding sees the same
    "conversation shape" cues the judge does, which keeps the two
    enrichment artefacts aligned conceptually.
    """
    source = str(step.get("source") or "?")
    message = step.get("message")
    if isinstance(message, str) and message.strip():
        msg_part = _truncate(message, STEP_MESSAGE_CAP)
    else:
        msg_part = "(no message)"
    parts = [f"{idx}. [{source}] {msg_part}"]

    tool_calls = step.get("tool_calls") or []
    if isinstance(tool_calls, list):
        names: list[str] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            name = call.get("function_name") or call.get("name")
            if isinstance(name, str) and name:
                names.append(name)
        if names:
            parts.append("{" + ", ".join(names) + "}")

    observation = step.get("observation") or {}
    result_text = _extract_observation_text(observation)
    if result_text:
        parts.append(f"-> {_truncate(result_text, STEP_RESULT_CAP)}")

    return " ".join(parts)


def _extract_observation_text(observation: Any) -> str | None:
    """Pull the first textual result content out of an ATIF observation.

    ATIF observations carry a ``results`` array; each result has
    ``content`` that is either a string (plain) or a list of blocks
    (``{"type": "text", "text": "..."}``). We unwrap whichever we
    find and return ``None`` for shapes we don't recognise.
    """
    if not isinstance(observation, dict):
        return None
    results = observation.get("results") or []
    if not isinstance(results, list) or not results:
        return None
    first = results[0]
    if not isinstance(first, dict):
        return None
    content = first.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text:
                    chunks.append(text)
        if chunks:
            return "\n".join(chunks)
    return None


def _collect_tool_names(steps: list[Any]) -> list[str]:
    """Unique-but-ordered list of all tool names invoked across the
    trace. Surfaced separately in the header so the embedding can
    cluster "all the bash-heavy traces" / "all the grep+edit
    traces" even when the conversation prose diverges.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        for call in step.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            name = call.get("function_name") or call.get("name")
            if isinstance(name, str) and name and name not in seen:
                seen.add(name)
                ordered.append(name)
    return ordered


def extract_searchable_text(
    atif: dict[str, Any],
    *,
    failure_label: str | None = None,
    failure_label_reasoning: str | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Build the embedding-input blob for one trace.

    Parameters
    ----------
    atif
        Parsed ATIF dict (the silver-layer JSON column, already
        ``json.loads``-ed). Missing / malformed sub-fields degrade
        gracefully: a trace with no steps still embeds with just
        agent + label header.
    failure_label
        Optional failure label string (e.g. ``"loop"``). Comes from
        the ``traces.failure_label`` column; ``None`` for traces
        the judge hasn't seen (10% sampling in slice 4 leaves most
        unlabelled). When present it goes into the header so it
        dominates the semantic neighbourhood.
    failure_label_reasoning
        Optional one-paragraph reasoning from the judge. Truncated
        to a fixed cap so a very chatty paragraph can't eat the
        whole budget.
    max_chars
        Hard cap on the returned blob. Default is
        :data:`DEFAULT_MAX_CHARS`. Callers (e.g. an ablation) can
        widen / narrow.

    Returns
    -------
    A plain-text blob suitable for passing into
    ``openai.embeddings.create(model="text-embedding-3-small",
    input=...)``. Empty trajectories return a minimal
    ``"Agent: unknown\\nModel: unknown\\n..."`` header.
    """
    if not isinstance(atif, dict):
        atif = {}

    agent = atif.get("agent") if isinstance(atif.get("agent"), dict) else {}
    agent_name = str(agent.get("name") or "unknown")
    model_name = str(agent.get("model_name") or "unknown")

    raw_steps = atif.get("steps")
    steps: list[Any] = raw_steps if isinstance(raw_steps, list) else []

    tool_names = _collect_tool_names(steps)

    header_lines: list[str] = [
        f"Agent: {agent_name}",
        f"Model: {model_name}",
        f"Step count: {len(steps)}",
    ]
    if tool_names:
        header_lines.append("Tools used: " + ", ".join(tool_names[:20]))
    if isinstance(failure_label, str) and failure_label.strip():
        header_lines.append(f"Failure label: {failure_label.strip()}")
    if isinstance(failure_label_reasoning, str) and failure_label_reasoning.strip():
        header_lines.append(
            "Label reasoning: " + _truncate(failure_label_reasoning, 600)
        )

    first_msg = _extract_first_user_message(steps)
    if first_msg:
        header_lines.append("Task: " + _truncate(first_msg, TASK_CHAR_CAP))

    header = "\n".join(header_lines)

    if not steps:
        return _truncate(header + "\n\nConversation: (no steps)", max_chars)

    head_omitted = max(0, len(steps) - TAIL_STEPS)
    tail = steps[-TAIL_STEPS:] if len(steps) > TAIL_STEPS else steps
    rendered: list[str] = []
    if head_omitted:
        rendered.append(f"(... {head_omitted} earlier steps omitted ...)")
    for offset, step in enumerate(tail):
        if not isinstance(step, dict):
            continue
        absolute_idx = head_omitted + offset + 1
        rendered.append(_render_step_line(step, idx=absolute_idx))

    body = "\n".join(rendered)
    full = header + "\n\nConversation:\n" + body
    if len(full) <= max_chars:
        return full
    # Header is always preserved -- drop body tail to fit.
    keep_for_body = max_chars - len(header) - len("\n\nConversation:\n") - 32
    if keep_for_body <= 0:
        return _truncate(header, max_chars)
    return (
        header
        + "\n\nConversation:\n"
        + body[:keep_for_body].rstrip()
        + "\n(... truncated ...)"
    )


__all__ = [
    "DEFAULT_MAX_CHARS",
    "STEP_MESSAGE_CAP",
    "STEP_RESULT_CAP",
    "TAIL_STEPS",
    "TASK_CHAR_CAP",
    "extract_searchable_text",
]
