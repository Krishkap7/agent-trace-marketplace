"""Anthropic-backed LLM judge for trace failure classification.

One public function: :func:`judge_trace`. It builds a compressed
prompt from a parsed ATIF dict + the structural signals already
computed by :mod:`failure_rules`, sends it to Sonnet 4.6 via the
SDK's tool-use feature (so the response is guaranteed-parseable
JSON), validates the returned label against :class:`FailureLabel`,
retries once on enum failure, and raises :class:`JudgeError` on
second failure or unrecoverable API error.

Why tool-use rather than JSON-mode prompting?

Tool definitions let us hand the model a strict schema with an
enum constraint on the label field. Free-form JSON output via
system-prompt instructions works most of the time but fails open
on edge cases (the model invents a near-but-not-quite label, or
wraps the JSON in markdown). Tool-use makes the failure mode
binary: either the model emits a valid call or we retry.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

from anthropic import Anthropic

from trace_marketplace.enrich.failure_taxonomy import (
    FailureLabel,
    labels_with_descriptions,
)

log = logging.getLogger(__name__)

JUDGE_MODEL: str = "claude-sonnet-4-6"
"""Pinned model slug. Sonnet 4.6 was the user's choice for this slice:
it handles the semantic distinctions between near-miss labels (e.g.
``wrong_approach`` vs ``misread_error``) more reliably than Haiku,
and the small corpus (~325 calls) makes the cost delta irrelevant."""

JUDGE_TEMPERATURE: float = 0.0
"""Reproducibility. Re-running the script should pick the same label
for the same trace, modulo provider-side nondeterminism."""

JUDGE_MAX_TOKENS: int = 400
"""Cap on the model's response. The tool call payload is short
(label + one-sentence reasoning); 400 tokens is ample headroom."""

INPUT_PRICE_PER_MTOK: float = 3.00
"""Sonnet 4.6 input token price in USD per million tokens. Updated
manually when Anthropic changes pricing. Source of truth for the
cost estimator in ``scripts/judge_failures.py``."""

OUTPUT_PRICE_PER_MTOK: float = 15.00
"""Sonnet 4.6 output token price in USD per million tokens."""

MAX_PROMPT_CHARS: int = 36_000
"""Upper bound on the compressed prompt body before the labels list
is appended. Roughly ~9 K tokens at 4 chars / token, leaving headroom
under the 10 K-token soft cap from the plan."""

STEP_RENDER_TAIL_COUNT: int = 30
"""How many *trailing* steps land in the rendered conversation.
Captures the trace's late-stage behaviour where failure modes
typically manifest (loops, give-ups, error spirals). Earlier steps
get a one-line summary so the model has crude context."""

STEP_MESSAGE_CHAR_CAP: int = 300
"""Per-step message truncation. 300 chars ~= 75 tokens; with 30
trailing steps that's ~2.2 K tokens for messages alone, well within
``MAX_PROMPT_CHARS``."""

STEP_RESULT_CHAR_CAP: int = 200
"""Per-step observation-result truncation. Tighter than messages
because tool outputs are noisy."""


@dataclass(frozen=True)
class JudgeResult:
    """Outcome of one :func:`judge_trace` call.

    ``usage`` mirrors what the Anthropic SDK returns on a successful
    response: ``input_tokens``, ``output_tokens``, plus the optional
    cache fields when prompt caching is in play. We pass it through
    so the calling script can sum tokens and compute total spend.
    """

    label: FailureLabel
    reasoning: str
    usage: dict[str, int]


class JudgeError(RuntimeError):
    """Raised when :func:`judge_trace` can't produce a valid label
    even after the single allowed retry. Callers in
    ``scripts/judge_failures.py`` log the trace id and continue
    (one bad row shouldn't tank a 325-call batch)."""


# ---------------------------------------------------------------------------
# Prompt construction.
# ---------------------------------------------------------------------------


_LABEL_ENUM_VALUES: list[str] = [m.value for m in FailureLabel]


def _truncate(text: str, cap: int) -> str:
    text = text.strip()
    if len(text) <= cap:
        return text
    return text[: cap - 1].rstrip() + "\u2026"  # add ellipsis


def _render_step(step: dict[str, Any]) -> str:
    """Single-line summary of one ATIF step.

    Format: ``[<source>] <message_or_pending> {<tool_call_names>} -> <observation_first_chars>``
    """
    source = str(step.get("source") or "?")
    message = step.get("message")
    if isinstance(message, str) and message.strip():
        msg_part = _truncate(message, STEP_MESSAGE_CHAR_CAP)
    else:
        msg_part = "(no message)"
    bits = [f"[{source}] {msg_part}"]
    tool_calls = step.get("tool_calls") or []
    if tool_calls:
        names = [
            str(c.get("function_name") or "?")
            for c in tool_calls
            if isinstance(c, dict)
        ]
        if names:
            bits.append("{" + ", ".join(names) + "}")
    observation = step.get("observation") or {}
    if isinstance(observation, dict):
        results = observation.get("results") or []
        if results and isinstance(results[0], dict):
            content = results[0].get("content")
            if isinstance(content, str):
                bits.append(f"-> {_truncate(content, STEP_RESULT_CHAR_CAP)}")
            elif isinstance(content, list):
                joined = "\n".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and isinstance(b.get("text"), str)
                )
                if joined:
                    bits.append(f"-> {_truncate(joined, STEP_RESULT_CHAR_CAP)}")
    return " ".join(bits)


def _render_conversation(traj: dict[str, Any]) -> str:
    """Compress the trajectory steps into a model-friendly summary.

    Strategy:
    * Always render the last :data:`STEP_RENDER_TAIL_COUNT` steps in
      full single-line form -- failures usually manifest at the tail.
    * For the preceding steps (if any), drop in a single "(N earlier
      steps omitted)" placeholder so the model knows context exists
      but doesn't pay tokens to read it.
    * Hard-cut the whole rendering at :data:`MAX_PROMPT_CHARS` to keep
      the prompt under the soft 10 K-token plan budget.
    """
    steps = traj.get("steps") or []
    if not isinstance(steps, list):
        return "(no steps)"
    if not steps:
        return "(no steps)"
    tail_count = STEP_RENDER_TAIL_COUNT
    if len(steps) > tail_count:
        head_omitted = len(steps) - tail_count
        rendered = [f"(... {head_omitted} earlier steps omitted ...)"]
        rendered.extend(
            _render_step(s) for s in steps[-tail_count:] if isinstance(s, dict)
        )
    else:
        rendered = [_render_step(s) for s in steps if isinstance(s, dict)]
    body = "\n".join(rendered)
    if len(body) > MAX_PROMPT_CHARS:
        body = body[: MAX_PROMPT_CHARS - 50].rstrip() + "\n(... truncated ...)"
    return body


def _build_user_prompt(traj: dict[str, Any], signals: dict[str, bool]) -> str:
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

    fired = sorted(name for name, hit in signals.items() if hit)
    fired_str = ", ".join(fired) if fired else "(none)"

    parts = [
        "You are classifying agent trace failures from a coding-agent marketplace.",
        "",
        "Trace summary:",
        f"- Agent: {agent_name}   Model: {model_name}",
        f"- Steps: {num_steps}    Tool calls: {num_tool_calls}",
        f"- Rule-based signals fired: {fired_str}",
        "",
        "Conversation (compressed, latest steps last):",
        _render_conversation(traj),
        "",
        "Pick exactly ONE label that best describes the trace.",
        "Use the `submit_label` tool to return your decision.",
        "",
        "Allowed labels:",
        labels_with_descriptions(),
    ]
    return "\n".join(parts)


_SUBMIT_TOOL: dict[str, Any] = {
    "name": "submit_label",
    "description": (
        "Submit the final failure-mode classification for this trace. "
        "`label` must be one of the values listed in the prompt; "
        "`reasoning` is one sentence (max ~200 chars) explaining the choice."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["label", "reasoning"],
        "properties": {
            "label": {
                "type": "string",
                "enum": _LABEL_ENUM_VALUES,
                "description": "Exactly one of the failure labels from the prompt.",
            },
            "reasoning": {
                "type": "string",
                "description": "One sentence justifying the label.",
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Anthropic client protocol (kept tight so tests can mock without the SDK).
# ---------------------------------------------------------------------------


class _Messages(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _AnthropicLike(Protocol):
    messages: _Messages


# ---------------------------------------------------------------------------
# Public API.
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


def _extract_tool_call(message_obj: Any) -> tuple[str, str] | None:
    """Pull ``(label, reasoning)`` out of the SDK response.

    Returns ``None`` if no ``submit_label`` tool_use block is present
    OR if the input payload is missing either field. Caller decides
    whether to retry.
    """
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
        if name != _SUBMIT_TOOL["name"]:
            continue
        payload = getattr(block, "input", None) or (
            block.get("input") if isinstance(block, dict) else None
        )
        if not isinstance(payload, dict):
            return None
        label = payload.get("label")
        reasoning = payload.get("reasoning")
        if not isinstance(label, str) or not isinstance(reasoning, str):
            return None
        return label, reasoning
    return None


def _call_once(
    client: _AnthropicLike, user_prompt: str
) -> tuple[str, str, dict[str, int]]:
    """Single SDK request; returns the (label, reasoning, usage) triple
    or raises :class:`JudgeError` on parse failure.

    Forced tool_choice ensures the model can't reply with free-form
    prose: it MUST emit a ``submit_label`` tool_use block or the SDK
    will retry internally / surface an error we wrap.
    """
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=JUDGE_MAX_TOKENS,
        temperature=JUDGE_TEMPERATURE,
        tools=[_SUBMIT_TOOL],
        tool_choice={"type": "tool", "name": _SUBMIT_TOOL["name"]},
        messages=[{"role": "user", "content": user_prompt}],
    )
    parsed = _extract_tool_call(response)
    if parsed is None:
        raise JudgeError(
            "Model response did not include a valid submit_label tool_use block"
        )
    label, reasoning = parsed
    return label, reasoning, _normalise_usage(getattr(response, "usage", None))


def _validate_label(raw: str) -> FailureLabel | None:
    try:
        return FailureLabel(raw)
    except ValueError:
        return None


def judge_trace(
    trajectory: dict[str, Any],
    signals: dict[str, bool],
    *,
    client: _AnthropicLike | None = None,
) -> JudgeResult:
    """Classify ``trajectory`` into a :class:`FailureLabel`.

    ``signals`` is passed verbatim into the prompt as a structural prior;
    when a rule already fired the model is biased toward the matching
    behavioural label (``loop``, ``gave_up``, etc.) but is still free to
    overrule it with a semantic label that fits better.

    A custom ``client`` can be injected for tests; production callers
    pass ``None`` and we instantiate a fresh ``Anthropic()`` whose
    ``ANTHROPIC_API_KEY`` lookup happens via the SDK's normal env
    handling.

    Retries:
    * 1st failure (no tool_use block / unknown label) -> retry once.
    * 2nd failure -> raise :class:`JudgeError`. Caller decides whether
      to skip or abort the batch.
    """
    if client is None:
        client = Anthropic()

    user_prompt = _build_user_prompt(trajectory, signals)

    for attempt in (1, 2):
        try:
            label_str, reasoning, usage = _call_once(client, user_prompt)
        except JudgeError:
            if attempt == 2:
                raise
            log.warning("Judge: no tool_use block on attempt %d; retrying", attempt)
            continue
        label = _validate_label(label_str)
        if label is None:
            if attempt == 2:
                raise JudgeError(
                    f"Model returned label {label_str!r} that is not in FailureLabel"
                )
            log.warning(
                "Judge: invalid label %r on attempt %d; retrying", label_str, attempt
            )
            continue
        return JudgeResult(label=label, reasoning=reasoning.strip(), usage=usage)

    # Unreachable: both branches above either return or raise.
    raise JudgeError("Judge: exhausted retries without returning a label")


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Return the dollar cost of one or many judge calls.

    Pricing source-of-truth lives at the module level so the dry-run
    estimator in ``scripts/judge_failures.py`` and the post-run
    summariser can share constants -- no chance of the two drifting."""
    return (
        input_tokens / 1_000_000 * INPUT_PRICE_PER_MTOK
        + output_tokens / 1_000_000 * OUTPUT_PRICE_PER_MTOK
    )


def require_api_key() -> None:
    """Refuse to start when ``ANTHROPIC_API_KEY`` is unset.

    Lives next to the judge (rather than in each calling script)
    because every script that touches the judge has the same
    requirement; if we ever add more callers (e.g. a re-judge CLI)
    they all share this guard.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Export it before running this script:\n"
            "    export ANTHROPIC_API_KEY=sk-ant-..."
        )


__all__ = [
    "INPUT_PRICE_PER_MTOK",
    "JUDGE_MODEL",
    "JUDGE_TEMPERATURE",
    "OUTPUT_PRICE_PER_MTOK",
    "JudgeError",
    "JudgeResult",
    "estimate_cost_usd",
    "judge_trace",
    "require_api_key",
]
