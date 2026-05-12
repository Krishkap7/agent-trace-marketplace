"""Natural-language query parser for the trace marketplace.

One public entry point :func:`parse_nl_query`. It takes the user's
free-form English ("find loops in claude code that hit max turns")
and returns a structured :class:`ParsedQuery` carrying:

* ``filters`` -- a :class:`~trace_marketplace.ui.queries.ListFilters`
  compatible dict (enum-constrained on the LLM side, validated again
  here, then handed to the existing SQL builder).
* ``semantic_intent`` -- a short paraphrase of the *semantic* part of
  the query, used as the embedding input for the vector NN ranking.
* ``explanation`` -- a one-line summary the UI shows the user so the
  parse is debuggable at a glance.

Implementation follows the same pattern as
:mod:`trace_marketplace.enrich.failure_judge`:

1. Build a tight prompt that enumerates the available filter values
   and the failure-label vocabulary in full.
2. Hand the model a strict tool schema (Anthropic ``tool_choice``
   forcing) so the response is guaranteed-parseable JSON with enum
   constraints on the categorical filter fields.
3. Validate the parsed payload; retry once on schema failure; raise
   :class:`NLQueryError` on second failure.

Why tool-use rather than JSON-mode prompting? Same reason as the
judge: free-form JSON occasionally hallucinates near-but-not-quite
enum values (``claude-code`` vs ``claude_code``); tool-use turns
that class of bug into a hard error we can retry.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from anthropic import Anthropic

from trace_marketplace.enrich.failure_taxonomy import (
    FailureLabel,
    labels_with_descriptions,
)

log = logging.getLogger(__name__)

PARSER_MODEL: str = "claude-sonnet-4-6"
"""Same model the judge uses. The cost delta vs Haiku is rounding
error at slice-6 traffic (one call per NL query) and Sonnet handles
the enum constraints + semantic-intent rewrite reliably."""

PARSER_TEMPERATURE: float = 0.0
"""Reproducibility -- the same query should parse the same way."""

PARSER_MAX_TOKENS: int = 600
"""Cap on the model's response. Tool-call payload is tiny; this is
deliberately generous so a chatty ``explanation`` field can fit."""


SOURCE_FORMAT_VALUES: tuple[str, ...] = (
    "atif",
    "claude_code",
    "codex",
    "cursor",
    "swe_agent",
)
"""Allowed values for the ``source_format`` filter. Pinned here
rather than read from the live DB so the parser's enum constraint
is stable across deploys and tests."""

KNOWN_AGENT_NAMES: tuple[str, ...] = (
    "claude-code",
    "codex",
    "cursor",
    "mcp",
    "swe-agent",
)
"""Agent names seen in the current corpus. Surfaced in the prompt as
a hint -- the parser is allowed to emit unknown agent names (it's a
free-text column), but it's biased toward the known set."""


@dataclass(frozen=True)
class ParsedQuery:
    """Output of :func:`parse_nl_query`.

    All fields are normalised so the caller can drop them straight
    into :class:`~trace_marketplace.ui.queries.ListFilters` and the
    embedding API.
    """

    semantic_intent: str
    """Short paraphrase of the semantic component. Embedded by the
    NL-search pipeline; can be empty if the query is pure filter."""

    explanation: str
    """Human-readable summary of how the model interpreted the
    query. Surfaced in the UI as a "Search intent: ..." caption."""

    failure_labels: tuple[str, ...] = ()
    agent_names: tuple[str, ...] = ()
    source_formats: tuple[str, ...] = ()
    has_error: bool | None = None
    """Tri-state. ``True`` -> errors only, ``False`` -> successes
    only, ``None`` -> no constraint."""

    usage: dict[str, int] = field(default_factory=dict)
    """Token-usage echo, same shape as the judge's. Populated on the
    happy path so a future cost dashboard can sum without re-asking
    the API."""


class NLQueryError(RuntimeError):
    """Raised when :func:`parse_nl_query` cannot return a valid
    :class:`ParsedQuery` even after the single allowed retry."""


# ---------------------------------------------------------------------------
# Prompt + tool schema.
# ---------------------------------------------------------------------------


_LABEL_ENUM_VALUES: list[str] = [m.value for m in FailureLabel]


def _build_user_prompt(query: str) -> str:
    parts = [
        "You are the NL-query parser for a coding-agent trace marketplace.",
        "Users type free-form English questions; you must split the query "
        "into (a) structured filters that can be applied with a SQL WHERE "
        "clause, and (b) a `semantic_intent` -- a 1-2 sentence paraphrase "
        "of the *behavioural* part of the query that will be embedded for "
        "vector nearest-neighbour ranking.",
        "",
        "Return your answer by calling the `submit_parsed_query` tool.",
        "",
        "Filter rules:",
        "- Only include a filter field when the user clearly asks for it. "
        "Do NOT guess. If the query is purely semantic, leave `filters` "
        "empty and put everything in `semantic_intent`.",
        "- `failure_label` MUST come from the taxonomy below. Translate "
        "phrases ('agent gave up', 'kept looping', 'hallucinated a "
        "function') into the matching enum value.",
        "- `source_format` is the ingest format, NOT the agent name. "
        "Allowed: atif, claude_code, codex, cursor, swe_agent.",
        "- `agent_name` is the runtime agent identifier. Known values: "
        + ", ".join(KNOWN_AGENT_NAMES)
        + ". Unknown agent names are fine but bias toward this list.",
        "- `has_error` is a boolean tri-state: true for failures only, "
        "false for successes only. Omit unless the query is explicit.",
        "",
        "Semantic-intent rules:",
        "- Rewrite the behavioural slice of the query as a short "
        "self-contained sentence the embedding model can match. "
        "Strip filter words.",
        "- If the query is pure filter ('show me all gave_up traces'), "
        "set `semantic_intent` to the same phrase -- the embedding "
        "ranking will fall back to lexical similarity over the trace "
        "headers.",
        "- Never leave `semantic_intent` empty; it must be a non-empty string.",
        "",
        "Explanation rules:",
        "- One sentence, max ~30 words. Tell the user how you split "
        "their query so the parse is debuggable.",
        "",
        "Failure-label taxonomy:",
        labels_with_descriptions(),
        "",
        f"User query: {query}",
    ]
    return "\n".join(parts)


_PARSE_TOOL: dict[str, Any] = {
    "name": "submit_parsed_query",
    "description": (
        "Return the structured filters + semantic intent parsed from the "
        "user's natural-language query. Categorical fields must use the "
        "enum values listed in the prompt; unknown values will cause a "
        "retry."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["semantic_intent", "explanation"],
        "properties": {
            "filters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "failure_label": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": _LABEL_ENUM_VALUES,
                        },
                    },
                    "agent_name": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "source_format": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": list(SOURCE_FORMAT_VALUES),
                        },
                    },
                    "has_error": {"type": "boolean"},
                },
            },
            "semantic_intent": {
                "type": "string",
                "description": (
                    "1-2 sentence paraphrase of the behavioural slice of "
                    "the query. Used as the embedding input. Never empty."
                ),
            },
            "explanation": {
                "type": "string",
                "description": (
                    "One short sentence (<=30 words) telling the user "
                    "how you split their query."
                ),
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Anthropic client protocol -- same shape as failure_judge.
# ---------------------------------------------------------------------------


class _Messages(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _AnthropicLike(Protocol):
    messages: _Messages


# ---------------------------------------------------------------------------
# Response extraction.
# ---------------------------------------------------------------------------


def _normalise_usage(usage: Any) -> dict[str, int]:
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
    """Pull the ``submit_parsed_query`` tool input out of the SDK response."""
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
        if name != _PARSE_TOOL["name"]:
            continue
        payload = getattr(block, "input", None) or (
            block.get("input") if isinstance(block, dict) else None
        )
        if not isinstance(payload, dict):
            return None
        return payload
    return None


def _validate_payload(payload: dict[str, Any]) -> ParsedQuery | None:
    """Validate + normalise. Returns ``None`` on schema mismatch.

    Belt-and-braces validation: the tool schema already enforces
    enum / type, but a defensive pass means a misbehaving SDK
    version can't sneak garbage past us.
    """
    semantic_intent = payload.get("semantic_intent")
    explanation = payload.get("explanation")
    if not isinstance(semantic_intent, str) or not semantic_intent.strip():
        return None
    if not isinstance(explanation, str) or not explanation.strip():
        return None

    filters = payload.get("filters") or {}
    if not isinstance(filters, dict):
        return None

    def _str_tuple(value: Any) -> tuple[str, ...] | None:
        if value is None:
            return ()
        if not isinstance(value, list):
            return None
        out: list[str] = []
        for item in value:
            if not isinstance(item, str):
                return None
            out.append(item)
        return tuple(out)

    failure_labels = _str_tuple(filters.get("failure_label"))
    if failure_labels is None:
        return None
    for label in failure_labels:
        if label not in _LABEL_ENUM_VALUES:
            return None

    agent_names = _str_tuple(filters.get("agent_name"))
    if agent_names is None:
        return None

    source_formats = _str_tuple(filters.get("source_format"))
    if source_formats is None:
        return None
    for fmt in source_formats:
        if fmt not in SOURCE_FORMAT_VALUES:
            return None

    has_error = filters.get("has_error")
    if has_error is not None and not isinstance(has_error, bool):
        return None

    return ParsedQuery(
        semantic_intent=semantic_intent.strip(),
        explanation=explanation.strip(),
        failure_labels=failure_labels,
        agent_names=agent_names,
        source_formats=source_formats,
        has_error=has_error,
    )


def _call_once(
    client: _AnthropicLike, user_prompt: str
) -> tuple[ParsedQuery | None, dict[str, int]]:
    response = client.messages.create(
        model=PARSER_MODEL,
        max_tokens=PARSER_MAX_TOKENS,
        temperature=PARSER_TEMPERATURE,
        tools=[_PARSE_TOOL],
        tool_choice={"type": "tool", "name": _PARSE_TOOL["name"]},
        messages=[{"role": "user", "content": user_prompt}],
    )
    usage = _normalise_usage(getattr(response, "usage", None))
    payload = _extract_tool_payload(response)
    if payload is None:
        return None, usage
    return _validate_payload(payload), usage


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def parse_nl_query(
    query: str,
    *,
    client: _AnthropicLike | None = None,
) -> ParsedQuery:
    """Parse a free-form English query into structured filters + intent.

    ``query`` is the raw user input (UI passes it verbatim). The
    string is trimmed and validated to non-empty; bare-whitespace
    queries raise ``ValueError`` so the UI doesn't burn an API
    call on an obvious mistake.

    Retries:
    * 1st failure (no tool_use block / invalid enum / missing
      required field) -> retry once.
    * 2nd failure -> raise :class:`NLQueryError`.

    A custom ``client`` can be injected for tests; production
    callers pass ``None`` and we instantiate a fresh ``Anthropic()``
    whose ``ANTHROPIC_API_KEY`` lookup goes through the SDK.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("NL query must be a non-empty string")
    if client is None:
        client = Anthropic()

    user_prompt = _build_user_prompt(query.strip())
    last_usage: dict[str, int] = {}
    for attempt in (1, 2):
        parsed, usage = _call_once(client, user_prompt)
        last_usage = usage
        if parsed is not None:
            return ParsedQuery(
                semantic_intent=parsed.semantic_intent,
                explanation=parsed.explanation,
                failure_labels=parsed.failure_labels,
                agent_names=parsed.agent_names,
                source_formats=parsed.source_formats,
                has_error=parsed.has_error,
                usage=usage,
            )
        if attempt == 2:
            raise NLQueryError(
                "NL parser could not produce a valid parsed query after retry "
                f"(last usage: {last_usage})"
            )
        log.warning("NL parser: invalid payload on attempt %d; retrying", attempt)

    raise NLQueryError("NL parser exhausted retries without returning a query")


def require_api_key() -> None:
    """Refuse to start when ``ANTHROPIC_API_KEY`` is unset.

    Mirrors :func:`failure_judge.require_api_key` so the UI's
    graceful-degradation path can show a uniform message no matter
    which feature requires the key.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Export it before invoking the "
            "NL query parser:\n    export ANTHROPIC_API_KEY=sk-ant-..."
        )


__all__ = [
    "KNOWN_AGENT_NAMES",
    "NLQueryError",
    "PARSER_MODEL",
    "PARSER_TEMPERATURE",
    "ParsedQuery",
    "SOURCE_FORMAT_VALUES",
    "parse_nl_query",
    "require_api_key",
]
