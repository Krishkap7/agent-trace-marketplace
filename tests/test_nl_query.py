"""Tests for :mod:`trace_marketplace.search.nl_query`.

The Anthropic client is fully mocked. We pin:

1. The prompt embeds the failure-label taxonomy and the source-format
   allow-list (these are the contract with the model).
2. A clean parse round-trips: enum values land in the right fields,
   non-categorical text lands in ``semantic_intent`` / ``explanation``.
3. The parser retries once on a bad payload and succeeds on the second
   attempt.
4. The parser raises :class:`NLQueryError` after two consecutive
   bad payloads.
5. An unknown ``source_format`` enum value triggers the retry path.
6. Bare-whitespace queries raise ``ValueError`` without burning an API
   call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from trace_marketplace.search.nl_query import (
    NLQueryError,
    _build_user_prompt,
    parse_nl_query,
)


@dataclass
class _FakeBlock:
    type: str
    name: str
    input: dict[str, Any]


@dataclass
class _FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 30


@dataclass
class _FakeMessage:
    content: list[_FakeBlock]
    usage: _FakeUsage = field(default_factory=_FakeUsage)


def _payload(**overrides: Any) -> dict[str, Any]:
    base = {
        "semantic_intent": "find repetitive tool-call loops",
        "explanation": "Filtering by failure_label=loop with no other filters.",
        "filters": {"failure_label": ["loop"]},
    }
    base.update(overrides)
    return base


class _ScriptedClient:
    """Replays a queue of pre-baked responses (one per ``create`` call)."""

    def __init__(self, responses: list[_FakeMessage]):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("Test ran out of scripted responses")
        return self._responses.pop(0)


def _make_response(payload: dict[str, Any]) -> _FakeMessage:
    return _FakeMessage(
        content=[
            _FakeBlock(
                type="tool_use",
                name="submit_parsed_query",
                input=payload,
            )
        ]
    )


def test_build_user_prompt_includes_taxonomy_and_allowed_formats():
    prompt = _build_user_prompt("find loops in claude_code")
    # Failure-label taxonomy lines.
    assert "- loop:" in prompt
    assert "- hallucinated_api:" in prompt
    # Source-format allow-list.
    assert "claude_code" in prompt
    assert "swe_agent" in prompt
    # The query itself goes in verbatim.
    assert "find loops in claude_code" in prompt


def test_parse_nl_query_happy_path_pure_filter():
    client = _ScriptedClient(
        [
            _make_response(
                _payload(
                    semantic_intent="repetitive tool call loops",
                    explanation="Filter by failure_label=loop.",
                    filters={
                        "failure_label": ["loop"],
                        "source_format": ["claude_code"],
                    },
                )
            )
        ]
    )
    parsed = parse_nl_query("find loops in claude_code", client=client)
    assert parsed.failure_labels == ("loop",)
    assert parsed.source_formats == ("claude_code",)
    assert parsed.agent_names == ()
    assert parsed.has_error is None
    assert parsed.semantic_intent == "repetitive tool call loops"
    assert "loop" in parsed.explanation.lower()
    assert parsed.usage["input_tokens"] == 100


def test_parse_nl_query_bare_semantic_intent_no_filters():
    client = _ScriptedClient(
        [
            _make_response(
                _payload(
                    semantic_intent=(
                        "agents that invented a function or library that does not exist"
                    ),
                    explanation="Pure semantic query, no filters applied.",
                    filters={},
                )
            )
        ]
    )
    parsed = parse_nl_query("agents hallucinating tools", client=client)
    assert parsed.failure_labels == ()
    assert parsed.source_formats == ()
    assert parsed.has_error is None
    assert "invented" in parsed.semantic_intent


def test_parse_nl_query_retries_on_invalid_enum():
    bad = _make_response(
        _payload(filters={"source_format": ["claude-code"]})  # underscore expected
    )
    good = _make_response(_payload(filters={"source_format": ["claude_code"]}))
    client = _ScriptedClient([bad, good])
    parsed = parse_nl_query("find anything in claude_code", client=client)
    assert parsed.source_formats == ("claude_code",)
    assert len(client.calls) == 2


def test_parse_nl_query_raises_after_two_bad_responses():
    bad1 = _make_response(_payload(filters={"failure_label": ["not_a_label"]}))
    bad2 = _make_response(_payload(semantic_intent="   "))  # blank semantic_intent
    client = _ScriptedClient([bad1, bad2])
    with pytest.raises(NLQueryError):
        parse_nl_query("anything", client=client)
    assert len(client.calls) == 2


def test_parse_nl_query_raises_on_missing_tool_use_block():
    # Response has zero tool_use blocks -- should retry then raise.
    @dataclass
    class _PlainTextBlock:
        type: str = "text"
        text: str = "Sorry, I cannot help with that."

    bad = _FakeMessage(content=[_PlainTextBlock()])  # type: ignore[list-item]
    client = _ScriptedClient([bad, bad])
    with pytest.raises(NLQueryError):
        parse_nl_query("anything", client=client)


def test_parse_nl_query_rejects_blank_query():
    client = _ScriptedClient([])  # would assert if called
    with pytest.raises(ValueError):
        parse_nl_query("   ", client=client)
    assert client.calls == []


def test_parse_nl_query_handles_has_error_boolean():
    client = _ScriptedClient(
        [
            _make_response(
                _payload(
                    semantic_intent="successful traces",
                    explanation="Filter by has_error=false.",
                    filters={"has_error": False},
                )
            )
        ]
    )
    parsed = parse_nl_query("only successes", client=client)
    assert parsed.has_error is False
