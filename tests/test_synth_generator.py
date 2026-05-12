"""Tests for :mod:`trace_marketplace.synth.llm_generator`.

The Anthropic client and validator are both mocked. We pin:

1. Happy path: one call -> ok result, cost recorded, attempts=1.
2. Adapter rejects first attempt -> retry brief includes the error;
   second attempt succeeds.
3. Both attempts fail -> ok=False with last error captured.
4. The system message + tool schema are passed through verbatim.
5. Tool_choice is forced to ``submit_trace``.
6. Cost arithmetic sums correctly across calls.
7. Rate-limit error triggers backoff retry (we monkeypatch sleep).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from trace_marketplace.synth.llm_generator import (
    CostEstimate,
    GENERATOR_MODEL,
    generate_trace,
)
from trace_marketplace.synth.prompts import OUTPUT_TOOL_NAME
from trace_marketplace.synth.spec import TraceSpec
from trace_marketplace.synth.validator import ValidationResult


# ---------------------------------------------------------------------------
# Mocks.
# ---------------------------------------------------------------------------


@dataclass
class _FakeBlock:
    type: str
    name: str
    input: dict[str, Any]


@dataclass
class _FakeUsage:
    input_tokens: int = 1000
    output_tokens: int = 500
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class _FakeMessage:
    content: list[_FakeBlock]
    usage: _FakeUsage = field(default_factory=_FakeUsage)


class _ScriptedClient:
    """Anthropic-like client that replays a queue of pre-baked responses."""

    def __init__(self, responses: list[_FakeMessage]):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> _FakeMessage:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("Test ran out of scripted responses")
        return self._responses.pop(0)


def _make_response(file_bytes: str, *, usage: _FakeUsage | None = None) -> _FakeMessage:
    return _FakeMessage(
        content=[
            _FakeBlock(
                type="tool_use",
                name=OUTPUT_TOOL_NAME,
                input={"file_bytes": file_bytes},
            )
        ],
        usage=usage or _FakeUsage(),
    )


def _make_no_tool_response() -> _FakeMessage:
    """Response with no tool_use block -- forces the 'no submit_trace block' path."""
    return _FakeMessage(content=[_FakeBlock(type="text", name="", input={})])


def _spec() -> TraceSpec:
    return TraceSpec(
        output_format="cursor",
        primary_failure="loop",
        task="Fix the test",
        target_step_count=10,
    )


def _ok_validation(text: str) -> ValidationResult:
    return ValidationResult(ok=True, error="", trajectory=None)


def _fail_validation(reason: str):
    def _inner(_fmt: str, _text: str) -> ValidationResult:
        return ValidationResult(ok=False, error=reason, trajectory=None)

    return _inner


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_happy_path_single_call() -> None:
    client = _ScriptedClient([_make_response("ok bytes")])
    result = generate_trace(
        _spec(),
        client=client,
        validate_fn=lambda fmt, text: _ok_validation(text),
    )
    assert result.ok
    assert result.attempts == 1
    assert result.file_bytes == "ok bytes"
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == GENERATOR_MODEL
    assert call["tool_choice"] == {"type": "tool", "name": OUTPUT_TOOL_NAME}


def test_retries_once_on_validation_failure() -> None:
    client = _ScriptedClient(
        [_make_response("bad bytes"), _make_response("good bytes")]
    )
    seen: list[str] = []

    def _validate_seq(_fmt: str, text: str) -> ValidationResult:
        seen.append(text)
        if text == "bad bytes":
            return ValidationResult(ok=False, error="bad cross-ref", trajectory=None)
        return _ok_validation(text)

    result = generate_trace(_spec(), client=client, validate_fn=_validate_seq)
    assert result.ok
    assert result.attempts == 2
    assert len(client.calls) == 2
    # Retry brief should reference the prior error.
    assert "bad cross-ref" in client.calls[1]["messages"][0]["content"]


def test_gives_up_after_two_failures() -> None:
    client = _ScriptedClient(
        [_make_response("bad 1"), _make_response("bad 2")]
    )
    result = generate_trace(
        _spec(),
        client=client,
        validate_fn=_fail_validation("permanent failure"),
    )
    assert not result.ok
    assert result.attempts == 2
    assert "permanent failure" in result.error


def test_missing_tool_use_block_counts_as_failure() -> None:
    """If Sonnet returns text-only, we treat it as a failure and retry."""
    client = _ScriptedClient([_make_no_tool_response(), _make_response("good bytes")])
    result = generate_trace(
        _spec(),
        client=client,
        validate_fn=lambda fmt, text: _ok_validation(text),
    )
    assert result.ok
    assert result.attempts == 2


def test_cost_is_summed_across_calls() -> None:
    high_usage = _FakeUsage(input_tokens=2000, output_tokens=1000)
    low_usage = _FakeUsage(input_tokens=500, output_tokens=250)
    client = _ScriptedClient(
        [
            _make_response("bad", usage=high_usage),
            _make_response("good", usage=low_usage),
        ]
    )
    result = generate_trace(
        _spec(),
        client=client,
        validate_fn=lambda fmt, text: (
            _ok_validation(text)
            if text == "good"
            else ValidationResult(ok=False, error="x", trajectory=None)
        ),
    )
    assert result.ok
    assert result.cost.input_tokens == 2500
    assert result.cost.output_tokens == 1250
    # USD math: (2500 * 3.00 + 1250 * 15.00) / 1M = 0.0263
    assert abs(result.cost.usd - 0.0263) < 0.0001


def test_cost_estimate_addition() -> None:
    a = CostEstimate(input_tokens=100, output_tokens=50)
    b = CostEstimate(input_tokens=200, cache_read_tokens=30)
    c = a + b
    assert c.input_tokens == 300
    assert c.output_tokens == 50
    assert c.cache_read_tokens == 30


def test_system_message_includes_format_spec() -> None:
    client = _ScriptedClient([_make_response("ok")])
    generate_trace(
        TraceSpec(
            output_format="claude_code",
            primary_failure="loop",
            task="fix it",
            target_step_count=10,
        ),
        client=client,
        validate_fn=lambda fmt, text: _ok_validation(text),
    )
    system = client.calls[0]["system"]
    # At least one block has cache_control set.
    cached = [b for b in system if b.get("cache_control")]
    assert cached
    # The static block contains the Claude Code spec.
    assert "Claude Code session JSONL" in cached[0]["text"]


def test_rate_limit_retry_with_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """RateLimitError should trigger a backoff retry, not propagate."""
    from anthropic import RateLimitError

    slept: list[float] = []
    monkeypatch.setattr(
        "trace_marketplace.synth.llm_generator.time.sleep",
        lambda d: slept.append(d),
    )

    class _Flaky:
        def __init__(self):
            self.messages = self
            self.calls = 0

        def create(self, **_kwargs: Any) -> _FakeMessage:
            self.calls += 1
            if self.calls == 1:
                raise RateLimitError(
                    "rate limited",
                    response=_DummyResponse(),  # type: ignore[arg-type]
                    body=None,
                )
            return _make_response("ok bytes")

    client = _Flaky()
    result = generate_trace(
        _spec(),
        client=client,
        validate_fn=lambda fmt, text: _ok_validation(text),
        max_429_retries=4,
    )
    assert result.ok
    assert client.calls == 2
    assert len(slept) == 1


# ---------------------------------------------------------------------------
# Helper for the rate-limit test.
# ---------------------------------------------------------------------------


class _DummyResponse:
    """Tiny stand-in for ``httpx.Response`` used in RateLimitError construction."""

    status_code = 429
    headers: dict[str, str] = {}

    def __init__(self) -> None:
        self.request = self

    def json(self) -> dict[str, Any]:
        return {"error": {"type": "rate_limit_error", "message": "test"}}
