"""Unit tests for the LLM judge wrapper.

Mocks the Anthropic SDK so the tests run offline and deterministically.
Pins the public contract:

* tool-use response parsing
* enum validation
* one-retry behaviour on parse failure
* enum-mismatch retry behaviour
* raise on second failure
* prompt structure (labels + signals + a tail of conversation steps)
* cost helper math
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from trace_marketplace.enrich.failure_judge import (
    INPUT_PRICE_PER_MTOK,
    JUDGE_MODEL,
    OUTPUT_PRICE_PER_MTOK,
    JudgeError,
    JudgeResult,
    estimate_cost_usd,
    judge_trace,
    require_api_key,
)
from trace_marketplace.enrich.failure_judge import (
    _build_user_prompt,  # noqa: PLC2701 (private but stable enough to pin)
)
from trace_marketplace.enrich.failure_taxonomy import FailureLabel


# ---------- helpers ----------


def _tool_use_block(
    *, name: str = "submit_label", input_payload: dict[str, Any] | None = None
) -> Any:
    """Build a stand-in for an Anthropic ``ContentBlock`` of type
    ``tool_use``. ``MagicMock`` is convenient because the production
    code accesses fields via ``getattr``."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = input_payload or {}
    return block


def _text_block(text: str) -> Any:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _response(
    *content_blocks: Any, input_tokens: int = 100, output_tokens: int = 30
) -> Any:
    response = MagicMock()
    response.content = list(content_blocks)
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return response


def _make_client(responses: list[Any]) -> Any:
    """Return a fake Anthropic-shaped client whose ``messages.create``
    pops successive responses from ``responses``."""
    client = MagicMock()
    iter_responses = iter(responses)

    def _create(**_kwargs: Any) -> Any:
        try:
            return next(iter_responses)
        except StopIteration as exc:
            raise AssertionError("Test exhausted the mocked response queue") from exc

    client.messages.create.side_effect = _create
    return client


_TRIVIAL_TRAJ: dict[str, Any] = {
    "agent": {"name": "test-agent", "model_name": "test-model"},
    "steps": [
        {"source": "user", "message": "Fix the bug."},
        {"source": "agent", "message": "I can't access the repo."},
    ],
}
_TRIVIAL_SIGNALS = {"abandonment": True, "hit_max_turns": False}


# ---------- happy path ----------


def test_judge_trace_returns_label_reasoning_and_usage() -> None:
    client = _make_client(
        [
            _response(
                _tool_use_block(
                    input_payload={
                        "label": "gave_up",
                        "reasoning": "Agent explicitly refused the task.",
                    }
                ),
                input_tokens=500,
                output_tokens=40,
            )
        ]
    )
    result = judge_trace(_TRIVIAL_TRAJ, _TRIVIAL_SIGNALS, client=client)
    assert isinstance(result, JudgeResult)
    assert result.label is FailureLabel.GAVE_UP
    assert result.reasoning == "Agent explicitly refused the task."
    assert result.usage == {"input_tokens": 500, "output_tokens": 40}


def test_judge_trace_uses_correct_model_and_forces_tool_choice() -> None:
    client = _make_client(
        [
            _response(
                _tool_use_block(input_payload={"label": "success", "reasoning": "fine"})
            )
        ]
    )
    judge_trace(_TRIVIAL_TRAJ, _TRIVIAL_SIGNALS, client=client)

    _, kwargs = client.messages.create.call_args
    assert kwargs["model"] == JUDGE_MODEL
    assert kwargs["temperature"] == 0.0
    # tool_choice MUST pin the submit_label tool so the model can't
    # reply with free-form prose.
    assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_label"}
    assert isinstance(kwargs["tools"], list) and len(kwargs["tools"]) == 1
    assert kwargs["tools"][0]["name"] == "submit_label"


# ---------- retry behaviour ----------


def test_judge_trace_retries_once_on_missing_tool_use_block() -> None:
    # First response: only a text block (no tool_use). Second: valid tool_use.
    client = _make_client(
        [
            _response(_text_block("I'll classify this as a loop.")),
            _response(
                _tool_use_block(
                    input_payload={"label": "loop", "reasoning": "Same call x6."}
                )
            ),
        ]
    )
    result = judge_trace(_TRIVIAL_TRAJ, _TRIVIAL_SIGNALS, client=client)
    assert result.label is FailureLabel.LOOP
    assert client.messages.create.call_count == 2


def test_judge_trace_retries_once_on_invalid_enum_then_succeeds() -> None:
    client = _make_client(
        [
            _response(
                _tool_use_block(
                    input_payload={"label": "nope_not_a_label", "reasoning": "x"}
                )
            ),
            _response(
                _tool_use_block(
                    input_payload={
                        "label": "unclear",
                        "reasoning": "Insufficient context.",
                    }
                )
            ),
        ]
    )
    result = judge_trace(_TRIVIAL_TRAJ, _TRIVIAL_SIGNALS, client=client)
    assert result.label is FailureLabel.UNCLEAR
    assert client.messages.create.call_count == 2


def test_judge_trace_raises_after_two_invalid_label_attempts() -> None:
    client = _make_client(
        [
            _response(
                _tool_use_block(input_payload={"label": "bad-1", "reasoning": "x"})
            ),
            _response(
                _tool_use_block(input_payload={"label": "bad-2", "reasoning": "y"})
            ),
        ]
    )
    with pytest.raises(JudgeError):
        judge_trace(_TRIVIAL_TRAJ, _TRIVIAL_SIGNALS, client=client)
    assert client.messages.create.call_count == 2


def test_judge_trace_raises_after_two_missing_tool_use_blocks() -> None:
    client = _make_client(
        [
            _response(_text_block("text only")),
            _response(_text_block("text only again")),
        ]
    )
    with pytest.raises(JudgeError):
        judge_trace(_TRIVIAL_TRAJ, _TRIVIAL_SIGNALS, client=client)
    assert client.messages.create.call_count == 2


# ---------- prompt construction ----------


def test_build_user_prompt_lists_every_failure_label() -> None:
    prompt = _build_user_prompt(_TRIVIAL_TRAJ, _TRIVIAL_SIGNALS)
    for member in FailureLabel:
        assert member.value in prompt, (
            f"prompt missing label {member.value!r} -- the model wouldn't be "
            "able to pick it"
        )


def test_build_user_prompt_includes_fired_signal_names() -> None:
    prompt = _build_user_prompt(_TRIVIAL_TRAJ, _TRIVIAL_SIGNALS)
    # abandonment fired, hit_max_turns did not -- only the fired one
    # should appear in the "signals fired" line.
    assert "abandonment" in prompt
    assert "Rule-based signals fired:" in prompt


def test_build_user_prompt_summarises_long_trajectories_with_omitted_marker() -> None:
    long_traj: dict[str, Any] = {
        "agent": {"name": "agent", "model_name": "m"},
        "steps": [{"source": "agent", "message": f"step {i}"} for i in range(120)],
    }
    prompt = _build_user_prompt(long_traj, {})
    assert "earlier steps omitted" in prompt
    # The very first step must NOT appear verbatim (truncated to tail).
    assert "step 0 " not in prompt
    # The final step MUST appear.
    assert "step 119" in prompt


def test_build_user_prompt_asks_for_rich_shop_window_paragraph() -> None:
    """v3 of the schema asks for a rich 4-6 sentence paragraph that
    names concrete evidence, contrasts with a runner-up label, and
    flags notability for marketplace browsers. Pin the language so
    we notice if a future edit rolls the wording back."""
    prompt = _build_user_prompt(_TRIVIAL_TRAJ, _TRIVIAL_SIGNALS)
    assert "Reasoning style" in prompt
    assert "4-6 sentences" in prompt
    # The marketplace framing is what motivates the verbose reasoning.
    assert "marketplace" in prompt.lower()
    # Must instruct the model to compare against an alternative label
    # (the "runner-up" contrast) and to flag notability.
    assert "runner-up" in prompt or "alternative" in prompt
    assert "notable" in prompt.lower() or "notability" in prompt.lower()
    # Concrete evidence is the whole point; keep it pinned.
    assert "concrete evidence" in prompt
    assert "Do NOT" in prompt


# ---------- cost helpers ----------


def test_estimate_cost_usd_uses_module_constants() -> None:
    # 1M input tokens -> exactly INPUT_PRICE_PER_MTOK dollars.
    assert estimate_cost_usd(1_000_000, 0) == pytest.approx(INPUT_PRICE_PER_MTOK)
    assert estimate_cost_usd(0, 1_000_000) == pytest.approx(OUTPUT_PRICE_PER_MTOK)
    # Linear in both axes.
    assert estimate_cost_usd(500_000, 500_000) == pytest.approx(
        INPUT_PRICE_PER_MTOK / 2 + OUTPUT_PRICE_PER_MTOK / 2
    )


def test_estimate_cost_usd_zero_when_no_tokens() -> None:
    assert estimate_cost_usd(0, 0) == 0.0


# ---------- api key guard ----------


def test_require_api_key_exits_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        require_api_key()


def test_require_api_key_passes_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    require_api_key()  # must not raise
