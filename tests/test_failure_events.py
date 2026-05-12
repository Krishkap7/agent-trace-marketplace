"""Unit tests for :mod:`trace_marketplace.enrich.failure_events`.

Pins:

* :func:`summarize_label` severity rules (empty -> SUCCESS,
  all-recovered -> SUCCESS, mixed -> most-severe unrecovered).
* :func:`derive_ultimately_succeeded` rules including the
  existing-value-wins case for SWE-agent backfills.
* :func:`extract_events` tool-use parsing, schema validation, and the
  single retry on schema failure.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from trace_marketplace.enrich.failure_events import (
    EVENTS_MODEL,
    EventsExtractionError,
    EventsResult,
    FailureEvent,
    SEVERITY_ORDER,
    _render_conversation_with_indices,
    derive_ultimately_succeeded,
    estimate_cost_usd,
    extract_events,
    summarize_label,
)
from trace_marketplace.enrich.failure_judge import STEP_RENDER_TAIL_COUNT
from trace_marketplace.enrich.failure_taxonomy import FailureLabel


# ---------- helpers ----------


def _ev(
    label: FailureLabel,
    *,
    step_start: int = 1,
    step_end: int = 2,
    recovered: bool = False,
    recovery_step: int | None = None,
    recovery_summary: str | None = None,
) -> FailureEvent:
    return FailureEvent(
        label=label,
        step_start=step_start,
        step_end=step_end,
        recovered=recovered,
        recovery_step=recovery_step,
        recovery_summary=recovery_summary,
    )


def _tool_use_block(*, input_payload: dict[str, Any] | None = None) -> Any:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "record_failure_events"
    block.input = input_payload or {}
    return block


def _text_block(text: str) -> Any:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _response(*content_blocks: Any, in_tok: int = 100, out_tok: int = 50) -> Any:
    response = MagicMock()
    response.content = list(content_blocks)
    response.usage = MagicMock(input_tokens=in_tok, output_tokens=out_tok)
    return response


def _make_client(responses: list[Any]) -> Any:
    client = MagicMock()
    iterator = iter(responses)

    def _create(**_kwargs: Any) -> Any:
        try:
            return next(iterator)
        except StopIteration as exc:
            raise AssertionError("Exhausted mocked responses") from exc

    client.messages.create.side_effect = _create
    return client


_TRIVIAL_TRAJ: dict[str, Any] = {
    "agent": {"name": "test-agent", "model_name": "test-model"},
    "steps": [
        {"source": "user", "message": "Fix the bug."},
        {"source": "agent", "message": "Sure."},
    ],
}


# ---------- summarize_label ----------


def test_summarize_label_empty_returns_success() -> None:
    assert summarize_label([]) is FailureLabel.SUCCESS


def test_summarize_label_all_recovered_returns_success() -> None:
    events = [
        _ev(
            FailureLabel.LOOP,
            recovered=True,
            recovery_step=5,
            recovery_summary="...",
        ),
        _ev(
            FailureLabel.HALLUCINATED_API,
            recovered=True,
            recovery_step=10,
            recovery_summary="...",
        ),
    ]
    assert summarize_label(events) is FailureLabel.SUCCESS


def test_summarize_label_picks_most_severe_unrecovered() -> None:
    # Mix: gave_up (behavioural) and hallucinated_api (cognitive).
    # Cognitive wins per the locked severity ordering.
    events = [
        _ev(FailureLabel.GAVE_UP),
        _ev(FailureLabel.HALLUCINATED_API),
    ]
    assert summarize_label(events) is FailureLabel.HALLUCINATED_API


def test_summarize_label_ignores_recovered_when_an_unrecovered_event_exists() -> None:
    events = [
        # Cognitive but recovered -> doesn't count.
        _ev(
            FailureLabel.HALLUCINATED_API,
            recovered=True,
            recovery_step=4,
            recovery_summary="fixed it",
        ),
        # Behavioural and unrecovered -> wins.
        _ev(FailureLabel.LOOP),
    ]
    assert summarize_label(events) is FailureLabel.LOOP


def test_summarize_label_full_severity_chain() -> None:
    """Walking the entire severity ordering: each label beats every label
    after it. Catches ordering regressions when the ranking changes."""
    for i, higher in enumerate(SEVERITY_ORDER):
        for lower in SEVERITY_ORDER[i + 1 :]:
            assert summarize_label([_ev(higher), _ev(lower)]) is higher, (
                f"Expected {higher} to outrank {lower}"
            )


# ---------- derive_ultimately_succeeded ----------


def test_derive_succeeded_preserves_existing_swe_agent_value() -> None:
    # Existing value MUST win regardless of what events say.
    events = [_ev(FailureLabel.LOOP)]  # unrecovered
    assert derive_ultimately_succeeded(events, existing=1) == 1
    assert derive_ultimately_succeeded(events, existing=0) == 0
    assert derive_ultimately_succeeded([], existing=0) == 0


def test_derive_succeeded_empty_events_returns_one() -> None:
    assert derive_ultimately_succeeded([], existing=None) == 1


def test_derive_succeeded_any_unrecovered_returns_zero() -> None:
    events = [
        _ev(
            FailureLabel.LOOP,
            recovered=True,
            recovery_step=5,
            recovery_summary="...",
        ),
        _ev(FailureLabel.GAVE_UP),  # unrecovered
    ]
    assert derive_ultimately_succeeded(events, existing=None) == 0


def test_derive_succeeded_all_recovered_returns_one() -> None:
    events = [
        _ev(
            FailureLabel.LOOP,
            recovered=True,
            recovery_step=5,
            recovery_summary="A",
        ),
        _ev(
            FailureLabel.HALLUCINATED_API,
            recovered=True,
            recovery_step=10,
            recovery_summary="B",
        ),
    ]
    assert derive_ultimately_succeeded(events, existing=None) == 1


# ---------- extract_events: happy path ----------


def test_extract_events_returns_validated_events_with_usage() -> None:
    payload = {
        "events": [
            {
                "label": "loop",
                "step_start": 3,
                "step_end": 6,
                "recovered": True,
                "recovery_step": 7,
                "recovery_summary": "Re-read the AttributeError and used Read first.",
            },
            {
                "label": "hallucinated_api",
                "step_start": 10,
                "step_end": 12,
                "recovered": False,
                "recovery_step": None,
                "recovery_summary": None,
            },
        ]
    }
    client = _make_client(
        [_response(_tool_use_block(input_payload=payload), in_tok=2000, out_tok=300)]
    )
    result = extract_events(_TRIVIAL_TRAJ, client=client)
    assert isinstance(result, EventsResult)
    assert len(result.events) == 2
    assert result.events[0].label is FailureLabel.LOOP
    assert result.events[0].recovered is True
    assert result.events[0].recovery_step == 7
    assert result.events[1].label is FailureLabel.HALLUCINATED_API
    assert result.events[1].recovered is False
    assert result.events[1].recovery_step is None
    assert result.events[1].recovery_summary is None
    assert result.input_tokens == 2000
    assert result.output_tokens == 300


def test_extract_events_empty_list_is_legal() -> None:
    client = _make_client([_response(_tool_use_block(input_payload={"events": []}))])
    result = extract_events(_TRIVIAL_TRAJ, client=client)
    assert result.events == []


def test_extract_events_uses_opus_model_and_forces_tool_choice() -> None:
    client = _make_client([_response(_tool_use_block(input_payload={"events": []}))])
    extract_events(_TRIVIAL_TRAJ, client=client)
    _, kwargs = client.messages.create.call_args
    assert kwargs["model"] == EVENTS_MODEL
    assert kwargs["tool_choice"] == {
        "type": "tool",
        "name": "record_failure_events",
    }
    assert isinstance(kwargs["tools"], list) and len(kwargs["tools"]) == 1
    assert kwargs["tools"][0]["name"] == "record_failure_events"


def test_extract_events_omits_temperature_for_opus_reasoning_model() -> None:
    """Regression guard: Opus 4.7 rejects ``temperature`` with HTTP 400.

    The first live run on slice 10 failed all 544 traces because we
    were passing ``temperature=0.0``; this test pins the fix so a
    careless rewrite that re-introduces the param trips a unit test
    before burning ~$0 again on rejected calls.
    """
    client = _make_client([_response(_tool_use_block(input_payload={"events": []}))])
    extract_events(_TRIVIAL_TRAJ, client=client)
    _, kwargs = client.messages.create.call_args
    assert "temperature" not in kwargs


# ---------- extract_events: retry behaviour ----------


def test_extract_events_retries_once_on_missing_tool_use_block() -> None:
    client = _make_client(
        [
            _response(_text_block("Here is my analysis...")),
            _response(_tool_use_block(input_payload={"events": []})),
        ]
    )
    result = extract_events(_TRIVIAL_TRAJ, client=client)
    assert result.events == []
    assert client.messages.create.call_count == 2


def test_extract_events_retries_once_on_invalid_event_then_succeeds() -> None:
    # First attempt: invalid label. Second: valid empty list.
    bad_payload = {
        "events": [
            {
                "label": "not_a_real_label",
                "step_start": 1,
                "step_end": 2,
                "recovered": False,
                "recovery_step": None,
                "recovery_summary": None,
            }
        ]
    }
    client = _make_client(
        [
            _response(_tool_use_block(input_payload=bad_payload)),
            _response(_tool_use_block(input_payload={"events": []})),
        ]
    )
    result = extract_events(_TRIVIAL_TRAJ, client=client)
    assert result.events == []
    assert client.messages.create.call_count == 2


def test_extract_events_raises_after_two_failures() -> None:
    bad_payload = {
        "events": [
            {
                "label": "still-not-a-label",
                "step_start": 1,
                "step_end": 2,
                "recovered": False,
                "recovery_step": None,
                "recovery_summary": None,
            }
        ]
    }
    client = _make_client(
        [
            _response(_tool_use_block(input_payload=bad_payload)),
            _response(_tool_use_block(input_payload=bad_payload)),
        ]
    )
    with pytest.raises(EventsExtractionError):
        extract_events(_TRIVIAL_TRAJ, client=client)
    assert client.messages.create.call_count == 2


def test_extract_events_rejects_recovered_event_without_summary() -> None:
    """recovered=true must come with a non-empty summary -- the model
    must not paper over a wall with ``recovery_summary=null``."""
    bad_payload = {
        "events": [
            {
                "label": "loop",
                "step_start": 1,
                "step_end": 2,
                "recovered": True,
                "recovery_step": 3,
                "recovery_summary": None,  # invalid for recovered=true
            }
        ]
    }
    valid_payload = {"events": []}
    client = _make_client(
        [
            _response(_tool_use_block(input_payload=bad_payload)),
            _response(_tool_use_block(input_payload=valid_payload)),
        ]
    )
    result = extract_events(_TRIVIAL_TRAJ, client=client)
    assert result.events == []
    assert client.messages.create.call_count == 2


def test_extract_events_rejects_inverted_step_range() -> None:
    bad_payload = {
        "events": [
            {
                "label": "loop",
                "step_start": 5,
                "step_end": 2,  # end < start
                "recovered": False,
                "recovery_step": None,
                "recovery_summary": None,
            }
        ]
    }
    valid_payload = {"events": []}
    client = _make_client(
        [
            _response(_tool_use_block(input_payload=bad_payload)),
            _response(_tool_use_block(input_payload=valid_payload)),
        ]
    )
    result = extract_events(_TRIVIAL_TRAJ, client=client)
    assert result.events == []


# ---------- pricing helper ----------


def test_estimate_cost_uses_opus_pricing() -> None:
    # 1M input tokens at $15 = $15.00
    # 1M output tokens at $75 = $75.00
    # Total = $90.00
    assert estimate_cost_usd(1_000_000, 1_000_000) == pytest.approx(90.00)
    # Zero tokens -> zero cost.
    assert estimate_cost_usd(0, 0) == 0.0


# ---------- FailureEvent serialisation ----------


def test_failure_event_to_dict_round_trip() -> None:
    event = _ev(
        FailureLabel.HALLUCINATED_API,
        step_start=5,
        step_end=8,
        recovered=True,
        recovery_step=9,
        recovery_summary="Used Read to verify the API surface.",
    )
    d = event.to_dict()
    assert d == {
        "label": "hallucinated_api",
        "step_start": 5,
        "step_end": 8,
        "recovered": True,
        "recovery_step": 9,
        "recovery_summary": "Used Read to verify the API surface.",
    }


# ---------- _validate_event boundary semantics ----------
#
# Regression guards from Cursor Bugbot on PR #13:
#
# 1. recovery_step must be at or after step_end. A recovery_step in the
#    middle of the failure window (step_start <= rs < step_end) is
#    semantically inconsistent -- the failure hasn't ended yet -- and
#    must be rejected so the caller retries.
# 2. For recovered=false events, both ``recovery_step`` and
#    ``recovery_summary`` must end up None on the validated record,
#    regardless of what the model emitted, so downstream code (UI
#    rendering, JSON storage, derive_ultimately_succeeded) sees one
#    canonical shape.
#
# We exercise both via the top-level ``extract_events`` path so the
# validation chain (Anthropic tool-use parsing -> ``_validate_event``
# -> retry) is hit end-to-end, not just the helper in isolation.


def _events_payload(*events: dict[str, Any]) -> dict[str, Any]:
    return {"events": list(events)}


def test_validate_event_accepts_recovery_step_at_step_end_boundary() -> None:
    """``recovery_step == step_end`` is the legitimate edge case where
    the agent recovered on the last failing step (Opus 4.7 emits this
    ~5% of the time in our live corpus). Must pass validation."""
    payload = _events_payload(
        {
            "label": "misread_error",
            "step_start": 7,
            "step_end": 11,
            "recovered": True,
            "recovery_step": 11,  # boundary: == step_end
            "recovery_summary": "Re-read the traceback's chained cause.",
        }
    )
    client = _make_client([_response(_tool_use_block(input_payload=payload))])
    result = extract_events(_TRIVIAL_TRAJ, client=client)
    assert len(result.events) == 1
    assert result.events[0].recovery_step == 11


def test_validate_event_rejects_recovery_step_inside_failure_window() -> None:
    """``step_start <= recovery_step < step_end`` is contradictory --
    recovery can't happen while the failure is still ongoing. Caller
    must retry once on the schema failure, then raise."""
    bad_payload = _events_payload(
        {
            "label": "loop",
            "step_start": 3,
            "step_end": 10,
            "recovered": True,
            "recovery_step": 5,  # squarely inside the failure window
            "recovery_summary": "Should never validate.",
        }
    )
    # Retry returns the same broken payload, so extract_events should
    # raise after the single retry budget is exhausted.
    client = _make_client(
        [
            _response(_tool_use_block(input_payload=bad_payload)),
            _response(_tool_use_block(input_payload=bad_payload)),
        ]
    )
    with pytest.raises(EventsExtractionError):
        extract_events(_TRIVIAL_TRAJ, client=client)


def test_validate_event_rejects_recovery_step_before_step_start() -> None:
    """``recovery_step < step_start`` is also nonsensical; same rejection
    path as the inside-window case."""
    bad_payload = _events_payload(
        {
            "label": "loop",
            "step_start": 8,
            "step_end": 12,
            "recovered": True,
            "recovery_step": 4,  # before the failure even started
            "recovery_summary": "Time-travel recovery.",
        }
    )
    client = _make_client(
        [
            _response(_tool_use_block(input_payload=bad_payload)),
            _response(_tool_use_block(input_payload=bad_payload)),
        ]
    )
    with pytest.raises(EventsExtractionError):
        extract_events(_TRIVIAL_TRAJ, client=client)


def test_validate_event_nulls_recovery_summary_when_not_recovered() -> None:
    """When ``recovered=false`` the validator must coerce
    ``recovery_summary`` to None even if the model emitted a non-empty
    string -- otherwise the UI would render a "not recovered" event
    with a recovery summary, which is contradictory. Symmetric with
    how ``recovery_step`` is already nulled in this branch."""
    payload = _events_payload(
        {
            "label": "gave_up",
            "step_start": 5,
            "step_end": 9,
            "recovered": False,
            "recovery_step": 12,  # stray non-null int, must be coerced
            # Stray non-empty string -- previously this slipped through
            # unchanged, leaving the event with recovered=false plus an
            # actual summary. Must now be coerced to None.
            "recovery_summary": "model leaked text here despite recovered=false",
        }
    )
    client = _make_client([_response(_tool_use_block(input_payload=payload))])
    result = extract_events(_TRIVIAL_TRAJ, client=client)
    assert len(result.events) == 1
    event = result.events[0]
    assert event.recovered is False
    assert event.recovery_step is None
    assert event.recovery_summary is None


def test_validate_event_nulls_empty_string_recovery_summary_when_not_recovered() -> (
    None
):
    """The known Opus 4.7 quirk: recovered=false + recovery_summary=""
    (empty string). Previously coerced to None as a special case; the
    new unconditional null still produces the same result."""
    payload = _events_payload(
        {
            "label": "incomplete",
            "step_start": 4,
            "step_end": 6,
            "recovered": False,
            "recovery_step": None,
            "recovery_summary": "",
        }
    )
    client = _make_client([_response(_tool_use_block(input_payload=payload))])
    result = extract_events(_TRIVIAL_TRAJ, client=client)
    assert len(result.events) == 1
    assert result.events[0].recovery_summary is None


# ---------- _render_conversation_with_indices ----------
#
# Regression guard from Cursor Bugbot on PR #13 a754746: the prompt
# claimed "Step indices are 1-based and match the UI's step badges" but
# the upstream ``failure_judge._render_conversation`` we were calling
# strips all indices. The model was guessing step numbers from line
# position, which works for short traces but breaks across the head-
# omitted placeholder. The new helper prefixes every visible line with
# its 1-based index.


def _step(source: str, message: str) -> dict[str, Any]:
    return {"source": source, "message": message}


def test_render_indices_short_trace_prefixes_each_step() -> None:
    traj = {"steps": [_step("user", "do X"), _step("agent", "ok"), _step("user", "Y")]}
    out = _render_conversation_with_indices(traj)
    lines = out.splitlines()
    assert lines[0].startswith("[step 1]")
    assert lines[1].startswith("[step 2]")
    assert lines[2].startswith("[step 3]")


def test_render_indices_long_trace_emits_offset_header_and_aligned_prefixes() -> None:
    """When more than STEP_RENDER_TAIL_COUNT steps exist, the head-omitted
    placeholder must name the first visible step's index and every
    rendered step must carry the matching 1-based prefix -- this is
    the case the previous renderer was silently broken on."""
    total = STEP_RENDER_TAIL_COUNT + 50  # head_omitted = 50
    traj = {
        "steps": [
            _step("user" if i % 2 == 0 else "agent", f"msg {i}") for i in range(total)
        ]
    }
    out = _render_conversation_with_indices(traj)
    lines = out.splitlines()
    head_omitted = total - STEP_RENDER_TAIL_COUNT
    first_visible = head_omitted + 1
    assert lines[0] == (
        f"(... {head_omitted} earlier steps omitted; "
        f"visible steps start at step {first_visible} ...)"
    )
    assert lines[1].startswith(f"[step {first_visible}]")
    # Last rendered line should be [step <total>] since tail covers
    # the last STEP_RENDER_TAIL_COUNT steps inclusive of the final one.
    assert lines[-1].startswith(f"[step {total}]")


def test_render_indices_empty_steps_returns_placeholder() -> None:
    assert _render_conversation_with_indices({"steps": []}) == "(no steps)"
    assert _render_conversation_with_indices({}) == "(no steps)"


def test_render_indices_skips_non_dict_step_entries() -> None:
    """Defensive: ATIF should always emit dict steps but a malformed
    blob shouldn't crash the renderer. Non-dict entries are skipped
    silently -- the resulting indices are still correct because we
    iterate with ``enumerate`` over the original list and only skip on
    render, not on counting."""
    traj = {"steps": [_step("user", "first"), "garbage", _step("agent", "third")]}
    out = _render_conversation_with_indices(traj)
    lines = out.splitlines()
    # Two rendered lines; index 1 for "first" and index 3 for "third"
    # (slot 2 was the non-dict "garbage" entry).
    assert len(lines) == 2
    assert lines[0].startswith("[step 1]")
    assert lines[1].startswith("[step 3]")


def test_build_prompt_includes_indexed_lines_when_trace_is_long() -> None:
    """End-to-end check: the actual prompt sent to Opus contains the
    indexed step prefixes, not the old un-indexed lines."""
    total = STEP_RENDER_TAIL_COUNT + 10
    traj = {"steps": [_step("user", f"line {i}") for i in range(total)]}
    client = _make_client([_response(_tool_use_block(input_payload={"events": []}))])
    extract_events(traj, client=client)
    _, kwargs = client.messages.create.call_args
    user_msg = kwargs["messages"][0]["content"]
    body = user_msg if isinstance(user_msg, str) else user_msg[0].get("text", "")
    assert "[step 11]" in body  # first visible step after 10 omitted
    assert f"[step {total}]" in body
    assert "visible steps start at step 11" in body
