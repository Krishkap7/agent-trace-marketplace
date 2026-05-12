"""Fixture-driven tests for :class:`CursorAdapter`.

The Cursor v1 export envelope is a project-defined JSON shape (see
``trace_marketplace.adapters.cursor`` docstring for the spec). These
tests exercise the adapter's mapping rules against the synthetic
``cursor_sample.cursor.json`` fixture plus a handful of inline payloads
for defensive-parsing edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from harbor.models.trajectories import Trajectory

from trace_marketplace.adapters.cursor import CursorAdapter


@pytest.fixture(scope="module")
def adapter() -> CursorAdapter:
    return CursorAdapter()


@pytest.fixture(scope="module")
def fixture_text(cursor_sample_path: Path) -> str:
    return cursor_sample_path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def converted(
    adapter: CursorAdapter, fixture_text: str, cursor_sample_path: Path
) -> Trajectory:
    traj = adapter.to_atif(fixture_text, cursor_sample_path)
    assert traj is not None, "fixture should parse cleanly"
    return traj


def _envelope(**overrides) -> str:
    base = {
        "session_id": "test-session",
        "cursor_version": "0.45.0",
        "model": "claude-3.5-sonnet",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
    }
    base.update(overrides)
    return json.dumps(base)


def test_session_id_from_envelope(converted: Trajectory) -> None:
    assert converted.session_id == "sess-fixture-cursor-1"


def test_agent_identity_is_cursor(converted: Trajectory) -> None:
    assert converted.agent.name == "cursor"
    assert converted.agent.version == "0.45.0"
    assert converted.agent.model_name == "claude-3.5-sonnet"


def test_schema_version_is_v17(converted: Trajectory) -> None:
    assert converted.schema_version == "ATIF-v1.7"


def test_trajectory_extra_has_workspace_and_synthetic_marker(
    converted: Trajectory,
) -> None:
    extra = converted.extra or {}
    assert extra.get("workspace_path") == "/Users/me/projects/fictional"
    assert extra.get("synthetic") is True


def test_step_count_excludes_pure_tool_result_carriers(converted: Trajectory) -> None:
    """6 messages -> 4 steps (2 user-tool_result-only messages collapse)."""
    assert len(converted.steps) == 4


def test_step_ids_are_sequential_starting_at_one(converted: Trajectory) -> None:
    ids = [step.step_id for step in converted.steps]
    assert ids == [1, 2, 3, 4]


def test_first_step_is_user_with_text(converted: Trajectory) -> None:
    step = converted.steps[0]
    assert step.source == "user"
    assert "Find a bug" in step.message
    assert not step.tool_calls


def test_assistant_text_block_array_is_flattened(converted: Trajectory) -> None:
    step = converted.steps[1]
    assert step.source == "agent"
    assert "I'll read foo.py" in step.message


def test_tool_calls_attach_to_emitting_agent_step(converted: Trajectory) -> None:
    step = converted.steps[1]
    assert len(step.tool_calls) == 1
    tc = step.tool_calls[0]
    assert tc.tool_call_id == "tc-1"
    assert tc.function_name == "read_file"
    assert tc.arguments == {"path": "foo.py"}


def test_tool_result_attaches_to_prior_agent_step_observation(
    converted: Trajectory,
) -> None:
    """The standalone ``role:user`` tool_results message must NOT create
    its own step; it should append to the agent step that emitted tc-1."""
    step = converted.steps[1]
    assert step.observation is not None
    results = step.observation.results
    assert len(results) == 1
    assert results[0].source_call_id == "tc-1"
    assert "a - b" in results[0].content


def test_tool_result_content_blocks_array_flattened(converted: Trajectory) -> None:
    """tc-2's result uses the Anthropic-blocks content shape."""
    step = converted.steps[2]
    assert step.tool_calls[0].tool_call_id == "tc-2"
    assert step.observation is not None
    results = step.observation.results
    assert len(results) == 1
    assert "edit applied" in results[0].content


def test_final_assistant_step_has_no_tool_calls(converted: Trajectory) -> None:
    step = converted.steps[3]
    assert step.source == "agent"
    assert "Fixed" in step.message
    assert not step.tool_calls


def test_agent_steps_carry_model_name(converted: Trajectory) -> None:
    for step in converted.steps:
        if step.source == "agent":
            assert step.model_name == "claude-3.5-sonnet"


def test_invalid_json_returns_none(adapter: CursorAdapter) -> None:
    assert adapter.to_atif("{not valid json") is None


def test_top_level_array_returns_none(adapter: CursorAdapter) -> None:
    assert adapter.to_atif("[]") is None


def test_missing_messages_returns_none(adapter: CursorAdapter) -> None:
    payload = json.dumps({"session_id": "x", "messages": []})
    assert adapter.to_atif(payload) is None


def test_missing_session_id_falls_back_to_filename(
    adapter: CursorAdapter, tmp_path: Path
) -> None:
    payload = json.dumps({"messages": [{"role": "user", "content": "hi"}]})
    file_path = tmp_path / "named-session.cursor.json"
    file_path.write_text(payload, encoding="utf-8")
    traj = adapter.to_atif(payload, file_path)
    assert traj is not None
    assert traj.session_id == "named-session.cursor"


def test_synthetic_failure_mode_propagates_to_trajectory_extra(
    adapter: CursorAdapter,
) -> None:
    """Generators set `synthetic_failure_mode` as a sibling of `synthetic`
    at the envelope top level. The adapter must carry it through to
    ``trajectory.extra`` so it survives as a ground-truth label in the DB."""
    payload = json.dumps(
        {
            "session_id": "fm-test",
            "synthetic": True,
            "synthetic_failure_mode": "broke_working_code",
            "messages": [
                {"role": "user", "content": "fix it"},
                {"role": "assistant", "content": "fixed"},
            ],
        }
    )
    traj = adapter.to_atif(payload)
    assert traj is not None
    extra = traj.extra or {}
    assert extra.get("synthetic") is True
    assert extra.get("synthetic_failure_mode") == "broke_working_code"


def test_unknown_role_counted_in_extra(adapter: CursorAdapter) -> None:
    payload = json.dumps(
        {
            "session_id": "unk",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "moderator", "content": "filtered"},
                {"role": "moderator", "content": "again"},
                {"role": "assistant", "content": "ok"},
            ],
        }
    )
    traj = adapter.to_atif(payload)
    assert traj is not None
    extra = traj.extra or {}
    assert extra.get("unknown_roles") == {"moderator": 2}
    assert len(traj.steps) == 2


def test_unmatched_tool_result_counted_in_extra(adapter: CursorAdapter) -> None:
    payload = json.dumps(
        {
            "session_id": "orphan",
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "ok"},
                {
                    "role": "user",
                    "content": "",
                    "tool_results": [{"tool_call_id": "tc-ghost", "content": "?"}],
                },
            ],
        }
    )
    traj = adapter.to_atif(payload)
    assert traj is not None
    extra = traj.extra or {}
    assert extra.get("unmatched_tool_results") == 1


def test_assistant_with_tool_calls_and_no_text_still_emits_step(
    adapter: CursorAdapter,
) -> None:
    payload = json.dumps(
        {
            "session_id": "silent-call",
            "messages": [
                {"role": "user", "content": "do it"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "tc-x", "name": "run", "arguments": {}}],
                },
            ],
        }
    )
    traj = adapter.to_atif(payload)
    assert traj is not None
    assert len(traj.steps) == 2
    assert traj.steps[1].source == "agent"
    assert traj.steps[1].tool_calls[0].function_name == "run"


def test_malformed_tool_call_skipped_not_fatal(adapter: CursorAdapter) -> None:
    payload = json.dumps(
        {
            "session_id": "bad-tc",
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "trying",
                    "tool_calls": [
                        {"name": "missing_id"},
                        {"id": "tc-good", "name": "run", "arguments": {"x": 1}},
                    ],
                },
            ],
        }
    )
    traj = adapter.to_atif(payload)
    assert traj is not None
    agent_step = traj.steps[1]
    assert len(agent_step.tool_calls) == 1
    assert agent_step.tool_calls[0].tool_call_id == "tc-good"


def test_non_dict_arguments_preserved_under_raw_key(adapter: CursorAdapter) -> None:
    payload = json.dumps(
        {
            "session_id": "weird-args",
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "calling",
                    "tool_calls": [
                        {
                            "id": "tc-1",
                            "name": "fn",
                            "arguments": "actually-a-string",
                        }
                    ],
                },
            ],
        }
    )
    traj = adapter.to_atif(payload)
    assert traj is not None
    args = traj.steps[1].tool_calls[0].arguments
    assert args == {"_raw": "actually-a-string"}


def test_missing_cursor_version_defaults_to_unknown(adapter: CursorAdapter) -> None:
    traj = adapter.to_atif(_envelope(cursor_version=None))
    assert traj is not None
    assert traj.agent.version == "unknown"


def test_string_content_field_handled(adapter: CursorAdapter) -> None:
    traj = adapter.to_atif(_envelope())
    assert traj is not None
    assert traj.steps[0].message == "hello"
    assert traj.steps[1].message == "hi"


# --------------------------------------------------------------------------- #
# Cross-format sweep: Codex / Claude Code real-data bugs ported to Cursor
# --------------------------------------------------------------------------- #
#
# The Codex and Claude Code adapters had three real-data bugs (PRs #4, #5):
#   1. encrypted reasoning/thinking blocks silently dropped
#   2. ``final_metrics`` never aggregated from per-event usage data
#   3. unknown event/role types not catalogued in trajectory.extra
#
# Cursor inherited the first two and is the worst offender on #1 -- it
# never extracted reasoning at all. #3 was already covered by the
# existing ``unknown_roles`` counter. These tests pin the fixes.


def test_thinking_block_with_plain_text_lands_in_reasoning_content(
    adapter: CursorAdapter,
) -> None:
    """An Anthropic-style thinking block carrying real text MUST land
    on the agent step's ``reasoning_content`` rather than be silently
    dropped. Most Cursor flows that emit thinking blocks come from
    ``cursor-agent --output-format stream-json`` piped into a v1
    envelope, so the wire shape is Anthropic's."""
    payload = json.dumps(
        {
            "session_id": "thinking-plain-text",
            "model": "claude-3.5-sonnet",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Let me plan first."},
                        {"type": "text", "text": "Okay, here goes."},
                    ],
                },
            ],
        }
    )
    traj = adapter.to_atif(payload)
    assert traj is not None
    agent_steps = [s for s in traj.steps if s.source == "agent"]
    assert len(agent_steps) == 1
    assert agent_steps[0].message == "Okay, here goes."
    rc = agent_steps[0].reasoning_content or ""
    assert "let me plan first" in rc.lower(), (
        "thinking text must land on reasoning_content (was previously dropped)"
    )


def test_thinking_block_encrypted_signature_surfaces_as_placeholder(
    adapter: CursorAdapter,
) -> None:
    """Anthropic's extended-thinking shape with the human-readable
    trace withheld: ``{"type":"thinking", "thinking":"", "signature":"<base64>"}``.
    Direct mirror of the Codex ``encrypted_content`` and Claude Code
    ``thinking.signature`` placeholder fixes -- the silver column has
    to record "reasoning happened here" or the trajectory looks like
    the model never reasoned."""
    payload = json.dumps(
        {
            "session_id": "thinking-encrypted",
            "model": "claude-3.5-sonnet",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "", "signature": "x" * 250},
                        {"type": "text", "text": "Done."},
                    ],
                },
            ],
        }
    )
    traj = adapter.to_atif(payload)
    assert traj is not None
    agent_steps = [s for s in traj.steps if s.source == "agent"]
    assert len(agent_steps) == 1
    rc = agent_steps[0].reasoning_content or ""
    assert "encrypted thinking" in rc.lower()
    assert "250" in rc


def test_thinking_only_assistant_message_still_emits_a_step(
    adapter: CursorAdapter,
) -> None:
    """An assistant message whose entire content is a thinking block
    (no visible text, no tool_calls) is unusual but valid. Previously
    the empty-text-and-no-tool-calls skip dropped it entirely -- losing
    the reasoning placeholder along with the step. Must now emit a
    step carrying just the reasoning."""
    payload = json.dumps(
        {
            "session_id": "thinking-only",
            "model": "claude-3.5-sonnet",
            "messages": [
                {"role": "user", "content": "ok"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "", "signature": "x" * 100},
                    ],
                },
            ],
        }
    )
    traj = adapter.to_atif(payload)
    assert traj is not None
    agent_steps = [s for s in traj.steps if s.source == "agent"]
    assert len(agent_steps) == 1
    assert agent_steps[0].reasoning_content
    assert "encrypted thinking" in agent_steps[0].reasoning_content.lower()


def test_reasoning_not_attached_to_user_or_system_steps(
    adapter: CursorAdapter,
) -> None:
    """ATIF user/system steps don't carry reasoning. If a user message
    somehow contains a thinking block (malformed export), we strip the
    reasoning rather than attach it to a user-source step."""
    payload = json.dumps(
        {
            "session_id": "thinking-user",
            "model": "claude-3.5-sonnet",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "thinking", "thinking": "user thinking? no."},
                        {"type": "text", "text": "actual user text"},
                    ],
                },
                {"role": "assistant", "content": "ack"},
            ],
        }
    )
    traj = adapter.to_atif(payload)
    assert traj is not None
    user_steps = [s for s in traj.steps if s.source == "user"]
    assert len(user_steps) == 1
    assert user_steps[0].message == "actual user text"
    assert user_steps[0].reasoning_content is None


def test_final_metrics_aggregated_when_messages_carry_usage(
    adapter: CursorAdapter,
) -> None:
    """If a Cursor envelope's messages carry Anthropic-style ``usage``
    blocks (the shape ``cursor-agent --output-format stream-json``
    emits), the adapter must roll them up into trajectory-level
    ``final_metrics``. Otherwise Cursor rows are the only format whose
    final_metrics is always None, breaking downstream comparison."""
    payload = json.dumps(
        {
            "session_id": "cursor-with-usage",
            "model": "claude-3.5-sonnet",
            "messages": [
                {"role": "user", "content": "do thing"},
                {
                    "role": "assistant",
                    "content": "doing thing",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 200,
                        "cache_read_input_tokens": 50,
                        "cache_creation_input_tokens": 25,
                    },
                },
                {"role": "user", "content": "more"},
                {
                    "role": "assistant",
                    "content": "more done",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 20,
                        "cache_read_input_tokens": 5,
                    },
                },
            ],
        }
    )
    traj = adapter.to_atif(payload)
    assert traj is not None
    fm = traj.final_metrics
    assert fm is not None
    assert fm.total_prompt_tokens == 110
    assert fm.total_completion_tokens == 220
    assert fm.total_cached_tokens == 55
    extra = fm.extra or {}
    assert extra.get("cache_creation_input_tokens") == 25


def test_final_metrics_none_when_no_message_carries_usage(
    adapter: CursorAdapter,
) -> None:
    """Don't fabricate an empty final_metrics block when the envelope
    has no usage data -- keeping the field None signals "no token data
    available" honestly. Mirrors the Claude Code regression."""
    traj = adapter.to_atif(_envelope())
    assert traj is not None
    assert traj.final_metrics is None
