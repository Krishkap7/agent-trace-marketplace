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
