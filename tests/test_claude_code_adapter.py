"""Fixture-driven tests for :class:`ClaudeCodeAdapter`.

The fixture ``tests/fixtures/claude_code_minimal.jsonl`` is entirely
synthetic (no real session content). It exercises every event type the
adapter handles and asserts the expected ATIF shape comes out.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harbor.models.trajectories import Trajectory
from trace_marketplace.adapters.claude_code import ClaudeCodeAdapter


@pytest.fixture(scope="module")
def trajectory(claude_code_minimal_path: Path) -> Trajectory:
    raw = claude_code_minimal_path.read_text()
    traj = ClaudeCodeAdapter().to_atif(raw, source_path=claude_code_minimal_path)
    assert traj is not None, "fixture must validate"
    return traj


# --------------------------------------------------------------------------- #
# Top-level trajectory shape
# --------------------------------------------------------------------------- #


def test_schema_version_is_v1_7(trajectory: Trajectory) -> None:
    assert trajectory.schema_version == "ATIF-v1.7"


def test_session_id_pulled_from_first_event(trajectory: Trajectory) -> None:
    assert trajectory.session_id == "test-session-0001"


def test_agent_identity(trajectory: Trajectory) -> None:
    assert trajectory.agent.name == "claude-code"
    assert trajectory.agent.version == "9.9.9"
    assert trajectory.agent.model_name == "claude-test-1"


# --------------------------------------------------------------------------- #
# Steps: counts, ordering, sources
# --------------------------------------------------------------------------- #


def test_expected_step_count(trajectory: Trajectory) -> None:
    # 1 user (string) + 2 assistant + 1 system (the compact_boundary one
    # with content=null must NOT become a step).
    assert len(trajectory.steps) == 4


def test_step_ids_are_sequential_from_one(trajectory: Trajectory) -> None:
    assert [s.step_id for s in trajectory.steps] == [1, 2, 3, 4]


def test_only_canonical_sources_emitted(trajectory: Trajectory) -> None:
    sources = [s.source for s in trajectory.steps]
    assert sources == ["user", "agent", "agent", "system"]


def test_first_user_message_text(trajectory: Trajectory) -> None:
    first = trajectory.steps[0]
    assert first.source == "user"
    assert first.message == "Hello, can you help me list files?"


def test_compact_boundary_system_is_dropped(trajectory: Trajectory) -> None:
    """A system event with content=null must not surface as a Step."""
    system_steps = [s for s in trajectory.steps if s.source == "system"]
    assert len(system_steps) == 1
    assert system_steps[0].message == "Disk space low on /tmp"


# --------------------------------------------------------------------------- #
# Assistant content: text, thinking, tool_use, metrics
# --------------------------------------------------------------------------- #


def test_first_assistant_step_has_reasoning(trajectory: Trajectory) -> None:
    step = trajectory.steps[1]
    assert step.source == "agent"
    assert step.message == "Sure, I'll list the files."
    assert step.reasoning_content == "I should list the directory."


def test_first_assistant_step_has_one_tool_call(trajectory: Trajectory) -> None:
    step = trajectory.steps[1]
    assert step.tool_calls is not None
    assert len(step.tool_calls) == 1
    tc = step.tool_calls[0]
    assert tc.tool_call_id == "tool_call_001"
    assert tc.function_name == "Bash"
    assert tc.arguments == {"command": "ls"}


def test_assistant_metrics_mapped(trajectory: Trajectory) -> None:
    step = trajectory.steps[1]
    assert step.metrics is not None
    assert step.metrics.prompt_tokens == 100
    assert step.metrics.completion_tokens == 20
    assert step.metrics.cached_tokens == 10
    # Anthropic-specific fields should land in metrics.extra
    assert step.metrics.extra is not None
    assert step.metrics.extra.get("cache_creation_input_tokens") == 5
    assert step.metrics.extra.get("service_tier") == "default"


def test_second_assistant_step_has_two_tool_calls(trajectory: Trajectory) -> None:
    step = trajectory.steps[2]
    assert step.tool_calls is not None
    assert [tc.tool_call_id for tc in step.tool_calls] == [
        "tool_call_002",
        "tool_call_003",
    ]


# --------------------------------------------------------------------------- #
# Tool results -> Observations on the matching agent step
# --------------------------------------------------------------------------- #


def test_tool_result_attached_to_first_agent_step(trajectory: Trajectory) -> None:
    step = trajectory.steps[1]
    assert step.observation is not None
    assert len(step.observation.results) == 1
    r = step.observation.results[0]
    assert r.source_call_id == "tool_call_001"
    assert r.content == "file1.txt\nfile2.txt\n"


def test_tool_result_array_content_concatenated(trajectory: Trajectory) -> None:
    step = trajectory.steps[2]
    assert step.observation is not None
    results = {r.source_call_id: r for r in step.observation.results}
    assert results["tool_call_002"].content == "contents of file1"


def test_tool_result_is_error_flag_preserved(trajectory: Trajectory) -> None:
    step = trajectory.steps[2]
    results = {r.source_call_id: r for r in step.observation.results}
    err_result = results["tool_call_003"]
    assert err_result.content == "permission denied"
    assert err_result.extra is not None
    assert err_result.extra.get("is_error") is True
    assert err_result.extra.get("exitCode") == 1


# --------------------------------------------------------------------------- #
# Noise events: counted, never emitted as Steps
# --------------------------------------------------------------------------- #


def test_no_noise_events_leak_as_steps(trajectory: Trajectory) -> None:
    """Every step's source MUST be one of the three ATIF-canonical values."""
    for step in trajectory.steps:
        assert step.source in {"user", "agent", "system"}


def test_trajectory_extra_records_dropped_event_counts(trajectory: Trajectory) -> None:
    extra = trajectory.extra or {}
    assert extra.get("permission_mode_changes") == 1
    assert extra.get("file_history_count") == 1
    assert extra.get("last_prompt_count") == 1
    assert extra.get("queue_operation_count") == 1


def test_trajectory_extra_records_attachment_counts(trajectory: Trajectory) -> None:
    extra = trajectory.extra or {}
    attachment_counts = extra.get("attachment_counts") or {}
    assert attachment_counts.get("task_reminder") == 1
    assert attachment_counts.get("skill_listing") == 1


def test_trajectory_extra_carries_first_event_metadata(trajectory: Trajectory) -> None:
    extra = trajectory.extra or {}
    # cwd / gitBranch / userType / entrypoint sit on the first user event in
    # the fixture; permission-mode (event 0) doesn't carry them, so the
    # extractor must walk further.
    assert extra.get("cwd") == "/tmp/proj"
    assert extra.get("gitBranch") == "main"
    assert extra.get("userType") == "external"
    assert extra.get("entrypoint") == "cli"


# --------------------------------------------------------------------------- #
# Error handling: adapter MUST NOT raise on malformed input
# --------------------------------------------------------------------------- #


def test_empty_input_returns_none() -> None:
    assert ClaudeCodeAdapter().to_atif("") is None


def test_malformed_json_returns_none() -> None:
    """Garbage input should produce None, not an exception."""
    assert ClaudeCodeAdapter().to_atif("not json at all\n{also: bad}") is None


# --------------------------------------------------------------------------- #
# Real-data regressions discovered ingesting an actual session export
# (data/raw/claude_code_real/*.jsonl). Each test pins one failure mode the
# fixture-driven tests above didn't catch.
# --------------------------------------------------------------------------- #


import json as _json  # noqa: E402  (kept local so the import block above stays declarative)


def _make_session(events: list[dict]) -> str:
    return "\n".join(_json.dumps(e) for e in events)


def test_ai_title_event_is_known_noise_not_unknown() -> None:
    """``ai-title`` is Claude Code's async LLM-generated session title,
    not a step. Previously logged as an unknown event type. Must now be
    counted as drop-able metadata and surfaced through
    ``trajectory.extra.ai_title_count`` (+ the title itself if present).
    """
    sid = "sess-ai-title"
    payload = _make_session(
        [
            {"type": "permission-mode", "sessionId": sid, "permissionMode": "default"},
            {
                "type": "user",
                "sessionId": sid,
                "cwd": "/tmp",
                "message": {"role": "user", "content": "hi"},
            },
            {"type": "ai-title", "sessionId": sid, "aiTitle": "Build a Flask app"},
            {"type": "ai-title", "sessionId": sid, "aiTitle": "Build a Flask app"},
        ]
    )
    traj = ClaudeCodeAdapter().to_atif(payload)
    assert traj is not None
    extra = traj.extra or {}
    assert extra.get("ai_title_count") == 2
    assert extra.get("ai_title") == "Build a Flask app"
    assert "ai-title" not in (extra.get("unknown_event_counts") or {})


def test_thinking_block_encrypted_signature_surfaces_as_placeholder() -> None:
    """Extended-thinking mode emits ``{"type":"thinking", "thinking":"",
    "signature":"<~250B base64>"}`` -- the human-readable trace is
    withheld by Anthropic. Previously this whole shape was filtered
    away (empty ``thinking`` field) so extended-thinking sessions
    looked like the model never reasoned. Must now surface a
    placeholder so the silver column reflects what happened.

    Mirror of the Codex ``encrypted_content`` fix discovered the same
    session running through a real rollout file."""
    sid = "sess-encrypted-thinking"
    payload = _make_session(
        [
            {"type": "permission-mode", "sessionId": sid, "permissionMode": "default"},
            {
                "type": "assistant",
                "sessionId": sid,
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "content": [
                        {"type": "thinking", "thinking": "", "signature": "x" * 250},
                        {"type": "text", "text": "Hi!"},
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            },
        ]
    )
    traj = ClaudeCodeAdapter().to_atif(payload)
    assert traj is not None
    agent_steps = [s for s in traj.steps if s.source == "agent"]
    assert len(agent_steps) == 1
    rc = agent_steps[0].reasoning_content or ""
    assert "encrypted thinking" in rc.lower(), (
        f"empty thinking+signature should produce a placeholder, got: {rc!r}"
    )
    assert "250" in rc, "signature byte count should be surfaced"


def test_thinking_block_prefers_text_over_signature_placeholder() -> None:
    """When a thinking block has BOTH human-readable text AND a
    signature (theoretical but allowed by the schema), the text wins."""
    sid = "sess-both"
    payload = _make_session(
        [
            {"type": "permission-mode", "sessionId": sid, "permissionMode": "default"},
            {
                "type": "assistant",
                "sessionId": sid,
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "real plan",
                            "signature": "x" * 100,
                        },
                        {"type": "text", "text": "ack"},
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            },
        ]
    )
    traj = ClaudeCodeAdapter().to_atif(payload)
    assert traj is not None
    rc = (traj.steps[-1].reasoning_content or "").lower()
    assert "real plan" in rc
    assert "encrypted thinking" not in rc


def test_final_metrics_aggregated_across_assistant_usage_blocks() -> None:
    """Claude Code attaches a fresh ``usage`` to every assistant event
    rather than emitting standalone token_count events. Without an
    aggregation pass, ``Trajectory.final_metrics`` was always None on
    real sessions -- breaking parity with Codex/SWE-agent rows. Must
    now sum input/output/cache_read/cache_creation across all
    assistant turns."""
    sid = "sess-metrics"
    payload = _make_session(
        [
            {"type": "permission-mode", "sessionId": sid, "permissionMode": "default"},
            {
                "type": "assistant",
                "sessionId": sid,
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "content": [{"type": "text", "text": "first"}],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 20,
                        "cache_read_input_tokens": 100,
                        "cache_creation_input_tokens": 50,
                    },
                },
            },
            {
                "type": "assistant",
                "sessionId": sid,
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "content": [{"type": "text", "text": "second"}],
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 15,
                        "cache_read_input_tokens": 200,
                        "cache_creation_input_tokens": 0,
                    },
                },
            },
        ]
    )
    traj = ClaudeCodeAdapter().to_atif(payload)
    assert traj is not None
    fm = traj.final_metrics
    assert fm is not None, (
        "final_metrics must aggregate across per-event usage blocks"
    )
    assert fm.total_prompt_tokens == 15
    assert fm.total_completion_tokens == 35
    assert fm.total_cached_tokens == 300
    extra = fm.extra or {}
    assert extra.get("cache_creation_input_tokens") == 50


def test_final_metrics_none_when_no_assistant_carries_usage() -> None:
    """Don't fabricate a final_metrics block when no usage data exists --
    keeping the field None signals "no token data available" honestly."""
    sid = "sess-no-usage"
    payload = _make_session(
        [
            {"type": "permission-mode", "sessionId": sid, "permissionMode": "default"},
            {
                "type": "user",
                "sessionId": sid,
                "cwd": "/tmp",
                "message": {"role": "user", "content": "just a question"},
            },
        ]
    )
    traj = ClaudeCodeAdapter().to_atif(payload)
    assert traj is not None
    assert traj.final_metrics is None
