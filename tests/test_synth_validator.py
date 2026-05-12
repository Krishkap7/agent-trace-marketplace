"""Tests for :mod:`trace_marketplace.synth.validator`.

We drive the validator with real (tiny) ATIF-conformant trace text
for each format, plus malformed variants. Adapters are not mocked --
the whole point is to validate against the production adapter
behaviour.
"""

from __future__ import annotations

import json

from trace_marketplace.synth.spec import (
    FORMAT_CLAUDE_CODE,
    FORMAT_CODEX,
    FORMAT_CURSOR,
)
from trace_marketplace.synth.validator import MIN_STEPS, validate


# ---------------------------------------------------------------------------
# Cursor fixture (smallest format -- single JSON envelope).
# ---------------------------------------------------------------------------


def _cursor_trace(*, n_steps: int = 6) -> str:
    """Build a Cursor v1 envelope with ``n_steps`` exchanges + 1 tool call.

    The envelope has one user prompt, ``n_steps-2`` assistant turns
    (each calling read_file or edit_file), and a final assistant
    summary. Tool results land on the FOLLOWING message; one
    tool_call per assistant turn keeps the tool-call count above
    the validator's minimum.
    """
    messages: list[dict[str, object]] = [
        {"role": "user", "content": "Fix tests/test_api.py", "timestamp": "2026-01-01T00:00:00Z"},
    ]
    for i in range(n_steps - 2):
        tc_id = f"tc_{i}"
        messages.append(
            {
                "role": "assistant",
                "content": f"Step {i}: let me read the file.",
                "tool_calls": [
                    {"id": tc_id, "name": "read_file", "arguments": {"path": "tests/test_api.py"}}
                ],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": "",
                "tool_results": [{"tool_call_id": tc_id, "content": "def test_x(): assert True"}],
            }
        )
    messages.append({"role": "assistant", "content": "Done."})
    envelope = {
        "session_id": "test-session-cursor",
        "cursor_version": "0.45.0",
        "model": "claude-3.5-sonnet",
        "synthetic": True,
        "synthetic_failure_mode": "loop",
        "messages": messages,
    }
    return json.dumps(envelope)


def test_validate_accepts_valid_cursor_trace() -> None:
    result = validate(FORMAT_CURSOR, _cursor_trace(n_steps=8))
    assert result.ok, result.error
    assert result.trajectory is not None
    assert len(result.trajectory.steps) >= MIN_STEPS


def test_validate_rejects_short_trace() -> None:
    result = validate(FORMAT_CURSOR, _cursor_trace(n_steps=4))
    assert not result.ok
    assert "too few steps" in result.error


def test_validate_rejects_empty_string() -> None:
    result = validate(FORMAT_CURSOR, "")
    assert not result.ok
    assert "empty" in result.error


def test_validate_rejects_malformed_json() -> None:
    result = validate(FORMAT_CURSOR, "{ not json")
    assert not result.ok
    assert result.trajectory is None


def test_validate_rejects_no_tool_calls() -> None:
    """Trace with only chatter (no tool calls) is rejected as degenerate."""
    envelope = {
        "session_id": "no-tools",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "you're welcome"},
            {"role": "user", "content": "bye"},
            {"role": "assistant", "content": "goodbye"},
        ],
    }
    result = validate(FORMAT_CURSOR, json.dumps(envelope))
    assert not result.ok
    assert "no tool calls" in result.error


def test_validate_unknown_format_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        validate("not_a_format", "{}")


# ---------------------------------------------------------------------------
# Claude Code fixture (JSONL).
# ---------------------------------------------------------------------------


def _claude_code_trace() -> str:
    """Minimal valid Claude Code JSONL with 4 tool_use / tool_result pairs.

    Step count math: the adapter does NOT emit a step for "tool_result
    carrier" user events (those attach to the prior assistant step).
    So 1 user + N assistant + 1 final assistant = N+2 steps. We need
    >= 5 steps to clear the validator's MIN_STEPS floor.
    """
    lines: list[dict] = []
    lines.append(
        {
            "type": "user",
            "sessionId": "cc-test",
            "uuid": "u1",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "message": {"role": "user", "content": "Read tests/test_api.py"},
        }
    )
    for i in range(4):
        tool_id = f"toolu_{i}"
        lines.append(
            {
                "type": "assistant",
                "sessionId": "cc-test",
                "uuid": f"a{i}",
                "timestamp": f"2026-01-01T00:0{i + 1}:00.000Z",
                "message": {
                    "id": f"msg_{i}",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5",
                    "content": [
                        {"type": "text", "text": f"Reading attempt {i}"},
                        {"type": "tool_use", "id": tool_id, "name": "Read", "input": {"file": "x"}},
                    ],
                },
            }
        )
        lines.append(
            {
                "type": "user",
                "sessionId": "cc-test",
                "uuid": f"u_tr_{i}",
                "timestamp": f"2026-01-01T00:0{i + 1}:30.000Z",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": tool_id, "content": "file contents"}
                    ],
                },
            }
        )
    lines.append(
        {
            "type": "assistant",
            "sessionId": "cc-test",
            "uuid": "a_final",
            "timestamp": "2026-01-01T00:05:00.000Z",
            "message": {
                "id": "msg_final",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "Done."}],
            },
        }
    )
    return "\n".join(json.dumps(line) for line in lines)


def test_validate_accepts_valid_claude_code_trace() -> None:
    result = validate(FORMAT_CLAUDE_CODE, _claude_code_trace())
    assert result.ok, result.error
    assert result.trajectory is not None


# ---------------------------------------------------------------------------
# Codex fixture (JSONL).
# ---------------------------------------------------------------------------


def _codex_trace() -> str:
    """Minimal valid Codex JSONL with session_meta + tool calls."""
    lines: list[dict] = []
    lines.append(
        {
            "type": "session_meta",
            "payload": {
                "id": "codex-test",
                "timestamp": "2026-01-01T00:00:00.000Z",
                "cli_version": "0.20.0",
                "originator": "codex_cli",
                "cwd": "/tmp",
                "instructions": "",
            },
        }
    )
    lines.append(
        {
            "type": "turn_context",
            "payload": {"cwd": "/tmp", "model": "gpt-5.5", "summary": "auto"},
        }
    )
    lines.append(
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Fix the test"}],
            },
        }
    )
    for i in range(4):
        cid = f"call_{i}"
        lines.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "shell",
                    "call_id": cid,
                    "arguments": json.dumps({"command": ["cat", f"file_{i}.py"]}),
                },
            }
        )
        lines.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": cid,
                    "output": json.dumps({"output": "file contents", "metadata": {"exit_code": 0}}),
                },
            }
        )
    lines.append(
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Done."}],
            },
        }
    )
    return "\n".join(json.dumps(line) for line in lines)


def test_validate_accepts_valid_codex_trace() -> None:
    result = validate(FORMAT_CODEX, _codex_trace())
    assert result.ok, result.error
    assert result.trajectory is not None
