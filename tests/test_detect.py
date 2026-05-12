"""Unit tests for :func:`trace_marketplace.ingest.detect.detect_format`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_marketplace.ingest.detect import detect_format


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# ATIF (.json) detection
# --------------------------------------------------------------------------- #


def test_detects_atif_v1_7(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "traj.json",
        json.dumps({"schema_version": "ATIF-v1.7", "steps": []}),
    )
    assert detect_format(p) == "atif"


def test_detects_atif_older_version(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "traj.json",
        json.dumps({"schema_version": "ATIF-v1.2", "steps": []}),
    )
    assert detect_format(p) == "atif"


def test_json_without_schema_version_is_unknown(tmp_path: Path) -> None:
    p = _write(tmp_path / "x.json", json.dumps({"hello": "world"}))
    assert detect_format(p) == "unknown"


def test_malformed_json_returns_unknown(tmp_path: Path) -> None:
    p = _write(tmp_path / "x.json", "this is not json")
    assert detect_format(p) == "unknown"


# --------------------------------------------------------------------------- #
# SWE-agent (.json with the four required keys) detection
# --------------------------------------------------------------------------- #


def test_detects_swe_agent_row(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "row.json",
        json.dumps(
            {
                "instance_id": "AnalogJ__lexicon-336",
                "model_name": "swe-agent-llama-70b",
                "target": False,
                "trajectory": [{"role": "system", "system_prompt": "x", "text": ""}],
                "exit_status": "submitted",
            }
        ),
    )
    assert detect_format(p) == "swe_agent"


def test_swe_agent_takes_precedence_over_atif_keys(tmp_path: Path) -> None:
    """If a .json file somehow has BOTH schema_version AND the SWE-agent
    keys, the SWE-agent check fires first. (Defensive; shouldn't occur in
    real corpora, but pinning the precedence so a future regression is
    obvious.)"""
    p = _write(
        tmp_path / "weird.json",
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "instance_id": "x",
                "model_name": "y",
                "target": True,
                "trajectory": [],
            }
        ),
    )
    assert detect_format(p) == "swe_agent"


def test_atif_still_detected_when_swe_agent_keys_partial(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "atif.json",
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "instance_id": "x",  # only one of the four SWE-agent keys
                "steps": [],
            }
        ),
    )
    assert detect_format(p) == "atif"


# --------------------------------------------------------------------------- #
# Cursor (.cursor.json extension OR .json with v1 envelope shape) detection
# --------------------------------------------------------------------------- #


def test_detects_cursor_from_extension_hint(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "session.cursor.json",
        json.dumps(
            {
                "session_id": "x",
                "messages": [{"role": "user", "content": "hi"}],
            }
        ),
    )
    assert detect_format(p) == "cursor"


def test_detects_cursor_from_envelope_shape_without_extension(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "plain.json",
        json.dumps(
            {
                "session_id": "x",
                "messages": [{"role": "assistant", "content": "yo"}],
            }
        ),
    )
    assert detect_format(p) == "cursor"


def test_cursor_envelope_requires_role_on_first_message(tmp_path: Path) -> None:
    """Without a ``role`` field on messages[0] the file could be any
    other ``{session_id, messages}`` JSON; we don't claim it for Cursor."""
    p = _write(
        tmp_path / "ambiguous.json",
        json.dumps({"session_id": "x", "messages": [{"text": "hi"}]}),
    )
    assert detect_format(p) == "unknown"


def test_cursor_envelope_requires_non_empty_messages(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "ambiguous.json",
        json.dumps({"session_id": "x", "messages": []}),
    )
    assert detect_format(p) == "unknown"


def test_cursor_takes_precedence_over_atif_schema_version(tmp_path: Path) -> None:
    """The ``.cursor.json`` extension is an unambiguous Cursor signal
    even when the body somehow also carries ATIF schema markers."""
    p = _write(
        tmp_path / "weird.cursor.json",
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "session_id": "x",
                "messages": [{"role": "user", "content": "hi"}],
            }
        ),
    )
    assert detect_format(p) == "cursor"


# --------------------------------------------------------------------------- #
# Codex (.jsonl with OpenAI Responses-API outer event types) detection
# --------------------------------------------------------------------------- #


def test_detects_codex_from_session_meta_first_line(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "codex.jsonl",
        json.dumps(
            {
                "type": "session_meta",
                "payload": {"id": "sess-1", "cli_version": "0.32.0"},
            }
        ),
    )
    assert detect_format(p) == "codex"


def test_detects_codex_from_response_item_first_line(tmp_path: Path) -> None:
    """If a Codex JSONL gets sliced (no session_meta head), the first
    response_item line should still classify."""
    p = _write(
        tmp_path / "codex_no_head.jsonl",
        json.dumps(
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": []},
            }
        ),
    )
    assert detect_format(p) == "codex"


def test_codex_takes_precedence_over_claude_code(tmp_path: Path) -> None:
    """Codex's outer vocabulary doesn't overlap with Claude Code's. This
    test pins that ordering -- if a future event type collides, we'll
    catch it here."""
    p = _write(
        tmp_path / "ambiguous.jsonl",
        json.dumps(
            {
                "type": "session_meta",
                "sessionId": "ALSO_claude_looking",
                "payload": {"id": "x"},
            }
        ),
    )
    assert detect_format(p) == "codex"


def test_claude_code_still_wins_when_no_codex_marker(tmp_path: Path) -> None:
    """The discriminator: Claude Code first lines carry camelCase
    `sessionId` and have a vocabulary type like `permission-mode`."""
    p = _write(
        tmp_path / "cc.jsonl",
        json.dumps(
            {
                "type": "permission-mode",
                "sessionId": "abc-123",
                "mode": "auto",
            }
        ),
    )
    assert detect_format(p) == "claude_code"


def test_jsonl_with_unknown_top_level_type_is_unknown(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "weird.jsonl",
        json.dumps({"type": "completely-made-up-type", "payload": {}}),
    )
    assert detect_format(p) == "unknown"


def test_detects_codex_from_compacted_first_line(tmp_path: Path) -> None:
    """Forked Codex sessions can start with a `compacted` record. This
    is the 5th variant of RolloutItem per codex-rs/protocol/src/protocol.rs."""
    p = _write(
        tmp_path / "forked.jsonl",
        json.dumps(
            {
                "type": "compacted",
                "payload": {"message": "summary text", "replacement_history": []},
            }
        ),
    )
    assert detect_format(p) == "codex"


# --------------------------------------------------------------------------- #
# Claude Code (.jsonl) detection
# --------------------------------------------------------------------------- #


def test_detects_claude_code_from_user_first_event(tmp_path: Path) -> None:
    line = json.dumps({"type": "user", "sessionId": "abc", "message": {}})
    p = _write(tmp_path / "session.jsonl", line + "\n")
    assert detect_format(p) == "claude_code"


def test_detects_claude_code_from_permission_mode_first_event(tmp_path: Path) -> None:
    """Real Fleet sessions often start with permission-mode -- must accept."""
    line = json.dumps(
        {"type": "permission-mode", "sessionId": "abc", "permissionMode": "default"}
    )
    p = _write(tmp_path / "session.jsonl", line + "\n")
    assert detect_format(p) == "claude_code"


def test_detects_claude_code_from_attachment_first_event(tmp_path: Path) -> None:
    line = json.dumps(
        {"type": "attachment", "sessionId": "abc", "attachment": {"type": "auto_mode"}}
    )
    p = _write(tmp_path / "session.jsonl", line + "\n")
    assert detect_format(p) == "claude_code"


def test_jsonl_skips_leading_blank_lines(tmp_path: Path) -> None:
    body = (
        "\n\n   \n"
        + json.dumps({"type": "user", "sessionId": "abc", "message": {}})
        + "\n"
    )
    p = _write(tmp_path / "session.jsonl", body)
    assert detect_format(p) == "claude_code"


def test_jsonl_without_session_id_is_unknown(tmp_path: Path) -> None:
    line = json.dumps({"type": "user", "message": {}})  # no sessionId
    p = _write(tmp_path / "x.jsonl", line + "\n")
    assert detect_format(p) == "unknown"


def test_jsonl_unrecognized_first_event_type_is_unknown(tmp_path: Path) -> None:
    line = json.dumps({"type": "bogus", "sessionId": "abc"})
    p = _write(tmp_path / "x.jsonl", line + "\n")
    assert detect_format(p) == "unknown"


def test_empty_jsonl_is_unknown(tmp_path: Path) -> None:
    p = _write(tmp_path / "x.jsonl", "")
    assert detect_format(p) == "unknown"


def test_malformed_jsonl_first_line_is_unknown(tmp_path: Path) -> None:
    p = _write(tmp_path / "x.jsonl", "this is not json\n")
    assert detect_format(p) == "unknown"


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("suffix", [".txt", ".bin", "", ".csv"])
def test_unrecognized_extension_is_unknown(tmp_path: Path, suffix: str) -> None:
    p = _write(tmp_path / f"file{suffix}", '{"schema_version": "ATIF-v1.7"}')
    assert detect_format(p) == "unknown"


def test_does_not_raise_on_nonexistent_path(tmp_path: Path) -> None:
    """Missing file shouldn't crash detection; should fall through to unknown."""
    p = tmp_path / "does_not_exist.json"
    # Either returns "unknown" or raises FileNotFoundError -- spec says
    # "never raises". Pin behaviour: detect_format swallows OS errors.
    assert detect_format(p) == "unknown"
