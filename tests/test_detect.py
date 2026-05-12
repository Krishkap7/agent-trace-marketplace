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
