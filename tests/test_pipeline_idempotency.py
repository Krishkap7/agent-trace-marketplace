"""Idempotency tests for :func:`trace_marketplace.ingest.pipeline.ingest_file`.

The pipeline's ``insert_trace`` uses ``INSERT OR REPLACE`` on the
trace primary key, which only delivers the documented idempotency
guarantee if the trace_id is itself a deterministic function of the
input. A non-deterministic fallback (e.g. ``uuid.uuid4()`` — the bug
Cursor bugbot caught on PR #1, slice 1) silently breaks this: every
re-run produces a fresh primary key and the row count grows on every
ingest.

These tests pin the contract: same file content -> same trace_id ->
single row, regardless of how many times we ingest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_marketplace.ingest.pipeline import (
    _extract_raw_identity,
    _trace_id_for,
    ingest_file,
)
from trace_marketplace.storage.db import connect, count_traces, init_schema


# --------------------------------------------------------------------------- #
# _trace_id_for unit tests (no DB)
# --------------------------------------------------------------------------- #


def test_trace_id_prefers_trajectory_id_when_available(tmp_path: Path) -> None:
    """Documented preference order #1: ATIF trajectory_id wins."""

    class _FakeTraj:
        trajectory_id = "the-trajectory"
        session_id = "the-session"

    trace_id = _trace_id_for(_FakeTraj(), tmp_path / "anything.json", "raw text")
    assert trace_id == "the-trajectory"


def test_trace_id_falls_back_to_session_id(tmp_path: Path) -> None:
    """Preference order #2: ATIF session_id when no trajectory_id."""

    class _FakeTraj:
        trajectory_id = None
        session_id = "the-session"

    trace_id = _trace_id_for(_FakeTraj(), tmp_path / "name.json", "raw text")
    assert trace_id == "the-session"


def test_trace_id_recovers_atif_session_id_when_validation_fails(
    tmp_path: Path,
) -> None:
    """Preference order #3 (validation-failure path): pull session_id
    out of the raw JSON even when Pydantic validation produced no
    Trajectory. Regression for the bugbot finding that all 5 ATIF
    benchmark files collide on ``"trajectory"`` (the shared stem)
    when validation fails."""
    raw = json.dumps(
        {
            "schema_version": "ATIF-v1.7",
            "session_id": "mt-ops-recruiter__RuKFi6Dx",
            "agent": {"name": "mcp"},  # missing required fields -> fails Pydantic
        }
    )
    trace_id = _trace_id_for(None, tmp_path / "trajectory.json", raw)
    assert trace_id == "mt-ops-recruiter__RuKFi6Dx"


def test_trace_id_recovers_swe_agent_instance_id_when_validation_fails(
    tmp_path: Path,
) -> None:
    """SWE-agent rows are identified by ``instance_id``. Validation can
    fail (e.g. malformed trajectory entry) yet the row's identity is
    still recoverable from the raw JSON."""
    raw = json.dumps({"instance_id": "scikit-learn__scikit-learn-12345"})
    trace_id = _trace_id_for(None, tmp_path / "row.json", raw)
    assert trace_id == "scikit-learn__scikit-learn-12345"


def test_trace_id_recovers_claude_code_session_id_when_validation_fails(
    tmp_path: Path,
) -> None:
    """Claude Code first event carries camelCase ``sessionId``."""
    raw = json.dumps(
        {"type": "user", "sessionId": "abc-claude-uuid", "message": {}}
    )
    trace_id = _trace_id_for(None, tmp_path / "session.jsonl", raw)
    assert trace_id == "abc-claude-uuid"


def test_trace_id_recovers_codex_session_id_when_validation_fails(
    tmp_path: Path,
) -> None:
    """Codex first event is session_meta with the id under payload.id."""
    raw = json.dumps(
        {
            "type": "session_meta",
            "payload": {"id": "sess-real-codex-id", "cli_version": "0.32.0"},
        }
    )
    trace_id = _trace_id_for(None, tmp_path / "rollout.jsonl", raw)
    assert trace_id == "sess-real-codex-id"


def test_trace_id_falls_back_to_path_stem(tmp_path: Path) -> None:
    """Preference order #4: file stem when no identity is recoverable
    from either the validated Trajectory or the raw bytes."""
    p = tmp_path / "my-session.jsonl"
    p.write_text("not-json text body", encoding="utf-8")
    trace_id = _trace_id_for(None, p, "not-json text body")
    assert trace_id == "my-session"


def test_trace_id_falls_back_to_content_hash_when_path_stem_empty() -> None:
    """Preference order #5 (used to be uuid4, now deterministic content
    hash). The exact format is ``sha256:<32-hex-chars>``; what matters
    is that the SAME raw_text always produces the SAME id."""
    blob = "the file's raw contents"
    id1 = _trace_id_for(None, Path(""), blob)
    id2 = _trace_id_for(None, Path(""), blob)
    assert id1 == id2
    assert id1.startswith("sha256:")
    assert len(id1) == len("sha256:") + 32


def test_trace_id_content_hash_distinguishes_different_blobs() -> None:
    """Different raw text MUST produce different content-hash ids."""
    a = _trace_id_for(None, Path(""), "blob A")
    b = _trace_id_for(None, Path(""), "blob B")
    assert a != b


def test_two_failed_atif_validations_with_same_filename_get_distinct_ids(
    tmp_path: Path,
) -> None:
    """The most-impactful collision case: ATIF benchmark layout has
    every file named ``trajectory.json`` under a per-domain dir. If
    validation fails for two of them, the OLD path-stem-only fallback
    would collide them onto the single id ``"trajectory"`` and the
    second INSERT OR REPLACE would silently overwrite the first.

    With raw-identity peek, the two distinct session_ids are
    recovered even on validation failure."""
    raw_a = json.dumps(
        {"schema_version": "ATIF-bad", "session_id": "domain-a-session"}
    )
    raw_b = json.dumps(
        {"schema_version": "ATIF-bad", "session_id": "domain-b-session"}
    )
    id_a = _trace_id_for(None, tmp_path / "a" / "trajectory.json", raw_a)
    id_b = _trace_id_for(None, tmp_path / "b" / "trajectory.json", raw_b)
    assert id_a != id_b
    assert id_a == "domain-a-session"
    assert id_b == "domain-b-session"


# --------------------------------------------------------------------------- #
# _extract_raw_identity unit coverage
# --------------------------------------------------------------------------- #


def test_extract_raw_identity_returns_none_on_empty_input() -> None:
    assert _extract_raw_identity("") is None
    assert _extract_raw_identity("   \n  \n") is None


def test_extract_raw_identity_returns_none_on_unparseable_input() -> None:
    """Free-form text that's neither JSON nor JSONL should return None,
    not raise -- callers rely on this to fall through to path.stem."""
    assert _extract_raw_identity("hello world, this is not json") is None


def test_extract_raw_identity_prefers_trajectory_id_over_session_id() -> None:
    raw = json.dumps(
        {"trajectory_id": "TJ-99", "session_id": "SS-11"}
    )
    assert _extract_raw_identity(raw) == "TJ-99"


def test_extract_raw_identity_finds_codex_sessionmeta_in_single_line_jsonl() -> None:
    """A JSONL file with exactly one event parses as a JSON object on
    its whole. Identity must still be recovered from
    ``session_meta.payload.id``."""
    raw = json.dumps(
        {"type": "session_meta", "payload": {"id": "sess-x", "cli_version": "0.32"}}
    )
    assert _extract_raw_identity(raw) == "sess-x"


def test_extract_raw_identity_finds_claude_code_session_id_in_single_line() -> None:
    raw = json.dumps({"type": "user", "sessionId": "claude-uuid", "message": {}})
    assert _extract_raw_identity(raw) == "claude-uuid"


def test_extract_raw_identity_finds_codex_session_meta_in_multi_line_jsonl() -> None:
    """Multi-line JSONL: whole-document JSON parse fails; first-line
    interpretation must succeed."""
    raw = "\n".join(
        [
            json.dumps(
                {
                    "type": "session_meta",
                    "payload": {"id": "real-codex-sess", "cli_version": "0.32"},
                }
            ),
            json.dumps(
                {"type": "response_item", "payload": {"type": "message"}}
            ),
        ]
    )
    assert _extract_raw_identity(raw) == "real-codex-sess"


def test_extract_raw_identity_returns_none_when_no_known_field() -> None:
    """If no known identity field exists in either interpretation,
    we return None and let the caller fall back to path.stem."""
    raw = json.dumps({"some_field": "value", "other": 42})
    assert _extract_raw_identity(raw) is None


# --------------------------------------------------------------------------- #
# End-to-end idempotency through ingest_file + SQLite
# --------------------------------------------------------------------------- #


def _atif_payload() -> str:
    """Minimal valid ATIF v1.7 document for an idempotency probe."""
    return json.dumps(
        {
            "schema_version": "ATIF-v1.7",
            "session_id": "idempotency-probe-1",
            "agent": {"name": "test", "version": "0.0.0"},
            "steps": [{"step_id": 1, "source": "user", "message": "hi"}],
        }
    )


def test_ingest_file_is_idempotent_with_path_stem(tmp_path: Path) -> None:
    """Same file ingested 3x -> exactly 1 row in the traces table."""
    path = tmp_path / "session-x.json"
    path.write_text(_atif_payload(), encoding="utf-8")
    db = tmp_path / "marketplace.db"
    with connect(db) as conn:
        init_schema(conn)
        results = [ingest_file(path, conn) for _ in range(3)]
        assert {r.trace_id for r in results} == {"idempotency-probe-1"}
        assert count_traces(conn) == 1


def test_ingest_file_is_idempotent_for_unknown_format(tmp_path: Path) -> None:
    """Files with no adapter (unknown format) must still land on a
    deterministic id -- the bug we're patching primarily bit this
    code path because no ATIF identifiers were available."""
    path = tmp_path / "mystery.bin"
    path.write_text("opaque payload", encoding="utf-8")
    db = tmp_path / "marketplace.db"
    with connect(db) as conn:
        init_schema(conn)
        first = ingest_file(path, conn)
        second = ingest_file(path, conn)
        assert first.trace_id == second.trace_id == "mystery"
        assert count_traces(conn) == 1


def test_ingest_file_content_hash_id_when_no_other_identity(tmp_path: Path) -> None:
    """When path stem is empty AND no trajectory identifiers exist,
    re-ingest still produces the same row (via content hash)."""
    # ``Path("/some/dir/")`` has empty stem; emulate that case with a
    # symlink-ish setup: we can't actually create a real empty-stem
    # file on disk, so we drive _trace_id_for + insert directly.
    from trace_marketplace.storage.db import fetch_all_summaries, insert_trace

    db = tmp_path / "marketplace.db"
    blob = "raw bytes here"
    with connect(db) as conn:
        init_schema(conn)
        for _ in range(2):
            tid = _trace_id_for(None, Path(""), blob)
            insert_trace(
                conn,
                trace_id=tid,
                source_format="unknown",
                raw_blob=blob,
                atif=None,
                agent_name=None,
                model=None,
                num_steps=None,
                num_tool_calls=None,
            )
        rows = fetch_all_summaries(conn)
    assert len(rows) == 1
    assert rows[0]["id"].startswith("sha256:")
