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

from trace_marketplace.ingest.pipeline import _trace_id_for, ingest_file
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


def test_trace_id_falls_back_to_path_stem(tmp_path: Path) -> None:
    """Preference order #3: file stem when ATIF identifiers are absent."""
    p = tmp_path / "my-session.jsonl"
    p.write_text("body", encoding="utf-8")
    trace_id = _trace_id_for(None, p, "body")
    assert trace_id == "my-session"


def test_trace_id_falls_back_to_content_hash_when_path_stem_empty() -> None:
    """Preference order #4 (used to be uuid4, now deterministic content
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
