"""Schema migration tests for slice 10.

Pins:

* ``init_schema`` is idempotent over the four new columns
  (``failure_events``, ``recovered_recommendations``,
  ``failure_label_source``, ``ultimately_succeeded``).
* ``failure_label_source`` backfill correctly distinguishes SWE-agent
  rows from other-format judged rows.
* SWE-agent ``ultimately_succeeded`` backfill reads ``target`` from
  ``raw_blob`` and matches the ``1 - has_error`` invariant.
* The cross-check raises loudly when ``target`` and ``has_error``
  disagree -- i.e. when the adapter is broken.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from trace_marketplace.storage.db import (
    _backfill_failure_label_source,
    _backfill_ultimately_succeeded_swe_agent,
    connect,
    init_schema,
    insert_trace,
)


def _columns(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA table_info(traces)")}


def _new_db(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / f"sl10_{id(tmp_path)}.db"
    conn = connect(db_path)
    init_schema(conn)
    return conn


# ---------- idempotency ----------


def test_init_schema_adds_all_four_new_columns(tmp_path: Path) -> None:
    conn = _new_db(tmp_path)
    try:
        cols = _columns(conn)
        for col in (
            "failure_events",
            "recovered_recommendations",
            "failure_label_source",
            "ultimately_succeeded",
        ):
            assert col in cols, f"Missing column: {col}"
    finally:
        conn.close()


def test_init_schema_is_idempotent(tmp_path: Path) -> None:
    conn = _new_db(tmp_path)
    try:
        cols_first = _columns(conn)
        # Run a few more times -- each call must be a no-op.
        for _ in range(3):
            init_schema(conn)
        assert _columns(conn) == cols_first
    finally:
        conn.close()


# ---------- failure_label_source backfill ----------


def test_failure_label_source_backfill_swe_agent_vs_judge(tmp_path: Path) -> None:
    conn = _new_db(tmp_path)
    try:
        # SWE-agent row with a benchmark-derived failure_label.
        insert_trace(
            conn,
            trace_id="swe-1",
            source_format="swe_agent",
            raw_blob=json.dumps({"target": False}),
            atif='{"schema_version": "ATIF-v1.7"}',
            agent_name="swe",
            model="m",
            num_steps=10,
            num_tool_calls=5,
            has_error=1,
        )
        conn.execute("UPDATE traces SET failure_label = 'gave_up' WHERE id = 'swe-1'")
        # Claude Code row with a judge-derived failure_label.
        insert_trace(
            conn,
            trace_id="cc-1",
            source_format="claude_code",
            raw_blob="{}",
            atif='{"schema_version": "ATIF-v1.7"}',
            agent_name="claude",
            model="m",
            num_steps=10,
            num_tool_calls=5,
            has_error=1,
        )
        conn.execute("UPDATE traces SET failure_label = 'loop' WHERE id = 'cc-1'")
        # Clean trace with no label -- stays NULL.
        insert_trace(
            conn,
            trace_id="clean-1",
            source_format="claude_code",
            raw_blob="{}",
            atif='{"schema_version": "ATIF-v1.7"}',
            agent_name="claude",
            model="m",
            num_steps=5,
            num_tool_calls=2,
            has_error=0,
        )
        # Clear the slice-10 columns we want to repopulate.
        conn.execute("UPDATE traces SET failure_label_source = NULL")
        conn.commit()

        _backfill_failure_label_source(conn)

        result = {
            row["id"]: row["failure_label_source"]
            for row in conn.execute("SELECT id, failure_label_source FROM traces")
        }
        assert result["swe-1"] == "swe_agent_target"
        assert result["cc-1"] == "sonnet_judge"
        assert result["clean-1"] is None
    finally:
        conn.close()


def test_failure_label_source_backfill_is_idempotent(tmp_path: Path) -> None:
    conn = _new_db(tmp_path)
    try:
        insert_trace(
            conn,
            trace_id="swe-1",
            source_format="swe_agent",
            raw_blob=json.dumps({"target": True}),
            atif="{}",
            agent_name="swe",
            model="m",
            num_steps=10,
            num_tool_calls=5,
            has_error=0,
        )
        conn.execute("UPDATE traces SET failure_label = 'success' WHERE id = 'swe-1'")
        conn.commit()
        # Two backfill passes must converge on the same result.
        _backfill_failure_label_source(conn)
        first = conn.execute(
            "SELECT failure_label_source FROM traces WHERE id='swe-1'"
        ).fetchone()[0]
        _backfill_failure_label_source(conn)
        second = conn.execute(
            "SELECT failure_label_source FROM traces WHERE id='swe-1'"
        ).fetchone()[0]
        assert first == second == "swe_agent_target"
    finally:
        conn.close()


# ---------- ultimately_succeeded backfill ----------


def test_swe_agent_ultimately_succeeded_matches_target(tmp_path: Path) -> None:
    conn = _new_db(tmp_path)
    try:
        insert_trace(
            conn,
            trace_id="resolved",
            source_format="swe_agent",
            raw_blob=json.dumps({"target": True, "exit_status": "submitted"}),
            atif='{"schema_version": "ATIF-v1.7"}',
            agent_name="swe",
            model="m",
            num_steps=10,
            num_tool_calls=5,
            has_error=0,
        )
        insert_trace(
            conn,
            trace_id="unresolved",
            source_format="swe_agent",
            raw_blob=json.dumps({"target": False}),
            atif='{"schema_version": "ATIF-v1.7"}',
            agent_name="swe",
            model="m",
            num_steps=10,
            num_tool_calls=5,
            has_error=1,
        )
        conn.execute("UPDATE traces SET ultimately_succeeded = NULL")
        conn.commit()
        _backfill_ultimately_succeeded_swe_agent(conn)
        result = {
            row["id"]: row["ultimately_succeeded"]
            for row in conn.execute("SELECT id, ultimately_succeeded FROM traces")
        }
        assert result["resolved"] == 1
        assert result["unresolved"] == 0
    finally:
        conn.close()


def test_swe_agent_invariant_raises_on_mismatch(tmp_path: Path) -> None:
    """Adapter-bug guard: target=True with has_error=1 (or vice versa)
    is impossible under the documented adapter contract; the migration
    must abort loudly rather than write inconsistent data."""
    conn = _new_db(tmp_path)
    try:
        insert_trace(
            conn,
            trace_id="bad",
            source_format="swe_agent",
            raw_blob=json.dumps({"target": True}),  # implies succeeded
            atif="{}",
            agent_name="swe",
            model="m",
            num_steps=10,
            num_tool_calls=5,
            has_error=1,  # contradicts target=True
        )
        conn.execute("UPDATE traces SET ultimately_succeeded = NULL")
        conn.commit()
        with pytest.raises(RuntimeError, match="invariant violated"):
            _backfill_ultimately_succeeded_swe_agent(conn)
    finally:
        conn.close()


def test_ultimately_succeeded_backfill_skips_non_swe_agent(
    tmp_path: Path,
) -> None:
    """Only SWE-agent rows are backfilled; other formats stay NULL until
    the events extractor runs."""
    conn = _new_db(tmp_path)
    try:
        insert_trace(
            conn,
            trace_id="cc-1",
            source_format="claude_code",
            raw_blob="{}",
            atif="{}",
            agent_name="claude",
            model="m",
            num_steps=10,
            num_tool_calls=5,
            has_error=1,
        )
        _backfill_ultimately_succeeded_swe_agent(conn)
        succeeded = conn.execute(
            "SELECT ultimately_succeeded FROM traces WHERE id='cc-1'"
        ).fetchone()[0]
        assert succeeded is None
    finally:
        conn.close()


def test_ultimately_succeeded_backfill_is_idempotent(tmp_path: Path) -> None:
    conn = _new_db(tmp_path)
    try:
        insert_trace(
            conn,
            trace_id="swe-r",
            source_format="swe_agent",
            raw_blob=json.dumps({"target": True}),
            atif="{}",
            agent_name="swe",
            model="m",
            num_steps=10,
            num_tool_calls=5,
            has_error=0,
        )
        conn.commit()
        _backfill_ultimately_succeeded_swe_agent(conn)
        first = conn.execute(
            "SELECT ultimately_succeeded FROM traces WHERE id='swe-r'"
        ).fetchone()[0]
        # Running again must be a no-op (the WHERE clause skips
        # already-populated rows) and the value stays correct.
        _backfill_ultimately_succeeded_swe_agent(conn)
        second = conn.execute(
            "SELECT ultimately_succeeded FROM traces WHERE id='swe-r'"
        ).fetchone()[0]
        assert first == second == 1
    finally:
        conn.close()
