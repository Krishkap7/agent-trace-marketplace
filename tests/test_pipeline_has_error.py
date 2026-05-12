"""Tests for the has_error pipeline rule introduced in slice 2.5.

Covers ``trace_marketplace.ingest.pipeline._derive_has_error`` directly plus
a round-trip through ``ingest_file`` to confirm the value lands in the
SQLite column.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from harbor.models.trajectories import Agent, Step, Trajectory
from trace_marketplace.ingest.pipeline import _derive_has_error, ingest_file
from trace_marketplace.storage.db import connect, init_schema

_MINIMAL_STEP = Step(step_id=1, source="user", message="placeholder")
_MINIMAL_STEP_DICT = {"step_id": 1, "source": "user", "message": "placeholder"}


def _build_trajectory(extra: dict | None = None) -> Trajectory:
    """Minimal valid Trajectory for the purposes of has_error tests.

    ATIF requires steps >= 1; one stub user step is enough.
    """
    return Trajectory(
        schema_version="ATIF-v1.7",
        session_id="test-has-error",
        agent=Agent(name="test", version="0.0.0"),
        steps=[_MINIMAL_STEP],
        extra=extra,
    )


# --------------------------------------------------------------------------- #
# Pure unit tests on _derive_has_error
# --------------------------------------------------------------------------- #


def test_resolved_true_maps_to_zero() -> None:
    traj = _build_trajectory(extra={"resolved": True})
    assert _derive_has_error(traj) == 0


def test_resolved_false_maps_to_one() -> None:
    traj = _build_trajectory(extra={"resolved": False})
    assert _derive_has_error(traj) == 1


def test_no_resolved_key_maps_to_none() -> None:
    traj = _build_trajectory(extra={"other_key": "value"})
    assert _derive_has_error(traj) is None


def test_missing_extra_maps_to_none() -> None:
    traj = _build_trajectory(extra=None)
    assert _derive_has_error(traj) is None


@pytest.mark.parametrize(
    "not_a_bool", [None, "true", "false", 0, 1, [True], {"x": True}]
)
def test_non_bool_resolved_maps_to_none(not_a_bool: object) -> None:
    """Defensive: only honour an actual Python bool. Strings/ints stay NULL.

    SQLite would happily coerce 0/1 to INTEGER, but that would mask adapter
    bugs where an int sneaks in expecting bool semantics.
    """
    traj = _build_trajectory(extra={"resolved": not_a_bool})
    assert _derive_has_error(traj) is None


# --------------------------------------------------------------------------- #
# Round-trip through ingest_file: synthesise a passthrough-ATIF file with
# extra.resolved set and confirm the has_error column gets populated.
# --------------------------------------------------------------------------- #


def _write_atif_with_resolved(path: Path, *, resolved: bool, session_id: str) -> Path:
    payload = {
        "schema_version": "ATIF-v1.7",
        "session_id": session_id,
        "agent": {"name": "test", "version": "0.0.0"},
        "steps": [_MINIMAL_STEP_DICT],
        "extra": {"resolved": resolved},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _read_has_error(conn: sqlite3.Connection, trace_id: str) -> int | None:
    cur = conn.execute("SELECT has_error FROM traces WHERE id = ?", (trace_id,))
    row = cur.fetchone()
    return None if row is None else row[0]


def test_ingest_persists_has_error_zero_when_resolved_true(tmp_path: Path) -> None:
    p = _write_atif_with_resolved(
        tmp_path / "ok.json", resolved=True, session_id="he-zero"
    )
    db_path = tmp_path / "test.db"
    with connect(db_path) as conn:
        init_schema(conn)
        result = ingest_file(p, conn)
        assert result.validated
        assert _read_has_error(conn, "he-zero") == 0


def test_ingest_persists_has_error_one_when_resolved_false(tmp_path: Path) -> None:
    p = _write_atif_with_resolved(
        tmp_path / "fail.json", resolved=False, session_id="he-one"
    )
    db_path = tmp_path / "test.db"
    with connect(db_path) as conn:
        init_schema(conn)
        result = ingest_file(p, conn)
        assert result.validated
        assert _read_has_error(conn, "he-one") == 1


def test_ingest_leaves_has_error_null_when_extra_absent(tmp_path: Path) -> None:
    """The slice-1/2 backwards-compat case -- traces with no resolved key."""
    payload = {
        "schema_version": "ATIF-v1.7",
        "session_id": "he-null",
        "agent": {"name": "test", "version": "0.0.0"},
        "steps": [_MINIMAL_STEP_DICT],
    }
    p = tmp_path / "no_extra.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    db_path = tmp_path / "test.db"
    with connect(db_path) as conn:
        init_schema(conn)
        result = ingest_file(p, conn)
        assert result.validated
        assert _read_has_error(conn, "he-null") is None
