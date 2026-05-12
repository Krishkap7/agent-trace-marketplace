"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from trace_marketplace.ingest.pipeline import ingest_file
from trace_marketplace.storage.db import connect, init_schema, insert_trace

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Absolute path to ``tests/fixtures``."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def claude_code_minimal_path(fixtures_dir: Path) -> Path:
    """Path to the trimmed Claude Code JSONL fixture."""
    return fixtures_dir / "claude_code_minimal.jsonl"


@pytest.fixture(scope="session")
def swe_agent_sample_path(fixtures_dir: Path) -> Path:
    """Path to the synthetic SWE-agent HF row fixture."""
    return fixtures_dir / "swe_agent_sample.json"


@pytest.fixture(scope="session")
def codex_sample_path(fixtures_dir: Path) -> Path:
    """Path to the synthetic Codex session JSONL fixture."""
    return fixtures_dir / "codex_sample.jsonl"


@pytest.fixture(scope="session")
def cursor_sample_path(fixtures_dir: Path) -> Path:
    """Path to the synthetic Cursor export JSON fixture."""
    return fixtures_dir / "cursor_sample.cursor.json"


@pytest.fixture()
def populated_db(
    tmp_path: Path,
    claude_code_minimal_path: Path,
    swe_agent_sample_path: Path,
    codex_sample_path: Path,
    cursor_sample_path: Path,
) -> Iterator[sqlite3.Connection]:
    """Tmp SQLite DB pre-loaded with a deliberately mixed trace set.

    The mix is small but exercises everything the UI must cope with:

    * 4 real ingest paths (claude_code / swe_agent / codex / cursor)
      from the shipped fixtures -- gives us 4 distinct ``source_format``
      values, validated atif blobs, and real step counts.
    * 2 synthetic rows the test writes directly with
      :func:`insert_trace`, so we can pin the ``has_error`` 3-way filter:
      one ``has_error=0`` success-labelled row plus one ``atif IS NULL``
      / unknown-format fallback row.

    The unknown-format row also gives ``test_get_trace_with_null_atif``
    something to assert against without monkey-patching adapters.
    """
    db_path = tmp_path / "marketplace.db"
    conn = connect(db_path)
    init_schema(conn)

    for path in (
        claude_code_minimal_path,
        swe_agent_sample_path,
        codex_sample_path,
        cursor_sample_path,
    ):
        ingest_file(path, conn)

    # Synthetic "labelled success" row so the has_error=0 branch has a hit.
    insert_trace(
        conn,
        trace_id="__test_success__",
        source_format="atif",
        raw_blob='{"hello": "world"}',
        atif='{"schema_version": "ATIF-v1.7", "session_id": "s-success"}',
        agent_name="test-agent",
        model="test-model",
        num_steps=2,
        num_tool_calls=1,
        has_error=0,
    )
    # Synthetic unknown-format / atif-NULL row so the fallback path is
    # represented and ``has_atif=False`` has something to find.
    insert_trace(
        conn,
        trace_id="__test_unknown__",
        source_format="unknown",
        raw_blob="not actually a trace",
        atif=None,
        agent_name=None,
        model=None,
        num_steps=None,
        num_tool_calls=None,
        has_error=None,
    )

    try:
        yield conn
    finally:
        conn.close()
