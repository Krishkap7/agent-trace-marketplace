"""SQLite-backed trace storage.

This implements the bronze/silver/gold pattern from the design doc:

- ``raw_blob`` (TEXT, NOT NULL) is the bronze layer: the exact bytes a user
  uploaded, untouched, so we can always re-process from source.
- ``atif`` (TEXT, NULLABLE) is the silver layer: the validated/canonicalised
  ATIF representation. NULL when validation fails or when no adapter exists
  for the source format yet (e.g. unknown formats).
- The remaining ``agent_name``/``model``/``num_steps``/``num_tool_calls``
  columns are gold: cheap derived facts pulled out of the validated ATIF on
  insert so structured filter queries don't have to JSON-parse every row.
- ``has_error``/``failure_label``/``embedding_id`` are reserved for later
  enrichment slices and stay NULL for now.

JSON is stored as ``TEXT`` (not the JSON1 type) for simplicity; we'll move to
``JSONB`` if/when this graduates to Postgres.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "marketplace.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS traces (
    id                       TEXT PRIMARY KEY,
    source_format            TEXT NOT NULL,
    raw_blob                 TEXT NOT NULL,
    atif                     TEXT,
    ingested_at              TEXT NOT NULL,
    -- Derived from the validated ATIF on insert. ``agent_name`` is the
    -- ATIF ``agent.name`` field, i.e. the agent SYSTEM identifier ("mcp",
    -- "openhands", "claude-code", ...). It is NOT the per-task agent name
    -- (e.g. "accountant", "recruiter") which only exists implicitly in
    -- the source directory layout. Add a ``domain`` column later if needed.
    agent_name               TEXT,
    model                    TEXT,
    num_steps                INTEGER,
    num_tool_calls           INTEGER,
    -- Slice 2.5: derived from adapter-set ``trajectory.extra["resolved"]``
    -- when the source format carries ground truth (SWE-agent ``target``).
    -- Slice 4 fills this for the unlabelled formats from rule-based
    -- signals; benchmark labels are never overwritten (COALESCE keeps
    -- them).
    has_error                INTEGER,
    -- Slice 4: structural signal JSON (e.g. ``{"loop": true, ...}``)
    -- plus the LLM-judge label + one-sentence reasoning.
    failure_signals          TEXT,
    failure_label            TEXT,
    failure_label_reasoning  TEXT,
    -- Reserved for slice 5 (embeddings + semantic search).
    embedding_id             TEXT
);
"""


def connect(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection, creating the parent directory if needed."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _migrate_add_failure_columns(conn: sqlite3.Connection) -> None:
    """Idempotently add the slice 4 enrichment columns to an existing DB.

    SQLite's ``ALTER TABLE ADD COLUMN`` is non-idempotent and raises on
    duplicate columns, so we probe ``PRAGMA table_info`` first. A fresh
    DB created with the slice 4 ``SCHEMA_SQL`` already has both columns
    -- this function is a no-op in that case. Existing DBs from slice
    1-3 only have ``failure_label``; we add ``failure_signals`` and
    ``failure_label_reasoning`` here without re-creating the table.
    """
    cur = conn.execute("PRAGMA table_info(traces)")
    existing = {row[1] for row in cur.fetchall()}
    if not existing:
        # No traces table at all; SCHEMA_SQL above will create it
        # cleanly. ``init_schema`` calls us after that DDL, so this
        # branch is just defensive.
        return
    if "failure_signals" not in existing:
        conn.execute("ALTER TABLE traces ADD COLUMN failure_signals TEXT")
    if "failure_label_reasoning" not in existing:
        conn.execute("ALTER TABLE traces ADD COLUMN failure_label_reasoning TEXT")
    conn.commit()


def init_schema(conn: sqlite3.Connection) -> None:
    """Create the ``traces`` table if it doesn't exist, then run any
    idempotent column-addition migrations against an existing DB."""
    conn.executescript(SCHEMA_SQL)
    _migrate_add_failure_columns(conn)
    conn.commit()


def insert_trace(
    conn: sqlite3.Connection,
    *,
    trace_id: str,
    source_format: str,
    raw_blob: str,
    atif: str | None,
    agent_name: str | None,
    model: str | None,
    num_steps: int | None,
    num_tool_calls: int | None,
    has_error: int | None = None,
    ingested_at: str | None = None,
) -> None:
    """Insert (or replace) a single trace row.

    Uses ``INSERT OR REPLACE`` so re-running the ingest pipeline on the same
    source corpus is idempotent. The primary key is the trace's session/trace
    id, which is stable across re-ingestion.

    ``has_error`` is the only "enrichment" column the pipeline populates at
    ingest time. When the adapter sets ``trajectory.extra["resolved"]`` (a
    bool), the pipeline derives ``has_error = int(not resolved)`` and passes
    it here. Slice 2.5 wires this for SWE-agent traces, where ``target`` is
    ground truth. Other adapters leave it None and the column stays NULL,
    which slice 3's failure-detection pass will populate via structural
    rules / LLM judges.
    """
    if ingested_at is None:
        ingested_at = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """
        INSERT OR REPLACE INTO traces (
            id, source_format, raw_blob, atif, ingested_at,
            agent_name, model, num_steps, num_tool_calls, has_error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trace_id,
            source_format,
            raw_blob,
            atif,
            ingested_at,
            agent_name,
            model,
            num_steps,
            num_tool_calls,
            has_error,
        ),
    )
    conn.commit()


def count_traces(conn: sqlite3.Connection) -> int:
    """Return the total number of rows in ``traces``."""
    cur = conn.execute("SELECT COUNT(*) FROM traces")
    return int(cur.fetchone()[0])


def fetch_all_summaries(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return a compact summary row per trace, suitable for the verify CLI."""
    cur = conn.execute(
        """
        SELECT
            id,
            source_format,
            agent_name,
            model,
            num_steps,
            num_tool_calls,
            has_error,
            length(raw_blob) AS raw_blob_len,
            length(atif)     AS atif_len,
            ingested_at
        FROM traces
        ORDER BY id
        """
    )
    return [dict(row) for row in cur.fetchall()]


__all__ = [
    "DEFAULT_DB_PATH",
    "SCHEMA_SQL",
    "connect",
    "count_traces",
    "fetch_all_summaries",
    "init_schema",
    "insert_trace",
    "_migrate_add_failure_columns",
]


if __name__ == "__main__":
    # Smoke test: create the schema, insert one synthetic row, read it back.
    with connect() as _conn:
        init_schema(_conn)
        insert_trace(
            _conn,
            trace_id="__smoketest__",
            source_format="atif",
            raw_blob=json.dumps({"hello": "world"}),
            atif=json.dumps({"schema_version": "ATIF-v1.7"}),
            agent_name="test-agent",
            model="test-model",
            num_steps=0,
            num_tool_calls=0,
        )
        print(f"rows: {count_traces(_conn)}")
        for row in fetch_all_summaries(_conn):
            print(row)
        _conn.execute("DELETE FROM traces WHERE id = '__smoketest__'")
        _conn.commit()
