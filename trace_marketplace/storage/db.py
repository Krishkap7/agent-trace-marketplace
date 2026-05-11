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
    id              TEXT PRIMARY KEY,
    source_format   TEXT NOT NULL,
    raw_blob        TEXT NOT NULL,
    atif            TEXT,
    ingested_at     TEXT NOT NULL,
    -- Derived from the validated ATIF on insert. ``agent_name`` is the ATIF
    -- ``agent.name`` field, i.e. the agent SYSTEM identifier ("mcp",
    -- "openhands", "claude-code", ...). It is NOT the per-task agent name
    -- (e.g. "accountant", "recruiter") which only exists implicitly in the
    -- source directory layout. Add a ``domain`` column later if needed.
    agent_name      TEXT,
    model           TEXT,
    num_steps       INTEGER,
    num_tool_calls  INTEGER,
    -- Reserved for enrichment slices; NULL until then.
    has_error       INTEGER,
    failure_label   TEXT,
    embedding_id    TEXT
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


def init_schema(conn: sqlite3.Connection) -> None:
    """Create the ``traces`` table if it doesn't exist."""
    conn.executescript(SCHEMA_SQL)
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
    ingested_at: str | None = None,
) -> None:
    """Insert (or replace) a single trace row.

    Uses ``INSERT OR REPLACE`` so re-running the ingest pipeline on the same
    source corpus is idempotent. The primary key is the trace's session/trace
    id, which is stable across re-ingestion.
    """
    if ingested_at is None:
        ingested_at = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """
        INSERT OR REPLACE INTO traces (
            id, source_format, raw_blob, atif, ingested_at,
            agent_name, model, num_steps, num_tool_calls
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
