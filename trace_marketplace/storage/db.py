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
    -- them). NOTE: this is *process* failure (any wall hit). Slice 10
    -- adds the separate ``ultimately_succeeded`` column for outcome.
    has_error                INTEGER,
    -- Slice 4: structural signal JSON (e.g. ``{"loop": true, ...}``)
    -- plus the LLM-judge label + one-sentence reasoning.
    failure_signals          TEXT,
    failure_label            TEXT,
    failure_label_reasoning  TEXT,
    -- Reserved for slice 5 (embeddings + semantic search).
    embedding_id             TEXT,
    -- Slice 10: multi-event failure timeline produced by the Opus 4.7
    -- extractor (``enrich/failure_events.py``). JSON list of
    -- ``FailureEvent`` objects, possibly empty. The single-label
    -- ``failure_label`` column is overwritten from this list (most
    -- severe unrecovered event) once events have been extracted.
    failure_events           TEXT,
    -- Slice 10: pre-computed "recovered recommendations" panel. JSON
    -- list of ``{trace_id, similarity, source_format, agent_name,
    -- recoveries:[...]}``. Populated for ``has_error=1`` traces only.
    recovered_recommendations TEXT,
    -- Slice 10: provenance for ``failure_label``. One of
    -- ``'swe_agent_target'``, ``'sonnet_judge'``,
    -- ``'opus_events_derived'``. Backfilled from existing labels +
    -- updated by the events extractor.
    failure_label_source     TEXT,
    -- Slice 10: outcome signal, complement to the process signal in
    -- ``has_error``. ``1`` = trace ended successfully; ``0`` = trace
    -- did not ultimately succeed; ``NULL`` = unknown. Backfilled from
    -- SWE-agent ``target`` for benchmark rows; set by the events
    -- extractor for everything else.
    ultimately_succeeded     INTEGER
);

CREATE TABLE IF NOT EXISTS trace_embeddings (
    -- Slice 6: one row per trace, holding a single text-embedding-3-small
    -- vector packed as float32 bytes. The vector is keyed off the source
    -- text hash so re-running the backfill is idempotent (we skip rows
    -- whose ``source_text_hash`` already matches) and stale rows can be
    -- detected if text-extraction logic changes later. ``ON DELETE
    -- CASCADE`` keeps the embedding table tidy if a trace is ever
    -- removed (though slice 5's upload path is INSERT-OR-REPLACE, not
    -- delete-then-insert, so the cascade is just a safety net).
    trace_id          TEXT PRIMARY KEY REFERENCES traces(id) ON DELETE CASCADE,
    embedding         BLOB NOT NULL,
    model             TEXT NOT NULL,
    embedded_at       TEXT NOT NULL,
    source_text_hash  TEXT NOT NULL
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


def _migrate_add_embeddings_table(conn: sqlite3.Connection) -> None:
    """Idempotently create the slice 6 ``trace_embeddings`` table.

    ``SCHEMA_SQL`` already includes the ``CREATE TABLE IF NOT EXISTS``
    so on a fresh DB this is a no-op. On an existing DB that pre-dates
    slice 6 we still want the table to appear without forcing a manual
    ``init_schema`` invocation, so the migration is just an explicit
    ``executescript`` of the same DDL. Pulled into its own helper so
    tests can assert that running it twice is safe and so future
    schema tweaks (e.g. an index) have an obvious place to live.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS trace_embeddings (
            trace_id          TEXT PRIMARY KEY REFERENCES traces(id) ON DELETE CASCADE,
            embedding         BLOB NOT NULL,
            model             TEXT NOT NULL,
            embedded_at       TEXT NOT NULL,
            source_text_hash  TEXT NOT NULL
        );
        """
    )
    conn.commit()


def _migrate_add_event_columns(conn: sqlite3.Connection) -> None:
    """Idempotently add the slice 10 columns to an existing DB.

    Adds four columns (``failure_events``, ``recovered_recommendations``,
    ``failure_label_source``, ``ultimately_succeeded``) and backfills the
    two columns that have known values today:

    * ``failure_label_source`` -- ``'swe_agent_target'`` for SWE-agent
      rows that already have a ``failure_label`` (ground truth from
      benchmark ``target``), ``'sonnet_judge'`` for everything else
      with a populated label.
    * ``ultimately_succeeded`` -- read ``target`` out of each SWE-agent
      ``raw_blob`` and store ``int(target)``. We also cross-check
      against ``has_error``: the adapter at ``adapters/swe_agent.py``
      derives ``has_error = int(not target)``, so for any SWE-agent
      row ``ultimately_succeeded == 1 - has_error`` MUST hold. If it
      doesn't, the adapter is broken and we want a loud failure, not
      a silent log line -- raise ``RuntimeError`` so the migration
      aborts before any downstream code runs against bad data.

    The remaining columns (``failure_events``,
    ``recovered_recommendations``) stay NULL; the Opus extractor and
    the recommendations script in slice 10 populate them.
    """
    cur = conn.execute("PRAGMA table_info(traces)")
    existing = {row[1] for row in cur.fetchall()}
    if not existing:
        return
    if "failure_events" not in existing:
        conn.execute("ALTER TABLE traces ADD COLUMN failure_events TEXT")
    if "recovered_recommendations" not in existing:
        conn.execute("ALTER TABLE traces ADD COLUMN recovered_recommendations TEXT")
    if "failure_label_source" not in existing:
        conn.execute("ALTER TABLE traces ADD COLUMN failure_label_source TEXT")
    if "ultimately_succeeded" not in existing:
        conn.execute("ALTER TABLE traces ADD COLUMN ultimately_succeeded INTEGER")
    conn.commit()

    _backfill_failure_label_source(conn)
    _backfill_ultimately_succeeded_swe_agent(conn)


def _backfill_failure_label_source(conn: sqlite3.Connection) -> None:
    """Populate ``failure_label_source`` for rows where it's still NULL.

    Provenance rule:
    * SWE-agent rows with a populated ``failure_label`` -> the label
      came from the benchmark's ``target`` field (slice 2.5).
    * Every other row with a populated ``failure_label`` -> the label
      came from the slice 4 Sonnet judge.
    * Rows with NULL ``failure_label`` stay NULL.

    Idempotent: only touches rows whose ``failure_label_source`` is
    NULL, so re-running ``init_schema`` against a DB that already had
    this migration applied does nothing.
    """
    conn.execute(
        """
        UPDATE traces
           SET failure_label_source = 'swe_agent_target'
         WHERE source_format = 'swe_agent'
           AND failure_label IS NOT NULL
           AND failure_label_source IS NULL
        """
    )
    conn.execute(
        """
        UPDATE traces
           SET failure_label_source = 'sonnet_judge'
         WHERE failure_label IS NOT NULL
           AND failure_label_source IS NULL
        """
    )
    conn.commit()


def _backfill_ultimately_succeeded_swe_agent(conn: sqlite3.Connection) -> None:
    """Populate ``ultimately_succeeded`` for SWE-agent rows from ``target``.

    Per the slice 10 plan we read ``target`` out of the raw_blob rather
    than reusing ``has_error`` directly, so the column is grounded in
    the benchmark's ground truth instead of an intermediate derivation.
    We then assert ``ultimately_succeeded == 1 - has_error`` -- this
    invariant SHOULD hold because ``adapters/swe_agent.py`` line 119-121
    sets ``trajectory.extra["resolved"] = target`` and the pipeline at
    ``ingest/pipeline.py`` line 52-69 sets ``has_error = int(not
    resolved)`` -- so any mismatch indicates the adapter is broken and
    we abort the migration loudly rather than persisting bad data.

    Idempotent: only touches rows whose ``ultimately_succeeded`` is
    still NULL.
    """
    cur = conn.execute(
        """
        SELECT id, raw_blob, has_error
          FROM traces
         WHERE source_format = 'swe_agent'
           AND ultimately_succeeded IS NULL
        """
    )
    pending: list[tuple[int, str]] = []
    for row in cur.fetchall():
        trace_id = row["id"]
        try:
            blob = json.loads(row["raw_blob"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError(
                f"SWE-agent row {trace_id!r} has unparseable raw_blob "
                f"during ultimately_succeeded backfill: {exc}"
            ) from exc
        if not isinstance(blob, dict):
            raise RuntimeError(
                f"SWE-agent row {trace_id!r}: raw_blob is not a JSON object"
            )
        target = blob.get("target")
        if not isinstance(target, bool):
            raise RuntimeError(
                f"SWE-agent row {trace_id!r}: ``target`` is not a bool "
                f"(got {type(target).__name__!r}={target!r})"
            )
        succeeded = int(target)
        has_error = row["has_error"]
        if has_error is not None and succeeded != (1 - int(has_error)):
            raise RuntimeError(
                f"SWE-agent adapter invariant violated for trace {trace_id!r}: "
                f"target={target!r} implies ultimately_succeeded={succeeded}, "
                f"but has_error={has_error!r} implies "
                f"ultimately_succeeded={1 - int(has_error)}. "
                f"Check adapters/swe_agent.py + ingest/pipeline.py for drift."
            )
        pending.append((succeeded, trace_id))
    if not pending:
        return
    conn.executemany(
        "UPDATE traces SET ultimately_succeeded = ? WHERE id = ?",
        pending,
    )
    conn.commit()


def init_schema(conn: sqlite3.Connection) -> None:
    """Create the ``traces`` table if it doesn't exist, then run any
    idempotent column-addition migrations against an existing DB."""
    conn.executescript(SCHEMA_SQL)
    _migrate_add_failure_columns(conn)
    _migrate_add_embeddings_table(conn)
    _migrate_add_event_columns(conn)
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
    "_migrate_add_embeddings_table",
    "_migrate_add_event_columns",
    "_backfill_failure_label_source",
    "_backfill_ultimately_succeeded_swe_agent",
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
