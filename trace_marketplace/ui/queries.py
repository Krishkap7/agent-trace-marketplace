"""Pure SQL helpers for the Streamlit viewer.

Design rules:

- No Streamlit import. These functions take a ``sqlite3.Connection`` and
  return plain Python data, so they're trivially unit-testable.
- Every WHERE clause uses ``?`` placeholders. The only place a column
  name flows in as a string is :func:`distinct_values`, which guards
  against SQL injection by restricting the argument to an allow-list.
- Caching is the caller's responsibility (Streamlit wraps these with
  ``st.cache_data`` at the call site).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

LIST_COLUMNS: tuple[str, ...] = (
    "id",
    "source_format",
    "agent_name",
    "model",
    "num_steps",
    "num_tool_calls",
    "has_error",
    "failure_label",
    "ingested_at",
    "atif",
)
"""Columns projected into a list-view row.

``atif`` is included so :func:`components.render_directory_row` can
extract a human-readable task title from the first user step (or the
first substantive system step for SWE-agent traces) instead of just
showing the opaque trace_id UUID. Pagination caps the page at 100 rows
so the wire cost stays bounded; SQLite's column projection of TEXT is
in-process so there's no real network overhead.

The slice 9 directory-card renderer doesn't depend on column order
(unlike the old st.dataframe), so order here just controls SELECT
projection."""

DISTINCT_VALUE_COLUMNS: frozenset[str] = frozenset(
    {"source_format", "agent_name", "model", "failure_label"}
)
"""Columns the UI is allowed to ask :func:`distinct_values` about. This
is an explicit allow-list because the column name is interpolated into
the SQL string and we need to be sure it can't come from user input."""

HAS_ERROR_OPTIONS: tuple[str, ...] = ("any", "errors", "successes", "unknown")
"""Canonical, *ordered* list of values for the ``has_error`` 4-way
filter. The order is the order the UI shows them in, so the first
entry (``"any"``) is also the radio's default.

Lives as a tuple, not a frozenset, because ``frozenset`` iteration
order depends on PYTHONHASHSEED -- if the UI did ``list(HAS_ERROR_VALUES)``
the default radio choice could rotate between process restarts.
(Caught by Cursor bugbot on commit 655c824.)
"""

HAS_ERROR_VALUES: frozenset[str] = frozenset(HAS_ERROR_OPTIONS)
"""Membership-check set derived from :data:`HAS_ERROR_OPTIONS`. Used
by :func:`_build_where` for O(1) validation; never iterated."""


@dataclass(frozen=True)
class ListFilters:
    """Filter knobs surfaced in the sidebar.

    Empty multiselects mean "no constraint" (NOT "match nothing"). The
    UI does not currently expose a way to ask for the empty intersection
    -- if you select nothing in a multiselect we treat it as "All".
    """

    source_formats: tuple[str, ...] = ()
    agent_names: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    failure_labels: tuple[str, ...] = ()
    has_error: str = "any"
    """One of ``HAS_ERROR_VALUES``."""
    has_atif: bool | None = None
    """``True`` -> only validated rows; ``False`` -> only NULL atif;
    ``None`` -> no constraint."""
    min_steps: int | None = None
    max_steps: int | None = None
    search: str = ""
    """Case-insensitive substring; matches ``id`` / ``agent_name`` / ``model``."""


def _build_where(filters: ListFilters) -> tuple[str, list[Any]]:
    """Translate :class:`ListFilters` into a ``(where_sql, params)`` pair.

    Returns the bare ``WHERE ...`` clause (no leading ``WHERE``) and the
    parameter tuple matching its ``?`` placeholders. Caller is expected
    to splice the result into a SELECT.
    """
    clauses: list[str] = []
    params: list[Any] = []

    def _in_clause(column: str, values: tuple[str, ...]) -> None:
        if not values:
            return
        placeholders = ",".join("?" for _ in values)
        clauses.append(f"{column} IN ({placeholders})")
        params.extend(values)

    _in_clause("source_format", filters.source_formats)
    _in_clause("agent_name", filters.agent_names)
    _in_clause("model", filters.models)
    _in_clause("failure_label", filters.failure_labels)

    if filters.has_error not in HAS_ERROR_VALUES:
        raise ValueError(
            f"has_error must be one of {sorted(HAS_ERROR_VALUES)}, "
            f"got {filters.has_error!r}"
        )
    if filters.has_error == "errors":
        clauses.append("has_error = 1")
    elif filters.has_error == "successes":
        clauses.append("has_error = 0")
    elif filters.has_error == "unknown":
        clauses.append("has_error IS NULL")

    if filters.has_atif is True:
        clauses.append("atif IS NOT NULL")
    elif filters.has_atif is False:
        clauses.append("atif IS NULL")

    if filters.min_steps is not None:
        clauses.append("num_steps >= ?")
        params.append(int(filters.min_steps))
    if filters.max_steps is not None:
        clauses.append("num_steps <= ?")
        params.append(int(filters.max_steps))

    if filters.search:
        # Case-insensitive substring across id / agent_name / model.
        # SQLite's LIKE is case-insensitive only for ASCII; the columns
        # this targets (ids, agent names, model strings) are all ASCII
        # so this is fine.
        needle = f"%{filters.search}%"
        clauses.append("(id LIKE ? OR agent_name LIKE ? OR model LIKE ?)")
        params.extend([needle, needle, needle])

    where_sql = " AND ".join(clauses) if clauses else "1=1"
    return where_sql, params


def list_traces(
    conn: sqlite3.Connection,
    filters: ListFilters | None = None,
    *,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Return ``(rows, total_count)`` for the filtered, paginated list view.

    ``total_count`` is the full match count (before LIMIT/OFFSET) so the
    caller can render "page 1 of N" without an extra query at the call
    site. Rows are :class:`dict` instances with :data:`LIST_COLUMNS`.
    """
    filters = filters or ListFilters()
    where_sql, params = _build_where(filters)

    count_sql = f"SELECT COUNT(*) AS n FROM traces WHERE {where_sql}"
    cur = conn.execute(count_sql, params)
    total = int(cur.fetchone()[0])

    select_cols = ", ".join(LIST_COLUMNS)
    list_sql = (
        f"SELECT {select_cols} FROM traces WHERE {where_sql} "
        f"ORDER BY ingested_at DESC, id ASC LIMIT ? OFFSET ?"
    )
    cur = conn.execute(list_sql, [*params, int(limit), int(offset)])
    rows = [dict(row) for row in cur.fetchall()]
    return rows, total


def list_traces_by_ids(
    conn: sqlite3.Connection, trace_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Return ``{trace_id: row}`` for each requested id.

    The output is keyed by id so the caller (slice 6 NL search /
    Find Similar) can preserve its own similarity ranking order
    -- SQL's ``IN (...)`` doesn't guarantee row order, and we don't
    want the embedding ranking to get re-sorted by some incidental
    SELECT order. Columns match :data:`LIST_COLUMNS` so the
    rendering code is identical to the standard filter path.
    """
    if not trace_ids:
        return {}
    placeholders = ",".join("?" * len(trace_ids))
    select_cols = ", ".join(LIST_COLUMNS)
    sql = f"SELECT {select_cols} FROM traces WHERE id IN ({placeholders})"
    cur = conn.execute(sql, trace_ids)
    return {row["id"]: dict(row) for row in cur.fetchall()}


def list_trace_ids(
    conn: sqlite3.Connection, filters: ListFilters | None = None
) -> list[str]:
    """Return every trace id matching ``filters`` (no pagination).

    Slice 6 NL-search joins the structured filter result with the
    vector-NN ranking; the ranker only needs the id set, not the
    full row payload. Pulled out as its own helper so list_view
    doesn't have to splice a giant ``limit`` argument into
    :func:`list_traces` to mean "give me everything".

    Callers that care about similarity ranking (the NL-search path)
    must set ``filters.has_atif=True`` -- rows without ATIF have no
    embedding (the backfill skips them), so they can never appear
    in a ranking. We deliberately don't hardcode that constraint
    here so the function stays general-purpose and the SQL only
    contains the clauses :func:`_build_where` emits (no silent
    duplicates, no surprise filtering when a future caller passes
    ``has_atif=False``).
    """
    filters = filters or ListFilters()
    where_sql, params = _build_where(filters)
    sql = f"SELECT id FROM traces WHERE {where_sql} ORDER BY id ASC"
    cur = conn.execute(sql, params)
    return [row["id"] for row in cur.fetchall()]


def get_trace(conn: sqlite3.Connection, trace_id: str) -> dict[str, Any] | None:
    """Return one full trace row (including ``raw_blob`` and ``atif``).

    Returns ``None`` when no row matches. The ``atif`` column is left as
    its raw JSON string here so the caller can decide whether to decode
    it -- the detail view does, while the raw-blob fallback path does
    not have to.
    """
    cur = conn.execute(
        """
        SELECT id, source_format, raw_blob, atif, ingested_at,
               agent_name, model, num_steps, num_tool_calls,
               has_error, failure_signals, failure_label,
               failure_label_reasoning, embedding_id
        FROM traces
        WHERE id = ?
        """,
        (trace_id,),
    )
    row = cur.fetchone()
    return dict(row) if row is not None else None


def distinct_values(conn: sqlite3.Connection, column: str) -> list[str]:
    """Return the sorted, non-NULL distinct values of ``column``.

    ``column`` must be in :data:`DISTINCT_VALUE_COLUMNS`. We can't bind
    a column name as a parameter in SQL, so this is the only place a
    column name is interpolated -- the allow-list keeps it safe.
    """
    if column not in DISTINCT_VALUE_COLUMNS:
        raise ValueError(
            f"distinct_values: column {column!r} not in allow-list "
            f"{sorted(DISTINCT_VALUE_COLUMNS)}"
        )
    cur = conn.execute(
        f"SELECT DISTINCT {column} AS v FROM traces "
        f"WHERE {column} IS NOT NULL ORDER BY v ASC"
    )
    return [row["v"] for row in cur.fetchall()]


def trace_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Return aggregate counts used by the sidebar header.

    Keys: ``total`` (all rows), ``with_atif`` (validated), ``errors``
    (has_error=1), ``successes`` (has_error=0), ``unknown_outcome``
    (has_error IS NULL).
    """
    cur = conn.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN atif IS NOT NULL THEN 1 ELSE 0 END) AS with_atif,
            SUM(CASE WHEN has_error = 1 THEN 1 ELSE 0 END) AS errors,
            SUM(CASE WHEN has_error = 0 THEN 1 ELSE 0 END) AS successes,
            SUM(CASE WHEN has_error IS NULL THEN 1 ELSE 0 END) AS unknown_outcome
        FROM traces
        """
    )
    row = cur.fetchone()
    return {
        "total": int(row["total"] or 0),
        "with_atif": int(row["with_atif"] or 0),
        "errors": int(row["errors"] or 0),
        "successes": int(row["successes"] or 0),
        "unknown_outcome": int(row["unknown_outcome"] or 0),
    }


def bounds(conn: sqlite3.Connection) -> dict[str, int]:
    """Return ``{"min_steps": ..., "max_steps": ...}`` for the slider.

    Falls back to ``(0, 0)`` when the table is empty so the caller can
    still render a slider without special-casing.
    """
    cur = conn.execute("SELECT MIN(num_steps) AS lo, MAX(num_steps) AS hi FROM traces")
    row = cur.fetchone()
    lo = row["lo"] if row["lo"] is not None else 0
    hi = row["hi"] if row["hi"] is not None else 0
    return {"min_steps": int(lo), "max_steps": int(hi)}


__all__ = [
    "DISTINCT_VALUE_COLUMNS",
    "HAS_ERROR_OPTIONS",
    "HAS_ERROR_VALUES",
    "LIST_COLUMNS",
    "ListFilters",
    "bounds",
    "distinct_values",
    "get_trace",
    "list_trace_ids",
    "list_traces",
    "list_traces_by_ids",
    "trace_counts",
]
