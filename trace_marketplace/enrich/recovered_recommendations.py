"""Pre-compute "Recoveries from similar failures" for failed traces.

Given a trace ``T`` with ``has_error = 1``, this module finds the top-k
traces in the corpus that:

* are SIMILAR to ``T`` (cosine NN against the slice-6 embedding table), and
* RECOVERED within their own session -- i.e. their ``failure_events`` JSON
  contains at least one entry with ``recovered=true``.

The recovered events from each match (label + recovery summary + step)
are returned alongside the similarity score so the detail-view panel
renders concrete actionable advice ("Agent re-read the AttributeError
and tried Read first before calling the unknown method.") rather than
just a list of opaque IDs.

Why pre-compute instead of running live in the detail view?

* Cosine NN over ~1000 embeddings is fast (~5 ms) but the EXISTS-on-
  json_each candidate pool query plus the metadata join are an extra
  ~20 ms; cached results make the detail-view paint instantaneous.
* The recommendation set only changes when (a) new traces are
  ingested, (b) someone re-runs the events extractor, or (c) someone
  re-runs the embeddings backfill. All three are batch operations, so
  re-running this cache after each is cheap.
* Pre-computing as a JSON column means the UI doesn't need to know
  anything about embeddings or json_each -- it just renders the list.

Public API is two functions:

* :func:`compute_recommendations` -- per-trace logic, returns the
  JSON-serialisable result.
* :func:`select_failed_target_ids` -- selection set helper used by the
  batch script.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from trace_marketplace.enrich.embeddings import load_embedding
from trace_marketplace.search.similar import find_similar

log = logging.getLogger(__name__)

DEFAULT_TOP_K: int = 10
"""How many recommendations to surface per failed trace. The detail-view
panel was designed around 10 rows -- enough variety that at least one
hit usually feels relevant, few enough that the panel doesn't dominate
the page."""

SIMILARITY_PRECISION: int = 4
"""Decimal places to keep on the stored similarity score. Four places
(``0.8717``) is comfortably under what cosine similarity actually
resolves; storing the raw float burns ~16 chars per number in JSON for
no UI benefit."""


def _qualifying_candidate_ids(
    conn: sqlite3.Connection, *, exclude_trace_id: str
) -> list[str]:
    """Return ids of traces that recovered at least one wall in-session.

    The UI panel is titled "Recoveries from similar failures": a
    candidate is qualifying iff its ``failure_events`` JSON contains at
    least one event with ``recovered = 1``. SQLite's JSON1 EXISTS +
    ``json_each`` is the cleanest way to express this without loading
    every event blob into Python.

    Excludes the source trace explicitly so a trace can't recommend
    itself.
    """
    cur = conn.execute(
        """
        SELECT id
          FROM traces
         WHERE failure_events IS NOT NULL
           AND id != ?
           AND EXISTS (
                SELECT 1
                  FROM json_each(traces.failure_events)
                 WHERE json_extract(value, '$.recovered') = 1
           )
        """,
        (exclude_trace_id,),
    )
    return [row["id"] for row in cur.fetchall()]


def _hit_metadata(
    conn: sqlite3.Connection, trace_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Batched fetch of source_format / agent_name / failure_events for hits.

    One SELECT for all hits keeps DB chatter linear in the number of
    recommended traces. Returns ``{trace_id: {source_format, agent_name,
    failure_events_json}}`` -- the failure_events string is left as raw
    JSON so the caller decides whether to parse it.
    """
    if not trace_ids:
        return {}
    placeholders = ",".join("?" * len(trace_ids))
    cur = conn.execute(
        f"""
        SELECT id, source_format, agent_name, failure_events
          FROM traces
         WHERE id IN ({placeholders})
        """,
        trace_ids,
    )
    return {
        row["id"]: {
            "source_format": row["source_format"],
            "agent_name": row["agent_name"],
            "failure_events": row["failure_events"],
        }
        for row in cur.fetchall()
    }


def _recoveries_for(events_blob: str | None) -> list[dict[str, Any]]:
    """Filter a trace's ``failure_events`` JSON down to the recovered events.

    Returns a list of compact dicts (``label`` / ``summary`` /
    ``recovery_step``) suitable for direct rendering in the detail-view
    panel. Malformed event payloads are skipped silently -- the panel
    just shows fewer recoveries rather than blowing up.
    """
    if not events_blob:
        return []
    try:
        events = json.loads(events_blob)
    except json.JSONDecodeError:
        return []
    if not isinstance(events, list):
        return []
    out: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if not event.get("recovered"):
            continue
        summary = event.get("recovery_summary")
        if not isinstance(summary, str) or not summary.strip():
            continue
        out.append(
            {
                "label": event.get("label"),
                "summary": summary.strip(),
                "recovery_step": event.get("recovery_step"),
            }
        )
    return out


def compute_recommendations(
    conn: sqlite3.Connection,
    trace_id: str,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Return up to ``top_k`` similar traces that recovered in-session.

    Each item is a JSON-serialisable dict with:

    * ``trace_id``       -- the recommended trace's id
    * ``similarity``     -- cosine sim to the source, rounded to 4 dp
    * ``source_format``  -- e.g. ``claude_code``, ``swe_agent``
    * ``agent_name``     -- ATIF agent.name
    * ``recoveries``     -- list of ``{label, summary, recovery_step}``
                            entries pulled from the match's
                            ``failure_events`` (one per recovered wall)

    Returns ``[]`` when:

    * the source trace has no cached embedding (we skip rather than
      embed on-demand -- this function runs in the batch script, and
      paying for embeddings should happen in the embeddings backfill),
    * no candidate traces qualify (no traces have recovered events yet,
      e.g. fresh DB before the events extractor has run).

    Hits that exist in the embedding table but have no metadata in the
    ``traces`` table are dropped defensively; this shouldn't happen
    given the ``ON DELETE CASCADE`` foreign key but the cost of a
    defensive ``if metadata is None`` guard is negligible.
    """
    source = load_embedding(conn, trace_id)
    if source is None:
        return []

    candidate_ids = _qualifying_candidate_ids(conn, exclude_trace_id=trace_id)
    if not candidate_ids:
        return []

    hits = find_similar(
        conn,
        source.embedding,
        candidate_ids=candidate_ids,
        exclude_ids={trace_id},
        top_k=top_k,
    )
    if not hits:
        return []

    metadata = _hit_metadata(conn, [h.trace_id for h in hits])

    out: list[dict[str, Any]] = []
    for hit in hits:
        meta = metadata.get(hit.trace_id)
        if meta is None:
            continue
        recoveries = _recoveries_for(meta.get("failure_events"))
        if not recoveries:
            # Candidate passed the JSON1 EXISTS filter but had no
            # *summary*-bearing recoveries (e.g. recovered=true with
            # summary=null, which the extractor shouldn't produce but
            # we tolerate). Skip rather than render an empty bullet.
            continue
        out.append(
            {
                "trace_id": hit.trace_id,
                "similarity": round(float(hit.similarity), SIMILARITY_PRECISION),
                "source_format": meta.get("source_format"),
                "agent_name": meta.get("agent_name"),
                "recoveries": recoveries,
            }
        )
    return out


def persist_recommendations_for_trace(
    conn: sqlite3.Connection,
    trace_id: str,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]] | None:
    """Compute + persist recommendations for one freshly-ingested trace.

    Mirrors :func:`trace_marketplace.enrich.failure_rules.persist_signals_for_trace`
    in shape: the upload flow calls this on every validated insert so the
    "Recoveries from similar failures" panel renders straight away on the
    upload result page (and on the detail view it links to) rather than
    waiting for the next batch refresh.

    Scope:

    * Only runs when the trace has ``has_error = 1``. The detail-view
      panel only renders for failed traces, so computing for successes
      would just write empty arrays that no one reads. NULL is also
      skipped (raw-blob uploads, adapter failures, anything where the
      rule pass didn't conclude).
    * Pure-local: cosine NN over the existing embeddings table. No API
      calls, no cost.

    Returns:

    * ``None`` when the trace is out of scope (missing row, or
      ``has_error != 1``). Callers can treat this as "don't render the
      panel" without a separate flag.
    * The computed list (possibly empty) when the trace is in scope. An
      empty list means the corpus has no qualifying candidates yet --
      e.g. a fresh DB before :mod:`scripts.extract_failure_events` has
      run -- and the panel can render a small caption to that effect.

    Does not commit; the caller batches the commit with the other
    upload-flow writes (signals, embedding) so a single transaction
    covers the whole row.
    """
    row = conn.execute(
        "SELECT has_error FROM traces WHERE id = ?", (trace_id,)
    ).fetchone()
    if row is None or row["has_error"] != 1:
        return None
    recommendations = compute_recommendations(conn, trace_id, top_k=top_k)
    conn.execute(
        "UPDATE traces SET recovered_recommendations = ? WHERE id = ?",
        (json.dumps(recommendations), trace_id),
    )
    return recommendations


def select_failed_target_ids(conn: sqlite3.Connection) -> list[str]:
    """Return ids of every trace that needs a recommendations refresh.

    A trace is in scope iff it has ``has_error = 1`` AND a cached
    embedding (no embedding -> no NN query possible). Returns a stable
    sort order so the batch script's progress logs make sense.
    """
    cur = conn.execute(
        """
        SELECT t.id
          FROM traces t
          JOIN trace_embeddings e ON e.trace_id = t.id
         WHERE t.has_error = 1
         ORDER BY t.id ASC
        """
    )
    return [row["id"] for row in cur.fetchall()]


def count_qualifying_candidates(conn: sqlite3.Connection) -> int:
    """Total number of traces that could ever appear as a recommendation.

    Helper for the batch script's pre-flight warning: if this is zero,
    the events extractor hasn't been run yet and every recommendation
    list will be empty -- we want to tell the user that loudly instead
    of silently writing 500 empty arrays.
    """
    cur = conn.execute(
        """
        SELECT COUNT(*) AS n
          FROM traces
         WHERE failure_events IS NOT NULL
           AND EXISTS (
                SELECT 1
                  FROM json_each(traces.failure_events)
                 WHERE json_extract(value, '$.recovered') = 1
           )
        """
    )
    return int(cur.fetchone()[0])


__all__ = [
    "DEFAULT_TOP_K",
    "SIMILARITY_PRECISION",
    "compute_recommendations",
    "count_qualifying_candidates",
    "persist_recommendations_for_trace",
    "select_failed_target_ids",
]
