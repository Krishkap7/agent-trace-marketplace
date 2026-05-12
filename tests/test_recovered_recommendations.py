"""Unit tests for :mod:`trace_marketplace.enrich.recovered_recommendations`.

Builds a tiny in-memory DB with a mix of failed/recovered/clean traces
plus synthetic embeddings so the cosine-NN ranking has something to
work with, then pins:

* candidate pool only contains traces with at least one recovered event,
* the source trace is always excluded from its own recommendations,
* per-hit recoveries cover every recovered event in the matched trace,
* results are sorted descending by similarity,
* ``select_failed_target_ids`` returns only ``has_error=1`` rows with
  embeddings,
* :func:`compute_recommendations` returns ``[]`` when the source has no
  embedding (cold-start scenario).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pytest

from trace_marketplace.enrich.embeddings import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    store_embedding,
)
from trace_marketplace.enrich.recovered_recommendations import (
    compute_recommendations,
    count_qualifying_candidates,
    persist_recommendations_for_trace,
    select_failed_target_ids,
)
from trace_marketplace.storage.db import connect, init_schema, insert_trace


def _vec(*values: float) -> np.ndarray:
    """Pad a short prefix into a full-dim float32 embedding.

    Concrete values for the first few dims dominate the dot product, so
    we can construct unit-length-ish vectors whose ranking is
    deterministic without hand-engineering all 1536 floats."""
    arr = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    for i, v in enumerate(values):
        arr[i] = v
    return arr


def _events(*events: dict) -> str:
    return json.dumps(list(events))


def _recovered(label: str, step: int, summary: str, *, span: tuple[int, int]) -> dict:
    return {
        "label": label,
        "step_start": span[0],
        "step_end": span[1],
        "recovered": True,
        "recovery_step": step,
        "recovery_summary": summary,
    }


def _unrecovered(label: str, *, span: tuple[int, int]) -> dict:
    return {
        "label": label,
        "step_start": span[0],
        "step_end": span[1],
        "recovered": False,
        "recovery_step": None,
        "recovery_summary": None,
    }


@pytest.fixture()
def seeded_db(tmp_path) -> sqlite3.Connection:
    """In-memory mix:

    * src: the failed source trace -- has_error=1, embedding aligned
      with hit_recovered_A.
    * hit_recovered_A: similar to src, has a recovered event.
    * hit_recovered_B: similar but slightly less so, two recovered
      events.
    * hit_unrecovered: similar but the wall stuck -- must NOT appear.
    * hit_clean: success trace with NULL failure_events -- must NOT
      appear (no recoveries to surface).
    * hit_no_embedding: has a recovered event but no embedding -- the
      ranker can't see it, must NOT appear.
    """
    db_path = tmp_path / "recs.db"
    conn = connect(db_path)
    init_schema(conn)

    def _add(
        trace_id: str,
        *,
        has_error: int | None,
        events: str | None = None,
        emb: np.ndarray | None = None,
        source_format: str = "synth",
        agent_name: str = "test-agent",
    ) -> None:
        insert_trace(
            conn,
            trace_id=trace_id,
            source_format=source_format,
            raw_blob="{}",
            atif='{"schema_version": "ATIF-v1.7"}',
            agent_name=agent_name,
            model="test-model",
            num_steps=10,
            num_tool_calls=5,
            has_error=has_error,
        )
        if events is not None:
            conn.execute(
                "UPDATE traces SET failure_events = ? WHERE id = ?",
                (events, trace_id),
            )
        if emb is not None:
            store_embedding(
                conn,
                trace_id=trace_id,
                embedding=emb,
                source_text_hash=f"hash-{trace_id}",
                model=EMBEDDING_MODEL,
                embedded_at=datetime.now(timezone.utc).isoformat(),
            )
        conn.commit()

    _add(
        "src",
        has_error=1,
        events=_events(_unrecovered("loop", span=(4, 8))),
        emb=_vec(1.0, 0.0, 0.0),
    )
    _add(
        "hit_recovered_A",
        has_error=1,
        events=_events(
            _recovered(
                "loop",
                step=9,
                summary="Caught the AttributeError and used Read before retrying.",
                span=(5, 8),
            )
        ),
        emb=_vec(0.95, 0.05, 0.0),  # closest to src
    )
    _add(
        "hit_recovered_B",
        has_error=1,
        events=_events(
            _recovered(
                "wrong_approach",
                step=12,
                summary="Switched from sed to a Python script.",
                span=(7, 10),
            ),
            _recovered(
                "misread_error",
                step=14,
                summary="Re-read the traceback and noticed the chained cause.",
                span=(13, 13),
            ),
        ),
        emb=_vec(0.80, 0.15, 0.0),  # second closest
    )
    _add(
        "hit_unrecovered",
        has_error=1,
        events=_events(_unrecovered("hallucinated_api", span=(3, 5))),
        emb=_vec(0.99, 0.0, 0.0),  # very close, but no recoveries
    )
    _add("hit_clean", has_error=0, events=None, emb=_vec(0.90, 0.0, 0.0))
    _add(
        "hit_no_embedding",
        has_error=1,
        events=_events(
            _recovered(
                "environment_issue",
                step=4,
                summary="Re-ran with PYTHONPATH set.",
                span=(2, 3),
            )
        ),
        emb=None,
    )

    try:
        yield conn
    finally:
        conn.close()


# ---------- compute_recommendations ----------


def test_compute_recommendations_returns_only_recovered_candidates(
    seeded_db: sqlite3.Connection,
) -> None:
    recs = compute_recommendations(seeded_db, "src", top_k=10)
    ids = [r["trace_id"] for r in recs]
    assert "hit_recovered_A" in ids
    assert "hit_recovered_B" in ids
    # Unrecovered, clean, no-embedding candidates must NOT appear.
    assert "hit_unrecovered" not in ids
    assert "hit_clean" not in ids
    assert "hit_no_embedding" not in ids


def test_compute_recommendations_excludes_source_trace(
    seeded_db: sqlite3.Connection,
) -> None:
    recs = compute_recommendations(seeded_db, "src", top_k=10)
    assert all(r["trace_id"] != "src" for r in recs)


def test_compute_recommendations_sorted_by_similarity_descending(
    seeded_db: sqlite3.Connection,
) -> None:
    recs = compute_recommendations(seeded_db, "src", top_k=10)
    sims = [r["similarity"] for r in recs]
    assert sims == sorted(sims, reverse=True)


def test_compute_recommendations_recoveries_field_includes_each_recovered_event(
    seeded_db: sqlite3.Connection,
) -> None:
    recs = compute_recommendations(seeded_db, "src", top_k=10)
    by_id = {r["trace_id"]: r for r in recs}
    assert len(by_id["hit_recovered_A"]["recoveries"]) == 1
    # hit_recovered_B has TWO recovered events; both should be present.
    assert len(by_id["hit_recovered_B"]["recoveries"]) == 2
    summaries = [r["summary"] for r in by_id["hit_recovered_B"]["recoveries"]]
    assert any("sed" in s for s in summaries)
    assert any("traceback" in s for s in summaries)


def test_compute_recommendations_returns_empty_when_source_has_no_embedding(
    seeded_db: sqlite3.Connection,
) -> None:
    # hit_no_embedding has events but no embedding; querying it as the
    # SOURCE should produce an empty result (not raise).
    assert compute_recommendations(seeded_db, "hit_no_embedding") == []


def test_compute_recommendations_returns_empty_when_no_candidates(
    tmp_path,
) -> None:
    """Cold-start: a DB with a failed trace but no recovered candidates
    anywhere returns an empty list (not None, not error)."""
    conn = connect(tmp_path / "cold.db")
    try:
        init_schema(conn)
        insert_trace(
            conn,
            trace_id="lonely",
            source_format="synth",
            raw_blob="{}",
            atif='{"schema_version": "ATIF-v1.7"}',
            agent_name="a",
            model="m",
            num_steps=1,
            num_tool_calls=0,
            has_error=1,
        )
        store_embedding(
            conn,
            trace_id="lonely",
            embedding=_vec(1.0),
            source_text_hash="h",
            model=EMBEDDING_MODEL,
            embedded_at=datetime.now(timezone.utc).isoformat(),
        )
        assert compute_recommendations(conn, "lonely") == []
    finally:
        conn.close()


def test_compute_recommendations_respects_top_k(seeded_db: sqlite3.Connection) -> None:
    recs = compute_recommendations(seeded_db, "src", top_k=1)
    assert len(recs) == 1
    assert recs[0]["trace_id"] == "hit_recovered_A"


# ---------- select_failed_target_ids ----------


def test_select_failed_target_ids_returns_only_failed_with_embeddings(
    seeded_db: sqlite3.Connection,
) -> None:
    ids = select_failed_target_ids(seeded_db)
    # has_error=1 AND embedded: src + hit_recovered_A + hit_recovered_B
    # + hit_unrecovered. hit_no_embedding is excluded (no embedding);
    # hit_clean is excluded (has_error=0).
    assert set(ids) == {
        "src",
        "hit_recovered_A",
        "hit_recovered_B",
        "hit_unrecovered",
    }


def test_select_failed_target_ids_is_sorted(seeded_db: sqlite3.Connection) -> None:
    ids = select_failed_target_ids(seeded_db)
    assert ids == sorted(ids)


# ---------- count_qualifying_candidates ----------


def test_count_qualifying_candidates_returns_recovered_only(
    seeded_db: sqlite3.Connection,
) -> None:
    # Only hit_recovered_A, hit_recovered_B, hit_no_embedding qualify
    # (recovered=true present). hit_unrecovered is excluded (recovered=false
    # everywhere). src has no recovered events. hit_clean has no events.
    assert count_qualifying_candidates(seeded_db) == 3


# ---------- persist_recommendations_for_trace ----------
#
# Single-trace upload-flow helper. The semantics that matter for the
# upload UX:
#
# * has_error = 1     -> compute + persist + return list (possibly empty)
# * has_error = 0     -> return None, column stays NULL
# * has_error IS NULL -> return None, column stays NULL
# * missing trace_id  -> return None, no crash
#
# When has_error = 1 we always WRITE to the column, even if the result is
# empty -- that way the detail view can distinguish "computed but no
# matches" from "never computed" and rendering stays predictable.


def _read_recs_column(conn: sqlite3.Connection, trace_id: str) -> str | None:
    row = conn.execute(
        "SELECT recovered_recommendations FROM traces WHERE id = ?", (trace_id,)
    ).fetchone()
    return row["recovered_recommendations"] if row else None


def test_persist_recommendations_for_failed_trace_writes_and_returns_list(
    seeded_db: sqlite3.Connection,
) -> None:
    result = persist_recommendations_for_trace(seeded_db, "src")
    assert isinstance(result, list)
    assert len(result) >= 2  # hit_recovered_A + hit_recovered_B at minimum
    ids = {rec["trace_id"] for rec in result}
    assert "hit_recovered_A" in ids
    assert "hit_recovered_B" in ids
    # Verify it was persisted -- the column round-trips through JSON
    # cleanly.
    raw = _read_recs_column(seeded_db, "src")
    assert raw is not None
    persisted = json.loads(raw)
    assert [r["trace_id"] for r in persisted] == [r["trace_id"] for r in result]


def test_persist_recommendations_skips_success_traces(
    seeded_db: sqlite3.Connection,
) -> None:
    # hit_clean has has_error=0; we should NOT compute or write.
    assert persist_recommendations_for_trace(seeded_db, "hit_clean") is None
    assert _read_recs_column(seeded_db, "hit_clean") is None


def test_persist_recommendations_skips_null_has_error(tmp_path) -> None:
    """has_error IS NULL (raw blob / adapter failure) is out of scope."""
    conn = connect(tmp_path / "null_he.db")
    try:
        init_schema(conn)
        insert_trace(
            conn,
            trace_id="null_he",
            source_format="unknown",
            raw_blob="{}",
            atif=None,
            agent_name=None,
            model=None,
            num_steps=None,
            num_tool_calls=None,
            has_error=None,
        )
        conn.commit()
        assert persist_recommendations_for_trace(conn, "null_he") is None
        assert _read_recs_column(conn, "null_he") is None
    finally:
        conn.close()


def test_persist_recommendations_missing_trace_returns_none(
    seeded_db: sqlite3.Connection,
) -> None:
    """Defensive: a non-existent trace_id returns None instead of crashing."""
    assert persist_recommendations_for_trace(seeded_db, "does-not-exist") is None


def test_persist_recommendations_writes_empty_list_when_no_candidates(
    tmp_path,
) -> None:
    """Cold-start: failed trace, no qualifying candidates -> writes ``[]``
    (not NULL) so the detail view can distinguish "computed, found
    nothing" from "never computed"."""
    conn = connect(tmp_path / "cold_persist.db")
    try:
        init_schema(conn)
        insert_trace(
            conn,
            trace_id="lonely_failed",
            source_format="synth",
            raw_blob="{}",
            atif='{"schema_version": "ATIF-v1.7"}',
            agent_name="a",
            model="m",
            num_steps=1,
            num_tool_calls=0,
            has_error=1,
        )
        store_embedding(
            conn,
            trace_id="lonely_failed",
            embedding=_vec(1.0),
            source_text_hash="h",
            model=EMBEDDING_MODEL,
            embedded_at=datetime.now(timezone.utc).isoformat(),
        )
        conn.commit()

        result = persist_recommendations_for_trace(conn, "lonely_failed")
        assert result == []
        # Crucially, the column got written -- "[]", not left as NULL.
        raw = _read_recs_column(conn, "lonely_failed")
        assert raw is not None
        assert json.loads(raw) == []
    finally:
        conn.close()
