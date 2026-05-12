"""Tests for :mod:`trace_marketplace.search.similar`.

Pure numpy -- no SDK mocks needed. We pin five invariants:

1. **Identical vectors get similarity ~1.0.** Sanity that the
   normalisation + dot product is wired correctly.
2. **Orthogonal vectors get similarity ~0.0.** Ditto, in the
   other direction.
3. **``exclude_ids`` removes traces from the result set.** Find
   Similar relies on this to drop the source trace from its own
   list.
4. **``candidate_ids`` restricts the result set to a subset.** NL
   search relies on this to intersect the SQL filter with the
   ranking.
5. **``top_k`` cap is respected and results are sorted descending.**
   Both UI features show "top 10".
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from trace_marketplace.enrich.embeddings import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    store_embedding,
)
from trace_marketplace.search.similar import find_similar
from trace_marketplace.storage.db import init_schema


def _unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    init_schema(c)
    for tid in ("a", "b", "c", "d", "e"):
        c.execute(
            "INSERT INTO traces (id, source_format, raw_blob, ingested_at) "
            "VALUES (?, ?, ?, ?)",
            (tid, "atif", "{}", "2026-01-01T00:00:00+00:00"),
        )
    c.commit()
    return c


def test_find_similar_returns_top_k_sorted_descending(conn):
    # a is "the query", b is most similar, c is somewhat similar,
    # d is orthogonal-ish, e gets filtered out via candidate_ids.
    vec_a = _unit(1)
    vec_b = vec_a + 0.05 * _unit(2)
    vec_c = vec_a + 0.5 * _unit(3)
    vec_d = _unit(4)
    vec_e = _unit(5)
    for tid, vec in [
        ("a", vec_a),
        ("b", vec_b),
        ("c", vec_c),
        ("d", vec_d),
        ("e", vec_e),
    ]:
        store_embedding(
            conn,
            trace_id=tid,
            embedding=vec.astype(np.float32),
            source_text_hash=f"hash-{tid}",
        )

    hits = find_similar(conn, vec_a, exclude_ids=["a"], top_k=3)
    assert [h.trace_id for h in hits] == ["b", "c", "d"]
    # Similarity is monotone descending.
    sims = [h.similarity for h in hits]
    assert sims == sorted(sims, reverse=True)
    # b is closest -> very high similarity.
    assert hits[0].similarity > 0.95


def test_find_similar_identical_vectors_score_one(conn):
    vec = _unit(7)
    store_embedding(
        conn,
        trace_id="a",
        embedding=vec,
        source_text_hash="h",
    )
    hits = find_similar(conn, vec, top_k=1)
    assert len(hits) == 1
    assert hits[0].trace_id == "a"
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-5)


def test_find_similar_orthogonal_vectors_score_near_zero(conn):
    # Two unit vectors built from disjoint coordinate halves are
    # exactly orthogonal -> cosine similarity 0.0.
    vec_a = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vec_b = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vec_a[: EMBEDDING_DIM // 2] = 1.0
    vec_b[EMBEDDING_DIM // 2 :] = 1.0
    vec_a /= np.linalg.norm(vec_a)
    vec_b /= np.linalg.norm(vec_b)
    store_embedding(conn, trace_id="a", embedding=vec_a, source_text_hash="ha")
    store_embedding(conn, trace_id="b", embedding=vec_b, source_text_hash="hb")
    hits = find_similar(conn, vec_a, exclude_ids=["a"], top_k=1)
    assert hits[0].trace_id == "b"
    assert hits[0].similarity == pytest.approx(0.0, abs=1e-5)


def test_find_similar_candidate_ids_restricts_search_set(conn):
    for i, tid in enumerate(("a", "b", "c", "d")):
        store_embedding(
            conn,
            trace_id=tid,
            embedding=_unit(10 + i),
            source_text_hash=f"h{tid}",
        )
    query = _unit(99)
    hits = find_similar(conn, query, candidate_ids={"a", "c"}, top_k=10)
    assert {h.trace_id for h in hits} == {"a", "c"}


def test_find_similar_returns_empty_when_no_embeddings(conn):
    hits = find_similar(conn, _unit(0), top_k=5)
    assert hits == []


def test_find_similar_filters_by_model(conn):
    store_embedding(conn, trace_id="a", embedding=_unit(1), source_text_hash="h")
    store_embedding(
        conn,
        trace_id="b",
        embedding=_unit(2),
        source_text_hash="h2",
        model="ada-002",
    )
    # Default model filter excludes the ada-002 row.
    hits = find_similar(conn, _unit(1), top_k=10)
    assert [h.trace_id for h in hits] == ["a"]
    # Explicit ada-002 only returns that row.
    hits = find_similar(conn, _unit(2), top_k=10, model="ada-002")
    assert [h.trace_id for h in hits] == ["b"]


def test_find_similar_dimension_mismatch_raises():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    init_schema(c)
    c.execute(
        "INSERT INTO traces (id, source_format, raw_blob, ingested_at) "
        "VALUES (?, ?, ?, ?)",
        ("a", "atif", "{}", "2026-01-01T00:00:00+00:00"),
    )
    c.commit()
    store_embedding(c, trace_id="a", embedding=_unit(1), source_text_hash="h")
    bad_query = np.zeros(EMBEDDING_DIM - 1, dtype=np.float32)
    with pytest.raises(ValueError):
        find_similar(c, bad_query, top_k=1, model=EMBEDDING_MODEL)
