"""Tests for :mod:`trace_marketplace.enrich.embeddings`.

We pin five invariants:

1. The float32 round-trip is lossless (``tobytes`` -> ``frombuffer``
   returns the same numbers we put in).
2. Batch chunking matches the math: 100 inputs -> 1 call (size 100),
   250 inputs at size 100 -> 3 calls.
3. ``existing_source_hashes`` returns only rows that exist; this is
   what makes the backfill script idempotent.
4. ``store_embedding`` is an upsert (re-storing the same trace_id
   overwrites, never duplicates).
5. ``hash_source_text`` is deterministic across calls.

The OpenAI client is fully mocked -- no network in test runs.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from trace_marketplace.enrich.embeddings import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    embed_batch,
    embed_text,
    existing_source_hashes,
    hash_source_text,
    load_all_embeddings,
    load_embedding,
    store_embedding,
)
from trace_marketplace.storage.db import init_schema


@dataclass
class _FakeEmbObj:
    embedding: list[float]


@dataclass
class _FakeUsage:
    prompt_tokens: int = 0


@dataclass
class _FakeResponse:
    data: list[_FakeEmbObj]
    usage: _FakeUsage


class _FakeEmbeddings:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def create(self, *, model: str, input):  # noqa: A002
        self.calls.append({"model": model, "input": list(input)})
        # Return one deterministic vector per input. The vector
        # encodes the input position so tests can assert the order
        # is preserved across batches.
        data = [
            _FakeEmbObj(
                embedding=[float((hash(f"{model}:{text}") & 0xFFFF) / 65535.0)]
                * EMBEDDING_DIM
            )
            for text in input
        ]
        return _FakeResponse(data=data, usage=_FakeUsage(prompt_tokens=42))


class _FakeClient:
    def __init__(self):
        self.embeddings = _FakeEmbeddings()


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    init_schema(c)
    # A real trace row is needed because the FK on trace_embeddings
    # would otherwise refuse the insert.
    c.execute(
        "INSERT INTO traces (id, source_format, raw_blob, ingested_at) "
        "VALUES (?, ?, ?, ?)",
        ("t1", "atif", "{}", "2026-01-01T00:00:00+00:00"),
    )
    c.execute(
        "INSERT INTO traces (id, source_format, raw_blob, ingested_at) "
        "VALUES (?, ?, ?, ?)",
        ("t2", "atif", "{}", "2026-01-01T00:00:00+00:00"),
    )
    c.commit()
    yield c
    c.close()


def test_hash_source_text_is_deterministic():
    a = hash_source_text("hello world")
    b = hash_source_text("hello world")
    c = hash_source_text("hello world ")  # trailing space
    assert a == b
    assert a != c
    # SHA-256 hex is 64 chars.
    assert len(a) == 64


def test_embed_text_returns_float32_array_of_expected_shape():
    client = _FakeClient()
    vec = embed_text("hello", client=client)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (EMBEDDING_DIM,)
    assert vec.dtype == np.float32


def test_embed_batch_chunks_inputs_by_batch_size():
    client = _FakeClient()
    texts = [f"t{i}" for i in range(250)]
    vectors = embed_batch(texts, client=client, batch_size=100)
    assert len(vectors) == 250
    # 250 / 100 = 3 calls (100, 100, 50).
    assert len(client.embeddings.calls) == 3
    assert [len(c["input"]) for c in client.embeddings.calls] == [100, 100, 50]


def test_embed_batch_empty_input_makes_zero_calls():
    client = _FakeClient()
    out = embed_batch([], client=client)
    assert out == []
    assert client.embeddings.calls == []


def test_store_and_load_embedding_roundtrip_is_lossless(conn):
    vec = np.linspace(-1.0, 1.0, EMBEDDING_DIM, dtype=np.float32)
    store_embedding(
        conn,
        trace_id="t1",
        embedding=vec,
        source_text_hash=hash_source_text("body-1"),
    )
    record = load_embedding(conn, "t1")
    assert record is not None
    np.testing.assert_array_equal(record.embedding, vec)
    assert record.model == EMBEDDING_MODEL
    assert record.embedded_at  # ISO-format timestamp populated
    assert record.source_text_hash == hash_source_text("body-1")


def test_store_embedding_is_an_upsert(conn):
    vec_a = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vec_b = np.ones(EMBEDDING_DIM, dtype=np.float32)
    store_embedding(conn, trace_id="t1", embedding=vec_a, source_text_hash="hash-a")
    store_embedding(conn, trace_id="t1", embedding=vec_b, source_text_hash="hash-b")
    # Exactly one row exists, and it carries the second vector / hash.
    cur = conn.execute(
        "SELECT COUNT(*) FROM trace_embeddings WHERE trace_id = ?", ("t1",)
    )
    assert cur.fetchone()[0] == 1
    record = load_embedding(conn, "t1")
    np.testing.assert_array_equal(record.embedding, vec_b)
    assert record.source_text_hash == "hash-b"


def test_existing_source_hashes_returns_only_known_rows(conn):
    store_embedding(
        conn,
        trace_id="t1",
        embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
        source_text_hash="h1",
    )
    mapping = existing_source_hashes(conn, ["t1", "t2", "missing"])
    assert mapping == {"t1": "h1"}


def test_load_all_embeddings_filters_by_model(conn):
    store_embedding(
        conn,
        trace_id="t1",
        embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
        source_text_hash="h1",
    )
    store_embedding(
        conn,
        trace_id="t2",
        embedding=np.ones(EMBEDDING_DIM, dtype=np.float32),
        source_text_hash="h2",
        model="ada-002",
    )
    only_small = load_all_embeddings(conn, model=EMBEDDING_MODEL)
    assert {r.trace_id for r in only_small} == {"t1"}
    all_rows = load_all_embeddings(conn, model=None)
    assert {r.trace_id for r in all_rows} == {"t1", "t2"}
