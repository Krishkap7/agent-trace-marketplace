"""Tests for :func:`trace_marketplace.search.runner.embed_and_recommend_similar`.

This is slice 6's "embed the freshly-uploaded trace + tell the user
what's similar" helper. We hold it to four invariants because it
runs on the user-facing upload path:

1. **Missing ``OPENAI_API_KEY`` is non-fatal.** The function must
   return a soft skip (``embedded=False``, ``skipped_reason`` set)
   so the upload UI can keep working when the deploy is missing
   the key.
2. **Embeds + stores + ranks on the happy path.** With the key set
   and a working (mocked) embedder, the trace is persisted into
   ``trace_embeddings`` AND the function returns ranked neighbours.
3. **No ATIF -> graceful skip.** Raw-blob uploads (unknown formats)
   must short-circuit without raising; the trace is already in the
   DB and the UI shouldn't break.
4. **OpenAI failures are caught.** A transient API error returns a
   skip, never a stacktrace.
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
from trace_marketplace.search import runner
from trace_marketplace.storage.db import init_schema


def _insert_trace(
    conn: sqlite3.Connection,
    trace_id: str,
    *,
    atif: str
    | None = '{"schema_version": "ATIF-v1.7", "agent": {"name": "claude-code", "model_name": "sonnet-4"}, "steps": [{"step_id": "s1", "source": "user", "message": "fix the bug"}, {"step_id": "s2", "source": "agent", "message": "ok"}]}',
    label: str | None = None,
    reasoning: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO traces (id, source_format, raw_blob, atif, ingested_at, "
        "failure_label, failure_label_reasoning) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            trace_id,
            "claude_code",
            "{}",
            atif,
            "2026-01-01T00:00:00+00:00",
            label,
            reasoning,
        ),
    )
    conn.commit()


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    init_schema(c)
    yield c
    c.close()


def test_skips_when_openai_key_is_missing(conn, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _insert_trace(conn, "t1")
    result = runner.embed_and_recommend_similar(conn, "t1")
    assert result.embedded is False
    assert result.similar == []
    assert result.skipped_reason and "OPENAI_API_KEY" in result.skipped_reason
    # And no row landed in trace_embeddings.
    count = conn.execute("SELECT COUNT(*) FROM trace_embeddings").fetchone()[0]
    assert count == 0


def test_skips_when_trace_has_no_atif(conn, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _insert_trace(conn, "t1", atif=None)
    result = runner.embed_and_recommend_similar(conn, "t1")
    assert result.embedded is False
    assert result.skipped_reason and "ATIF" in result.skipped_reason


def test_happy_path_embeds_stores_and_returns_neighbours(conn, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Pre-populate two existing embeddings so find_similar has
    # something to rank against.
    _insert_trace(conn, "neighbour-1")
    _insert_trace(conn, "neighbour-2")
    vec_a = np.ones(EMBEDDING_DIM, dtype=np.float32)
    vec_b = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vec_b[0] = 1.0
    store_embedding(
        conn,
        trace_id="neighbour-1",
        embedding=vec_a,
        source_text_hash="h-a",
        model=EMBEDDING_MODEL,
    )
    store_embedding(
        conn,
        trace_id="neighbour-2",
        embedding=vec_b,
        source_text_hash="h-b",
        model=EMBEDDING_MODEL,
    )

    # The new upload. The fake embedder returns a vector very close
    # to neighbour-1 so the ranking has a clear winner.
    _insert_trace(conn, "upload")

    fake_vec = np.ones(EMBEDDING_DIM, dtype=np.float32) * 0.99
    monkeypatch.setattr(runner, "embed_text", lambda _text: fake_vec)

    result = runner.embed_and_recommend_similar(conn, "upload", top_k=2)
    assert result.embedded is True
    assert result.skipped_reason is None
    assert [h.trace_id for h in result.similar] == ["neighbour-1", "neighbour-2"]
    # And the upload landed in trace_embeddings.
    cur = conn.execute(
        "SELECT trace_id FROM trace_embeddings WHERE trace_id = ?", ("upload",)
    )
    assert cur.fetchone() is not None


def test_openai_failure_is_caught_and_surfaced(conn, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _insert_trace(conn, "t1")

    def _explode(_text):
        raise RuntimeError("OpenAI is having a moment")

    monkeypatch.setattr(runner, "embed_text", _explode)
    result = runner.embed_and_recommend_similar(conn, "t1")
    assert result.embedded is False
    assert result.skipped_reason and "OpenAI" in result.skipped_reason
    # No row stored even though we got past the key check.
    cur = conn.execute("SELECT COUNT(*) FROM trace_embeddings").fetchone()[0]
    assert cur == 0


def test_unknown_trace_id_short_circuits(conn, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # No trace inserted -- ID doesn't exist.
    result = runner.embed_and_recommend_similar(conn, "nonexistent")
    assert result.embedded is False
    assert result.skipped_reason
