"""OpenAI ``text-embedding-3-small`` integration.

Slice 6's write-side enrichment module. It owns three jobs:

1. **Calling the embedding API.** :func:`embed_text` for one text,
   :func:`embed_batch` for the bulk path (the OpenAI endpoint
   accepts up to ~2048 inputs per call; we batch to 100 to keep
   per-request latency predictable and to make retries cheap).
2. **Persistence.** :func:`store_embedding` writes the vector as
   ``float32`` bytes plus a ``source_text_hash`` column so the
   backfill script can skip rows whose underlying text hasn't
   changed.
3. **Loading.** :func:`load_embedding` for the "Find similar"
   path (one trace's vector) and :func:`load_all_embeddings` for
   the NL-query path (the whole matrix at search time).

The module sits in ``enrich/`` rather than ``search/`` because
embedding generation is conceptually a write-side enrichment pass
in the same family as ``failure_rules`` / ``failure_judge``.
``search/`` only consumes embeddings.

Cost / behaviour notes that callers care about:

* ``text-embedding-3-small`` returns 1536 floats per input, billed
  at $0.02 / MTok input. Our ~547 traces at ~1 K tokens each cost
  roughly $0.011 to backfill -- well inside the $1 slice budget.
* The vector is stored as ``np.float32.tobytes()`` (6144 B/row)
  rather than JSON. Saves ~3x on disk and parses 100x faster on
  read. Re-hydrate via ``np.frombuffer(buf, dtype=np.float32)``.
* ``source_text_hash`` is the SHA-256 of the embed-input string.
  The backfill skips rows whose hash already matches, so re-running
  the script is a free no-op.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Protocol, Sequence

import numpy as np

log = logging.getLogger(__name__)

EMBEDDING_MODEL: str = "text-embedding-3-small"
"""Pinned model slug. 1536 dims, $0.02 / MTok. Chosen over
``text-embedding-3-large`` (3072 dims, $0.13 / MTok) because the
~7x cost delta isn't justified at our corpus size; the small model
clusters loop/abandonment/hallucination buckets cleanly."""

EMBEDDING_DIM: int = 1536
"""Dimensionality of ``text-embedding-3-small`` output. Pinned as
a constant so loaders can assert the shape on read; a future model
change has to update both the model slug AND this constant."""

EMBEDDING_PRICE_PER_MTOK: float = 0.02
"""Pricing for ``text-embedding-3-small`` in USD per million tokens.
Source of truth for the dry-run estimator in
``scripts/generate_embeddings.py``; update if OpenAI changes their
sheet."""

DEFAULT_BATCH_SIZE: int = 100
"""Per-request batch size. The endpoint's hard cap is much higher
(~2048) but smaller batches mean smaller blast radius on a transient
error and bounded per-request latency."""


@dataclass(frozen=True)
class EmbeddingRecord:
    """Row representation when loading from the DB."""

    trace_id: str
    embedding: np.ndarray  # shape (EMBEDDING_DIM,), dtype float32
    model: str
    embedded_at: str
    source_text_hash: str


# ---------------------------------------------------------------------------
# OpenAI client protocol -- kept minimal so tests inject fakes without the SDK.
# ---------------------------------------------------------------------------


class _Embeddings(Protocol):
    def create(self, *, model: str, input: Any) -> Any: ...  # noqa: A002


class _OpenAILike(Protocol):
    embeddings: _Embeddings


def _make_client() -> _OpenAILike:
    """Instantiate the SDK lazily.

    Kept inside a helper rather than at module import so importing
    ``embeddings`` for tests / type-checks doesn't require the
    package to be installed at module-load time.
    """
    from openai import OpenAI

    return OpenAI()


# ---------------------------------------------------------------------------
# Hashing -- the staleness key for idempotent backfills.
# ---------------------------------------------------------------------------


def hash_source_text(text: str) -> str:
    """Return the SHA-256 hex digest of ``text``.

    Used as ``trace_embeddings.source_text_hash``. SHA-256 (rather
    than the cheaper MD5) is overkill for collision avoidance here
    but it's a single hash per trace; the marginal CPU cost vs the
    OpenAI round-trip is rounding error.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# API calls -- single + batch.
# ---------------------------------------------------------------------------


def _extract_vectors(response: Any) -> list[list[float]]:
    """Pull the ``embedding`` arrays out of an SDK response.

    The OpenAI SDK returns a Pydantic-like object with a ``data``
    list of ``Embedding`` objects, each exposing a ``.embedding``
    list. We also accept plain dicts so test fakes don't need to
    import the SDK.
    """
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")
    if not data:
        raise RuntimeError("OpenAI response missing `data`")
    vectors: list[list[float]] = []
    for item in data:
        emb = getattr(item, "embedding", None)
        if emb is None and isinstance(item, dict):
            emb = item.get("embedding")
        if not isinstance(emb, (list, tuple)) or not emb:
            raise RuntimeError("OpenAI response item missing `embedding`")
        vectors.append([float(x) for x in emb])
    return vectors


def _extract_usage(response: Any) -> int:
    """Return the input-token count reported by OpenAI for cost tracking.

    Defaults to 0 when the field isn't present (older SDK versions
    or test fakes); the caller only uses this for logging."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get("prompt_tokens") or usage.get("total_tokens") or 0)
    return int(
        getattr(usage, "prompt_tokens", None) or getattr(usage, "total_tokens", 0) or 0
    )


def embed_text(
    text: str,
    *,
    client: _OpenAILike | None = None,
    model: str = EMBEDDING_MODEL,
) -> np.ndarray:
    """Embed a single text. Returns a ``(EMBEDDING_DIM,) float32`` array.

    Convenience wrapper around :func:`embed_batch` for the
    one-shot use case (per-query NL search; per-click Find Similar
    when the trace doesn't have a cached embedding yet)."""
    vectors = embed_batch([text], client=client, model=model)
    return vectors[0]


def embed_batch(
    texts: Sequence[str],
    *,
    client: _OpenAILike | None = None,
    model: str = EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[np.ndarray]:
    """Embed many texts and return one ``float32`` vector per input.

    Chunks into ``batch_size`` requests so the endpoint's per-call
    payload stays bounded. The output order matches the input order
    so callers can zip back to trace_ids without bookkeeping.

    Raises ``RuntimeError`` if any response is malformed -- the
    backfill script wraps individual chunks in try/except so one
    bad batch doesn't abort the whole pass.
    """
    if not texts:
        return []
    if client is None:
        client = _make_client()

    out: list[np.ndarray] = []
    total_tokens = 0
    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start : start + batch_size])
        response = client.embeddings.create(model=model, input=chunk)
        vectors = _extract_vectors(response)
        if len(vectors) != len(chunk):
            raise RuntimeError(
                f"OpenAI returned {len(vectors)} vectors for {len(chunk)} inputs"
            )
        for vec in vectors:
            arr = np.asarray(vec, dtype=np.float32)
            if arr.shape != (EMBEDDING_DIM,):
                raise RuntimeError(
                    f"Expected ({EMBEDDING_DIM},) embedding; got {arr.shape}"
                )
            out.append(arr)
        total_tokens += _extract_usage(response)
    log.info(
        "embed_batch: %d texts -> %d vectors (%d tokens, ~$%.4f)",
        len(texts),
        len(out),
        total_tokens,
        estimate_cost_usd(total_tokens),
    )
    return out


def estimate_cost_usd(tokens: int) -> float:
    """Return the dollar cost of embedding ``tokens`` input tokens."""
    return tokens / 1_000_000 * EMBEDDING_PRICE_PER_MTOK


# ---------------------------------------------------------------------------
# Persistence helpers.
# ---------------------------------------------------------------------------


def store_embedding(
    conn: sqlite3.Connection,
    *,
    trace_id: str,
    embedding: np.ndarray,
    source_text_hash: str,
    model: str = EMBEDDING_MODEL,
    embedded_at: str | None = None,
) -> None:
    """Upsert a single embedding row.

    Uses ``INSERT OR REPLACE`` so the staleness flow ("text
    changed, re-embed") works without an explicit DELETE. The
    primary key is ``trace_id``, matching the schema in
    ``storage/db.py``.
    """
    if embedded_at is None:
        embedded_at = datetime.now(timezone.utc).isoformat()
    if not isinstance(embedding, np.ndarray):
        embedding = np.asarray(embedding, dtype=np.float32)
    if embedding.dtype != np.float32:
        embedding = embedding.astype(np.float32, copy=False)
    if embedding.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"Expected ({EMBEDDING_DIM},) embedding; got {embedding.shape}"
        )
    conn.execute(
        """
        INSERT OR REPLACE INTO trace_embeddings (
            trace_id, embedding, model, embedded_at, source_text_hash
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            trace_id,
            embedding.tobytes(),
            model,
            embedded_at,
            source_text_hash,
        ),
    )
    conn.commit()


def load_embedding(conn: sqlite3.Connection, trace_id: str) -> EmbeddingRecord | None:
    """Return the embedding row for ``trace_id`` or ``None``."""
    row = conn.execute(
        """
        SELECT trace_id, embedding, model, embedded_at, source_text_hash
        FROM trace_embeddings
        WHERE trace_id = ?
        """,
        (trace_id,),
    ).fetchone()
    if row is None:
        return None
    return EmbeddingRecord(
        trace_id=row["trace_id"] if isinstance(row, sqlite3.Row) else row[0],
        embedding=np.frombuffer(
            row["embedding"] if isinstance(row, sqlite3.Row) else row[1],
            dtype=np.float32,
        ).copy(),  # copy so callers can mutate / normalise in-place
        model=row["model"] if isinstance(row, sqlite3.Row) else row[2],
        embedded_at=(row["embedded_at"] if isinstance(row, sqlite3.Row) else row[3]),
        source_text_hash=(
            row["source_text_hash"] if isinstance(row, sqlite3.Row) else row[4]
        ),
    )


def load_all_embeddings(
    conn: sqlite3.Connection,
    *,
    model: str | None = EMBEDDING_MODEL,
) -> list[EmbeddingRecord]:
    """Return every embedding row, optionally filtered by ``model``.

    ``model=None`` returns everything; the default filters to rows
    embedded with the currently-pinned model so a future swap can't
    silently mix shapes. Used by ``search/similar.py`` to build the
    NN matrix.
    """
    if model is None:
        cur = conn.execute(
            "SELECT trace_id, embedding, model, embedded_at, source_text_hash "
            "FROM trace_embeddings"
        )
    else:
        cur = conn.execute(
            "SELECT trace_id, embedding, model, embedded_at, source_text_hash "
            "FROM trace_embeddings WHERE model = ?",
            (model,),
        )
    out: list[EmbeddingRecord] = []
    for row in cur.fetchall():
        out.append(
            EmbeddingRecord(
                trace_id=row["trace_id"],
                embedding=np.frombuffer(row["embedding"], dtype=np.float32).copy(),
                model=row["model"],
                embedded_at=row["embedded_at"],
                source_text_hash=row["source_text_hash"],
            )
        )
    return out


def existing_source_hashes(
    conn: sqlite3.Connection,
    trace_ids: Iterable[str],
    *,
    model: str = EMBEDDING_MODEL,
) -> dict[str, str]:
    """Return ``{trace_id: source_text_hash}`` for already-embedded rows.

    The backfill script joins this map against its candidate list
    to skip work whose underlying text hasn't changed. Bounded by
    the number of ``trace_ids`` passed in so a 100 K-row corpus
    doesn't load the whole table on every script invocation.
    """
    ids = list(trace_ids)
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    cur = conn.execute(
        f"""
        SELECT trace_id, source_text_hash
        FROM trace_embeddings
        WHERE model = ?
          AND trace_id IN ({placeholders})
        """,
        [model, *ids],
    )
    return {row["trace_id"]: row["source_text_hash"] for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# Env guard for callers that need the SDK at runtime.
# ---------------------------------------------------------------------------


def require_api_key() -> None:
    """Refuse to start when ``OPENAI_API_KEY`` is unset.

    Mirrors :func:`failure_judge.require_api_key` so the two enrichment
    scripts have a uniform "give the user a clear instruction" failure
    mode rather than dropping a stacktrace from inside the SDK.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export it before running this script:\n"
            "    export OPENAI_API_KEY=sk-..."
        )


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "EMBEDDING_PRICE_PER_MTOK",
    "EmbeddingRecord",
    "embed_batch",
    "embed_text",
    "estimate_cost_usd",
    "existing_source_hashes",
    "hash_source_text",
    "load_all_embeddings",
    "load_embedding",
    "require_api_key",
    "store_embedding",
]
