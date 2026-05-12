"""Brute-force cosine-similarity nearest-neighbour search.

This is the single primitive both slice 6 features ride on:

* **NL search**: embed the parsed ``semantic_intent`` -> pass through
  :func:`find_similar` with the SQL-filter result as ``candidate_ids``.
* **Find Similar**: load the current trace's embedding -> pass through
  :func:`find_similar` with that trace excluded from the result set.

At slice-6 scale (~547 traces, 1536 dims) the matrix is ~3 MB; the
dot product takes <5 ms even on a cold cache. We deliberately avoid
hand-rolling an index (FAISS, hnswlib, sqlite-vec) because the
marginal latency win doesn't pay back the deploy complexity at this
corpus size. When the catalogue grows past ~10 K rows we'll swap
this implementation behind the same function signature.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from trace_marketplace.enrich.embeddings import (
    EMBEDDING_MODEL,
    load_all_embeddings,
)


@dataclass(frozen=True)
class SimilarityHit:
    """One ranked result from :func:`find_similar`."""

    trace_id: str
    similarity: float  # cosine similarity in [-1.0, 1.0]


def _normalise(matrix: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation. Cosine similarity == dot product
    once both operands are unit-length, which lets the actual search
    be a single ``mat @ q`` call.

    ``eps`` guards against zero-vector rows (shouldn't happen with
    ``text-embedding-3-small`` but we don't trust the inputs)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return matrix / norms


def find_similar(
    conn: sqlite3.Connection,
    query: np.ndarray,
    *,
    candidate_ids: Iterable[str] | None = None,
    exclude_ids: Iterable[str] | None = None,
    top_k: int = 10,
    model: str = EMBEDDING_MODEL,
) -> list[SimilarityHit]:
    """Return the top ``top_k`` traces ranked by cosine similarity to ``query``.

    Parameters
    ----------
    conn
        Open SQLite connection. We pull every embedding row for
        ``model`` (default: the slice-6 pin) and run the dot product
        in-process.
    query
        ``(EMBEDDING_DIM,) float32`` array. Caller is responsible for
        getting the right dimensionality.
    candidate_ids
        Optional iterable of trace IDs the search must be restricted
        to. Used by the NL path to intersect the SQL filter with the
        semantic ranking. ``None`` == no restriction.
    exclude_ids
        Optional iterable of trace IDs to skip in the result. Used
        by Find Similar to drop the source trace from its own list.
    top_k
        How many hits to return. The function returns *up to*
        ``top_k`` -- if the candidate set is smaller, we return all
        of it. Hits are sorted by descending similarity.
    model
        Embedding model filter. Defaults to
        :data:`~trace_marketplace.enrich.embeddings.EMBEDDING_MODEL`.

    Returns
    -------
    List of :class:`SimilarityHit` sorted by descending similarity.
    Empty list when no candidates exist (corpus not embedded yet, or
    every row got filtered out) -- the UI handles this case with
    "no matches".
    """
    records = load_all_embeddings(conn, model=model)
    if not records:
        return []

    keep = {r.trace_id for r in records}
    if candidate_ids is not None:
        keep &= set(candidate_ids)
    if exclude_ids is not None:
        keep -= set(exclude_ids)
    if not keep:
        return []

    selected = [r for r in records if r.trace_id in keep]
    ids = [r.trace_id for r in selected]
    matrix = np.stack([r.embedding for r in selected]).astype(np.float32)
    matrix = _normalise(matrix)

    q = np.asarray(query, dtype=np.float32)
    if q.shape != (matrix.shape[1],):
        raise ValueError(
            f"Query shape {q.shape} does not match embedding dim ({matrix.shape[1]},)"
        )
    q = q / max(float(np.linalg.norm(q)), 1e-12)

    sims = matrix @ q  # shape (n_candidates,)
    effective_k = min(top_k, len(sims))
    # ``argpartition`` is O(n) and gets the top-k unordered; we then
    # sort the small slice. Faster than a full ``argsort`` for the
    # whole array at any meaningful corpus size.
    if effective_k == len(sims):
        order = np.argsort(-sims)
    else:
        partition_idx = np.argpartition(-sims, effective_k - 1)[:effective_k]
        order = partition_idx[np.argsort(-sims[partition_idx])]
    return [SimilarityHit(trace_id=ids[i], similarity=float(sims[i])) for i in order]


__all__ = [
    "SimilarityHit",
    "find_similar",
]
