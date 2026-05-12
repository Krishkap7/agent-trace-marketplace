"""Glue between the NL parser, the embedder, and the cosine NN.

The Streamlit views call into here so they don't have to know the
order of operations (parse -> filter -> embed intent -> rank). It
also gives us one place to do graceful degradation when API keys
are missing: callers see a typed result object with an ``error``
field instead of an exception.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass

import numpy as np

from trace_marketplace.enrich.embeddings import embed_text, load_embedding
from trace_marketplace.search.nl_query import (
    NLQueryError,
    ParsedQuery,
    parse_nl_query,
)
from trace_marketplace.search.similar import SimilarityHit, find_similar
from trace_marketplace.ui.queries import ListFilters, list_trace_ids

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NLSearchResult:
    """Outcome of one end-to-end NL search call.

    ``error`` is ``None`` on the happy path and a human-readable
    string when the runner short-circuited (missing API key, parser
    failure, no embeddings in DB). The UI surfaces ``error`` as a
    Streamlit ``st.error`` callout instead of crashing.
    """

    parsed: ParsedQuery | None
    hits: list[SimilarityHit]
    candidate_count: int
    error: str | None = None


def _key_missing(name: str) -> str | None:
    if os.environ.get(name):
        return None
    return (
        f"`{name}` is not configured in the Streamlit Cloud secrets. "
        "Search features are disabled until both `OPENAI_API_KEY` and "
        "`ANTHROPIC_API_KEY` are set."
    )


def run_nl_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    top_k: int = 20,
) -> NLSearchResult:
    """Parse + embed + rank in one call.

    Steps:
    1. Bail out early with an ``error`` if either API key is missing
       -- the UI catches this and shows a configuration hint.
    2. Send the query to :func:`parse_nl_query`. The structured
       filters land in a fresh :class:`ListFilters`; we then take
       the intersection with the corpus via :func:`list_trace_ids`.
    3. Embed ``parsed.semantic_intent`` and rank the candidate ids
       by cosine similarity.

    ``top_k`` defaults to 20 so the user sees more context than the
    "10 perfect hits" default from Find Similar -- NL queries are
    fuzzier and benefit from a wider window.
    """
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        msg = _key_missing(key)
        if msg:
            return NLSearchResult(parsed=None, hits=[], candidate_count=0, error=msg)

    try:
        parsed = parse_nl_query(query)
    except NLQueryError as exc:
        return NLSearchResult(
            parsed=None,
            hits=[],
            candidate_count=0,
            error=f"NL parser failed after retry: {exc}",
        )
    except ValueError as exc:
        return NLSearchResult(parsed=None, hits=[], candidate_count=0, error=str(exc))

    filters = ListFilters(
        source_formats=parsed.source_formats,
        agent_names=parsed.agent_names,
        failure_labels=parsed.failure_labels,
        has_error={
            True: "errors",
            False: "successes",
            None: "any",
        }[parsed.has_error],
        has_atif=True,
    )
    candidate_ids = list_trace_ids(conn, filters)
    if not candidate_ids:
        return NLSearchResult(parsed=parsed, hits=[], candidate_count=0, error=None)

    try:
        query_vec = embed_text(parsed.semantic_intent)
    except Exception as exc:  # pragma: no cover -- network errors
        log.exception("OpenAI embedding call failed")
        return NLSearchResult(
            parsed=parsed,
            hits=[],
            candidate_count=len(candidate_ids),
            error=f"OpenAI embedding call failed: {exc}",
        )

    hits = find_similar(conn, query_vec, candidate_ids=candidate_ids, top_k=top_k)
    return NLSearchResult(
        parsed=parsed, hits=hits, candidate_count=len(candidate_ids), error=None
    )


def run_find_similar(
    conn: sqlite3.Connection,
    trace_id: str,
    *,
    top_k: int = 10,
) -> NLSearchResult:
    """Rank all other traces by similarity to ``trace_id``.

    Used by the detail view's "Find similar" button. Returns the
    same :class:`NLSearchResult` shape as :func:`run_nl_search` (with
    ``parsed=None``) so the UI rendering code can be shared.
    """
    record = load_embedding(conn, trace_id)
    if record is None:
        return NLSearchResult(
            parsed=None,
            hits=[],
            candidate_count=0,
            error=(
                "This trace has no embedding yet. Run "
                "`scripts/generate_embeddings.py` to populate it."
            ),
        )
    hits = find_similar(
        conn,
        np.asarray(record.embedding, dtype=np.float32),
        exclude_ids=[trace_id],
        top_k=top_k,
    )
    return NLSearchResult(parsed=None, hits=hits, candidate_count=len(hits), error=None)


__all__ = ["NLSearchResult", "run_find_similar", "run_nl_search"]
