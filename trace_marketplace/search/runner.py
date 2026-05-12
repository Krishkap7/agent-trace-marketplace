"""Glue between the NL parser, the embedder, and the cosine NN.

The Streamlit views call into here so they don't have to know the
order of operations (parse -> filter -> embed intent -> rank). It
also gives us one place to do graceful degradation when API keys
are missing: callers see a typed result object with an ``error``
field instead of an exception.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from trace_marketplace.enrich.embeddings import (
    embed_text,
    hash_source_text,
    load_embedding,
    store_embedding,
)
from trace_marketplace.search.nl_query import (
    NLQueryError,
    ParsedQuery,
    parse_nl_query,
)
from trace_marketplace.search.similar import SimilarityHit, find_similar
from trace_marketplace.search.text_extraction import extract_searchable_text
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


@dataclass(frozen=True)
class EmbedAndSimilarResult:
    """Outcome of :func:`embed_and_recommend_similar`.

    Used by the slice 5 upload flow to surface "here are some related
    traces you should check out" immediately after ingest. Three
    invariants worth knowing:

    * ``embedded`` is True iff the OpenAI call succeeded AND the row
      was persisted to ``trace_embeddings``. Subsequent searches will
      include this trace.
    * ``similar`` is empty when the search corpus is still empty
      (slice 6's backfill hasn't run on this DB yet) OR when no
      vectors clear the cosine threshold. We never return ``similar``
      hits without having embedded the source trace first.
    * ``skipped_reason`` is set when we *deliberately* short-circuited
      (missing API key, unparseable ATIF). It's UI copy, not a
      stacktrace -- shown verbatim under the upload result.
    """

    embedded: bool
    similar: list[SimilarityHit] = field(default_factory=list)
    skipped_reason: str | None = None


def embed_and_recommend_similar(
    conn: sqlite3.Connection,
    trace_id: str,
    *,
    top_k: int = 5,
) -> EmbedAndSimilarResult:
    """Embed a freshly-ingested trace + return up to ``top_k`` similar traces.

    Called by :mod:`trace_marketplace.ui.upload_view` right after the
    rule-based pass. Designed to *never* raise: every failure mode is
    captured as a ``skipped_reason`` string the upload UI renders as a
    small caption. This matches the rule-pass behaviour ("trace is
    still ingested even if the enrichment step fails") and means a
    transient OpenAI outage can't tank uploads.

    The function does its own cost gate: if ``OPENAI_API_KEY`` is
    unset we exit before any API call. On Streamlit Cloud the key
    lives in Secrets; locally it comes from the shell env that
    launched ``streamlit run``.

    Parameters
    ----------
    conn
        Open SQLite connection. Reads ``atif`` + ``failure_label`` +
        ``failure_label_reasoning`` for the trace; writes the embedding
        + hash into ``trace_embeddings``.
    trace_id
        Primary key of the row just ingested.
    top_k
        How many neighbours to return for the upload-page panel.
        Five is enough to give the user three or four interesting
        clicks without overflowing the result card.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return EmbedAndSimilarResult(
            embedded=False,
            skipped_reason=(
                "OPENAI_API_KEY not configured -- similar-trace lookup "
                "is skipped on upload. Run "
                "`scripts/generate_embeddings.py` later, or set the key "
                "in the Streamlit Cloud secrets, to enable this."
            ),
        )

    row = conn.execute(
        "SELECT atif, failure_label, failure_label_reasoning FROM traces WHERE id = ?",
        (trace_id,),
    ).fetchone()
    if row is None or not row["atif"]:
        return EmbedAndSimilarResult(
            embedded=False,
            skipped_reason=(
                "No normalised ATIF on this trace, so there's nothing "
                "to embed. Similar-trace lookup is available for ATIF, "
                "Claude Code, Codex, Cursor, and SWE-agent formats."
            ),
        )

    try:
        atif = json.loads(row["atif"])
    except json.JSONDecodeError as exc:
        log.warning(
            "trace %s: skipping embed, ATIF column is malformed JSON (%s)",
            trace_id,
            exc,
        )
        return EmbedAndSimilarResult(
            embedded=False,
            skipped_reason=f"ATIF blob is not valid JSON ({exc}).",
        )

    text = extract_searchable_text(
        atif,
        failure_label=row["failure_label"],
        failure_label_reasoning=row["failure_label_reasoning"],
    )

    try:
        vector = embed_text(text)
    except Exception as exc:  # network / auth / quota -- treat as soft
        log.exception("OpenAI embed_text failed for %s", trace_id)
        return EmbedAndSimilarResult(
            embedded=False,
            skipped_reason=f"OpenAI embedding call failed: {exc}",
        )

    try:
        store_embedding(
            conn,
            trace_id=trace_id,
            embedding=vector,
            source_text_hash=hash_source_text(text),
        )
    except Exception as exc:  # pragma: no cover -- sqlite failures are rare
        log.exception("store_embedding failed for %s", trace_id)
        return EmbedAndSimilarResult(
            embedded=False,
            skipped_reason=f"Embedded OK but DB write failed: {exc}",
        )

    try:
        hits = find_similar(conn, vector, exclude_ids=[trace_id], top_k=top_k)
    except Exception as exc:  # pragma: no cover -- defensive
        # ``find_similar`` raises ValueError on dimension mismatch and
        # can also surface DB / numpy errors from ``load_all_embeddings``
        # (the corpus query) or ``np.stack`` (empty / oddly-shaped
        # cache). The function docstring promises "never raises" and
        # ``_process_upload`` in upload_view.py relies on that contract
        # to skip a try/except at the call site -- if we let this
        # propagate, the upload page crashes AFTER the trace was
        # already successfully ingested and embedded. Capture it as a
        # skipped_reason instead so the UI can show "embedded OK but
        # couldn't fetch similar traces because X".
        log.exception("find_similar failed for newly-embedded %s", trace_id)
        return EmbedAndSimilarResult(
            embedded=True,
            similar=[],
            skipped_reason=(
                f"Embedded OK, but similar-trace lookup failed: {exc}"
            ),
        )
    return EmbedAndSimilarResult(embedded=True, similar=hits, skipped_reason=None)


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
    try:
        hits = find_similar(
            conn,
            np.asarray(record.embedding, dtype=np.float32),
            exclude_ids=[trace_id],
            top_k=top_k,
        )
    except Exception as exc:  # pragma: no cover -- defensive
        # Mirror the contract on ``embed_and_recommend_similar``: a
        # corrupt or dimension-mismatched cached vector shouldn't take
        # down the detail panel's "Find similar" button. Surface the
        # failure as a user-readable error string so the panel can
        # render an inline message instead of a red traceback.
        log.exception("find_similar failed for detail-view %s", trace_id)
        return NLSearchResult(
            parsed=None,
            hits=[],
            candidate_count=0,
            error=f"Similar-trace lookup failed: {exc}",
        )
    return NLSearchResult(parsed=None, hits=hits, candidate_count=len(hits), error=None)


__all__ = [
    "EmbedAndSimilarResult",
    "NLSearchResult",
    "embed_and_recommend_similar",
    "run_find_similar",
    "run_nl_search",
]
