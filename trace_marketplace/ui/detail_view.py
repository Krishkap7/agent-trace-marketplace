"""Single-trace detail view.

The view is the same shape regardless of source format:

1. **Header strip**  -- IDs, badges, failure indicators
   (:func:`components.render_header`).
2. **Conversation thread** -- one :func:`components.render_step` per
   ATIF step, in order. Reasoning blocks render italic; encrypted-
   reasoning placeholders surface as a small lock badge so the slice
   2.7 fix is visible in the UI.
3. **Raw blob expander**  -- always present; lets us audit the source
   bytes when the rendering looks off.
4. **Normalised ATIF expander** -- present when ``atif`` is non-NULL.

When ``atif IS NULL`` (unknown format or adapter failure) we replace
the thread with a friendly notice and the raw blob takes over as the
primary content.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any
from urllib.parse import quote

import streamlit as st

from trace_marketplace.search.runner import NLSearchResult, run_find_similar
from trace_marketplace.ui.components import (
    render_failure_events_timeline,
    render_header,
    render_recovered_recommendations_panel,
    render_step,
)
from trace_marketplace.ui.queries import get_trace, list_traces_by_ids


_SIMILAR_STATE_KEY: str = "_slice6_similar_for"
"""Holds ``(trace_id, NLSearchResult)`` between Streamlit reruns so a
sidebar tweak or expander toggle doesn't re-run the search. The cache
key includes the source trace id so navigating to a different detail
page automatically invalidates."""


def _decode_atif(blob: str | None) -> dict[str, Any] | None:
    """Decode the ``atif`` column into a Python dict, or return None.

    Failure swallowed deliberately: a malformed JSON in the column is
    a data bug, not a UI bug, and we still want the raw-blob fallback
    to render so the user can see what the bytes actually look like.
    """
    if not blob:
        return None
    try:
        decoded = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


_THREAD_RENDER_SOFT_CAP: int = 130
"""Total steps above which the thread renders in compressed form by
default. Picked empirically: each step ranges from ~2-10 KB of HTML +
markdown, so ~130 steps lands comfortably under Streamlit Cloud's
~3.5 MB websocket message budget while still covering the full thread
for the vast majority of traces (the corpus median is ~30 steps, p95
~200). Larger thresholds risk re-tripping the RangeError that 1000+
step traces hit before we added this gate."""

_THREAD_HEAD_COUNT: int = 30
"""How many leading steps to show in compressed mode. Front-of-trace is
where the user prompt + task framing live -- enough context to know
what the agent was trying to do."""

_THREAD_TAIL_COUNT: int = 100
"""How many trailing steps to show in compressed mode. Tail is where
failures usually manifest, so we keep proportionally more here than
at the head."""

_THREAD_RENDER_ALL_KEY: str = "_detail_render_full_thread_for"
"""Session-state slot holding the trace_id whose full thread the user
explicitly asked to render. We key by trace_id so navigating to a
different detail page silently resets the toggle (no stale "render
all" state leaking across traces)."""


def _render_thread(atif: dict[str, Any], trace_id: str) -> None:
    """Render the conversation thread, compressing for huge traces.

    For traces under :data:`_THREAD_RENDER_SOFT_CAP` steps we render
    everything inline. Above that we default to first
    :data:`_THREAD_HEAD_COUNT` + last :data:`_THREAD_TAIL_COUNT` steps
    plus a placeholder, with an explicit button to opt into the full
    render. This protects against Streamlit Cloud's websocket message
    size limit (the user hit ``RangeError: index out of range`` on a
    1169-step trace whose full render exceeded ~5 MB; the limit is
    ~3.5 MB). The full thread is always still available via the
    "View normalised ATIF" expander at the bottom.
    """
    steps = atif.get("steps") or []
    if not steps:
        st.info("ATIF parsed but contains no steps.")
        return

    total = len(steps)

    # The full-render opt-in is keyed by trace_id so toggling for one
    # trace doesn't carry over to another. Reading session_state
    # defensively because the key may not exist yet.
    render_all = st.session_state.get(_THREAD_RENDER_ALL_KEY) == trace_id

    if total <= _THREAD_RENDER_SOFT_CAP or render_all:
        for idx, step in enumerate(steps):
            if isinstance(step, dict):
                render_step(step, idx)
        return

    head_end = _THREAD_HEAD_COUNT
    tail_start = total - _THREAD_TAIL_COUNT
    omitted = tail_start - head_end
    st.caption(
        f"_Showing first {_THREAD_HEAD_COUNT} + last "
        f"{_THREAD_TAIL_COUNT} of {total} steps. The full trace is "
        "available via the 'View normalised ATIF' expander at the "
        "bottom of the page._"
    )
    if st.button(
        f"Render all {total} steps inline",
        key=f"render_full_thread_{trace_id}",
        help=(
            "Forces the full conversation thread into this page. May "
            "exceed Streamlit's websocket message budget on very long "
            "traces (rule of thumb: traces over ~800 steps); if the "
            "page goes blank with a 'Failed to process Websocket "
            "message' error, reload and stick with the compressed view."
        ),
    ):
        st.session_state[_THREAD_RENDER_ALL_KEY] = trace_id
        st.rerun()

    for idx in range(head_end):
        step = steps[idx]
        if isinstance(step, dict):
            render_step(step, idx)

    st.markdown(
        f"<div class='tm-step-elision'>(... {omitted} middle steps "
        f"omitted to keep the page under Streamlit's message size "
        f"limit ...)</div>",
        unsafe_allow_html=True,
    )

    for idx in range(tail_start, total):
        step = steps[idx]
        if isinstance(step, dict):
            render_step(step, idx)


def _render_raw_blob(raw_blob: str) -> None:
    with st.expander("View raw blob", expanded=False):
        try:
            parsed = json.loads(raw_blob)
            st.json(parsed)
        except json.JSONDecodeError:
            # Likely JSONL or non-JSON bytes -- show as text so the user
            # can still read it line-by-line.
            st.code(raw_blob, language=None)


def _render_atif_json(atif_blob: str) -> None:
    with st.expander("View normalised ATIF", expanded=False):
        try:
            st.json(json.loads(atif_blob))
        except json.JSONDecodeError:
            st.code(atif_blob, language="json")


def _render_similar_panel(conn: sqlite3.Connection, trace_id: str) -> None:
    """Render the slice 6 "Find similar" button + inline ranked results.

    Behaviour:

    * The button lives just under the header so the affordance is
      obvious without scrolling. Clicking stores the result in
      ``st.session_state`` so subsequent reruns (sidebar tweaks,
      expander toggles) don't re-pay the vector NN cost.
    * If the trace has no cached embedding (slice 6's backfill
      script hasn't run for it yet) we show a clear instruction
      instead of a stack trace.
    * When the result is empty we show "no similar traces" rather
      than an empty table.
    """
    if st.button(
        "Find similar traces",
        key=f"similar_btn_{trace_id}",
        help=(
            "Cosine nearest-neighbour over the slice 6 embedding "
            "matrix. Click to load up to 10 traces ranked by "
            "similarity to this one."
        ),
    ):
        st.session_state[_SIMILAR_STATE_KEY] = (
            trace_id,
            run_find_similar(conn, trace_id),
        )

    cached = st.session_state.get(_SIMILAR_STATE_KEY)
    if not isinstance(cached, tuple) or cached[0] != trace_id:
        return
    result: NLSearchResult = cached[1]
    with st.container(border=True):
        st.markdown("**Similar traces** (cosine nearest neighbours)")
        if result.error:
            st.warning(result.error)
            return
        if not result.hits:
            st.info("No other traces are similar enough to surface.")
            return
        rows_by_id = list_traces_by_ids(conn, [h.trace_id for h in result.hits])
        for hit in result.hits:
            row = rows_by_id.get(hit.trace_id)
            if row is None:
                continue
            label = row.get("failure_label") or "—"
            agent = row.get("agent_name") or "?"
            model = row.get("model") or "?"
            steps = row.get("num_steps") or 0
            url = f"?trace_id={quote(str(hit.trace_id), safe='')}"
            st.markdown(
                f"- [{hit.trace_id}]({url}) — similarity "
                f"**{hit.similarity:.3f}** — label `{label}` — "
                f"agent `{agent}` · model `{model}` · {steps} steps"
            )


def render(conn: sqlite3.Connection, trace_id: str) -> None:
    """Render the detail view. ``app.py`` calls this when
    ``?trace_id=...`` is present.

    Navigation back to the list view is handled by the top nav strip
    rendered above this view in :mod:`trace_marketplace.ui.app`; the
    Browse anchor is a relative ``./`` link that lands cleanly without
    a query-params clear-and-rerun dance.
    """
    row = get_trace(conn, trace_id)
    if row is None:
        st.error(f"No trace found with id `{trace_id}`.")
        return

    render_header(row)
    # Slice 10: events timeline + recoveries panel sit between the
    # header and the live similarity search. They render conditionally
    # (timeline only when there are events; recoveries only for failed
    # traces with populated recommendations) so clean traces still see
    # an uncluttered detail page.
    render_failure_events_timeline(row.get("failure_events"))
    render_recovered_recommendations_panel(row)
    _render_similar_panel(conn, trace_id)
    st.divider()

    atif = _decode_atif(row.get("atif"))
    if atif is None:
        st.warning(
            "This trace has no normalised ATIF representation. "
            "Showing the raw blob instead."
        )
    else:
        # Visual signpost between the slice-4 analysis (everything above
        # the divider, written by us / our LLM judge) and the trace
        # content (everything below, written by the agent that produced
        # the trace). Without this, the first system-prompt block reads
        # as if the judge is still talking. The signpost names both
        # actors so there's no ambiguity.
        st.subheader("Agent trajectory")
        agent_blob = atif.get("agent") or {}
        agent_name = row.get("agent_name") or agent_blob.get("name") or "this agent"
        agent_model = row.get("model") or agent_blob.get("model_name") or "?"
        st.markdown(
            f"Step-by-step record of what the **{agent_name}** agent "
            f"(model `{agent_model}`) did at run time. **Everything "
            "below is the *original* trace** -- none of it is written "
            "by our LLM judge."
        )
        # Inline legend for the role badges so the user doesn't read
        # "System" as "our system" or "Sonnet's commentary". This
        # matters most for SWE-agent traces, whose first two steps are
        # System-role harness scaffolding (tool list, then the bug
        # report + instructions) before any agent turn appears.
        st.caption(
            "**Step roles:**  "
            "`System` = harness / tool scaffolding (NOT our judge).  "
            f"`Agent` = the `{agent_model}` model thinking and acting.  "
            "`Tool` / `User` = tool result or human task setup, when present."
        )
        agent_summary_bits: list[str] = []
        if agent_blob.get("version"):
            agent_summary_bits.append(f"version: `{agent_blob['version']}`")
        notes = atif.get("notes")
        if notes:
            agent_summary_bits.append(f"notes: {notes}")
        if agent_summary_bits:
            st.caption(" · ".join(agent_summary_bits))
        _render_thread(atif, trace_id)

    st.divider()
    _render_raw_blob(row.get("raw_blob") or "")
    if row.get("atif"):
        _render_atif_json(row["atif"])


__all__ = ["render"]
