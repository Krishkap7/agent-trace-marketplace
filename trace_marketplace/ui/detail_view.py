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

import streamlit as st

from trace_marketplace.ui.components import render_header, render_step
from trace_marketplace.ui.queries import get_trace


def _back_to_list_link() -> None:
    """Plain link button that clears ``?trace_id`` so the router lands
    back on the list view. Using a button (not an ``st.markdown`` link)
    means the click triggers a rerun in the same Streamlit session
    rather than a hard browser navigation that loses state."""
    if st.button("<- Back to list", key="detail_back"):
        st.query_params.clear()
        st.rerun()


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


def _render_thread(atif: dict[str, Any]) -> None:
    steps = atif.get("steps") or []
    if not steps:
        st.info("ATIF parsed but contains no steps.")
        return
    for idx, step in enumerate(steps):
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


def render(conn: sqlite3.Connection, trace_id: str) -> None:
    """Render the detail view. ``app.py`` calls this when
    ``?trace_id=...`` is present."""
    _back_to_list_link()

    row = get_trace(conn, trace_id)
    if row is None:
        st.error(f"No trace found with id `{trace_id}`.")
        return

    render_header(row)
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
        agent_name = (
            row.get("agent_name") or agent_blob.get("name") or "this agent"
        )
        agent_model = row.get("model") or agent_blob.get("model_name") or "?"
        st.caption(
            f"Step-by-step record of what the **{agent_name}** agent "
            f"(model `{agent_model}`) did at run time. Everything below "
            "is the *original* trace; the analysis above is from our "
            "slice-4 detection pipeline."
        )
        agent_summary_bits: list[str] = []
        if agent_blob.get("version"):
            agent_summary_bits.append(f"version: `{agent_blob['version']}`")
        notes = atif.get("notes")
        if notes:
            agent_summary_bits.append(f"notes: {notes}")
        if agent_summary_bits:
            st.caption(" · ".join(agent_summary_bits))
        _render_thread(atif)

    st.divider()
    _render_raw_blob(row.get("raw_blob") or "")
    if row.get("atif"):
        _render_atif_json(row["atif"])


__all__ = ["render"]
