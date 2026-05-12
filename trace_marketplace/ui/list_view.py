"""Filterable + paginated list view.

Composition only -- pulls counts/options/rows from :mod:`queries` and
hands the dataframe selection event back to ``app.py`` via
``st.query_params`` so detail-view links are bookmarkable.

The 100-row page cap is the prompt's acceptance criterion; pagination is
manual (LIMIT/OFFSET driven by ``st.number_input("Page")``) because
``st.dataframe`` has no native pager.
"""

from __future__ import annotations

import math
import sqlite3
from typing import Any

import streamlit as st

from trace_marketplace.ui.components import render_filters_summary
from trace_marketplace.ui.queries import (
    HAS_ERROR_OPTIONS,
    LIST_COLUMNS,
    ListFilters,
    bounds,
    distinct_values,
    list_traces,
    trace_counts,
)

PAGE_SIZE: int = 100
"""Acceptance-criteria-mandated cap on rows per page."""

_HAS_ERROR_LABELS: dict[str, str] = {
    "any": "All",
    "errors": "Errors only (has_error=1)",
    "successes": "Successes only (has_error=0)",
    "unknown": "Unknown only (has_error IS NULL)",
}


def _has_atif_label(value: str) -> bool | None:
    return {"All": None, "Has ATIF": True, "Raw blob only": False}[value]


def _sidebar_filters(conn: sqlite3.Connection) -> ListFilters:
    """Render the sidebar controls and return the user's selections.

    Distinct-value lists are queried once per render. They're cheap (one
    DISTINCT scan apiece) and a stale cache would mean the user can't
    pick a freshly-ingested source format until they refresh, which is
    worse than the cost.
    """
    st.sidebar.header("Filters")
    counts = trace_counts(conn)
    # render_filters_summary uses bare ``st.markdown`` (so callers can
    # render it anywhere they like), but we want it pinned to the
    # sidebar -- wrap it in the sidebar context so the call lands
    # alongside the filter widgets, not in the main body.
    with st.sidebar:
        render_filters_summary(counts)
    st.sidebar.divider()

    source_formats = tuple(
        st.sidebar.multiselect(
            "source_format",
            options=distinct_values(conn, "source_format"),
            default=[],
        )
    )
    agent_names = tuple(
        st.sidebar.multiselect(
            "agent_name",
            options=distinct_values(conn, "agent_name"),
            default=[],
        )
    )
    models = tuple(
        st.sidebar.multiselect(
            "model",
            options=distinct_values(conn, "model"),
            default=[],
        )
    )
    failure_labels = tuple(
        st.sidebar.multiselect(
            "failure_label  (populated by slice 4)",
            options=distinct_values(conn, "failure_label"),
            default=[],
        )
    )

    has_error_choice = st.sidebar.radio(
        "has_error",
        # Use the ordered tuple, not the frozenset: list(frozenset) is
        # hash-seed-dependent so index=0 would otherwise be flaky.
        options=HAS_ERROR_OPTIONS,
        index=0,
        format_func=lambda v: _HAS_ERROR_LABELS.get(v, v),
    )

    has_atif_choice = st.sidebar.radio(
        "ATIF column",
        options=("All", "Has ATIF", "Raw blob only"),
        index=0,
    )

    step_bounds = bounds(conn)
    lo = step_bounds["min_steps"]
    hi = max(step_bounds["max_steps"], lo + 1)
    selected_lo, selected_hi = st.sidebar.slider(
        "num_steps range",
        min_value=int(lo),
        max_value=int(hi),
        value=(int(lo), int(hi)),
    )

    search = st.sidebar.text_input(
        "Search (id / agent_name / model)",
        value="",
        placeholder="case-insensitive substring",
    )

    return ListFilters(
        source_formats=source_formats,
        agent_names=agent_names,
        models=models,
        failure_labels=failure_labels,
        has_error=has_error_choice,
        has_atif=_has_atif_label(has_atif_choice),
        min_steps=int(selected_lo),
        max_steps=int(selected_hi),
        search=search.strip(),
    )


def _on_row_select(
    rows: list[dict[str, Any]],
    selection_rows: list[int],
) -> None:
    """When the user single-clicks a row, push the trace id into the
    query params and rerun so ``app.py`` routes to the detail view."""
    if not selection_rows:
        return
    idx = selection_rows[0]
    if 0 <= idx < len(rows):
        st.query_params["trace_id"] = rows[idx]["id"]
        st.rerun()


def render(conn: sqlite3.Connection) -> None:
    """Render the list view. ``app.py`` calls this when no
    ``?trace_id=`` query param is present."""
    st.title("Trace marketplace")
    st.caption(
        "Browse ingested coding-agent traces. Use the sidebar to filter; "
        "click any row to open the detail view."
    )

    filters = _sidebar_filters(conn)

    page = st.number_input(
        "Page",
        min_value=1,
        step=1,
        value=1,
        help=f"{PAGE_SIZE} rows per page.",
    )
    offset = (int(page) - 1) * PAGE_SIZE
    rows, total = list_traces(conn, filters, limit=PAGE_SIZE, offset=offset)

    pages = max(1, math.ceil(total / PAGE_SIZE))
    st.markdown(
        f"Showing **{len(rows)}** of **{total}** matching rows. "
        f"Page **{int(page)}** of **{pages}**."
    )

    if total == 0:
        st.info("No traces match these filters.")
        return

    event = st.dataframe(
        rows,
        column_order=LIST_COLUMNS,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="list_table",
    )
    selection_rows = event.selection.rows if hasattr(event, "selection") else []
    _on_row_select(rows, list(selection_rows or []))


__all__ = ["PAGE_SIZE", "render"]
