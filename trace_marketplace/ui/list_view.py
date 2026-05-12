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
from urllib.parse import quote

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

_OPEN_LINK_COLUMN: str = "_open"
"""Synthetic column injected into every list row so ``st.dataframe`` can
render an explicit ``Open ->`` link via ``st.column_config.LinkColumn``.
Replaces the older ``selection_mode='single-row'`` checkbox pattern: a
labelled link is a much more discoverable affordance for "click to drill
in" than a single-row checkbox that looks like a multi-select toggle.
The underscore prefix keeps the name visually distinct from real
SQL-projected columns in :data:`LIST_COLUMNS`."""

_HAS_ERROR_LABELS: dict[str, str] = {
    "any": "All",
    "errors": "Errors only (has_error=1)",
    "successes": "Successes only (has_error=0)",
    "unknown": "Unknown only (has_error IS NULL)",
}


def _has_atif_label(value: str) -> bool | None:
    return {"All": None, "Has ATIF": True, "Raw blob only": False}[value]


def _slider_to_step_bounds(
    selected: tuple[int, int], slider_bounds: tuple[int, int]
) -> tuple[int | None, int | None]:
    """Translate the ``num_steps`` slider value into ``ListFilters``
    ``min_steps`` / ``max_steps`` arguments.

    When the slider sits at its widest position we return ``(None, None)``
    so the WHERE clause is skipped entirely. SQL's ``NULL >= ?`` evaluates
    to ``NULL`` (falsy), so passing through the integer bounds would
    permanently hide every row with ``num_steps IS NULL`` -- which is
    every unknown-format / adapter-failure row in the corpus -- and the
    user wouldn't have a way to recover them from the slider alone.
    (Caught by Cursor bugbot on commit 655c824.)

    When the slider has been narrowed in either direction we pass the
    integers through; rows with ``NULL`` step counts can't satisfy a
    numeric range and getting filtered out matches the standard semantic.
    """
    sel_lo, sel_hi = selected
    bound_lo, bound_hi = slider_bounds
    if sel_lo <= bound_lo and sel_hi >= bound_hi:
        return None, None
    return sel_lo, sel_hi


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
        help=(
            "Leave at the full range to also include traces whose "
            "num_steps is NULL (unknown-format / raw-blob rows)."
        ),
    )
    min_steps, max_steps = _slider_to_step_bounds(
        (int(selected_lo), int(selected_hi)),
        (int(lo), int(hi)),
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
        min_steps=min_steps,
        max_steps=max_steps,
        search=search.strip(),
    )


def render(conn: sqlite3.Connection) -> None:
    """Render the list view. ``app.py`` calls this when no
    ``?trace_id=`` or ``?page=...`` query param is present."""
    header_col, cta_col = st.columns([5, 1])
    with header_col:
        st.title("Trace marketplace")
        st.caption(
            "Browse ingested coding-agent traces. Use the sidebar to "
            "filter; click **Open ->** on any row to see the full trace."
        )
    with cta_col:
        # Plain anchor instead of ``st.button`` + ``st.query_params``
        # write: the link is bookmarkable, opens in a new tab on
        # middle-click, and is one fewer Streamlit rerun on the
        # happy path. The CTA lives in a right-aligned column so it
        # reads as a primary action without crowding the title.
        st.markdown(
            "<div style='text-align: right; padding-top: 1rem;'>"
            "<a href='?page=upload' style='font-size: 1rem;'>"
            "Upload trace -></a></div>",
            unsafe_allow_html=True,
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

    # Inject a same-page relative URL per row. ``quote(..., safe='')``
    # percent-encodes any character that's not in the unreserved set,
    # so SWE-agent IDs like ``Shoobx__mypy-zope-88`` survive verbatim
    # but a future trace_id containing ``&``, ``?``, ``#``, or ``=``
    # can't break out of the ``trace_id`` value. Streamlit detects
    # the query_params change on click and reruns -- ``app.py``'s
    # router lands the user on the detail view.
    for row in rows:
        row[_OPEN_LINK_COLUMN] = f"?trace_id={quote(str(row['id']), safe='')}"

    st.dataframe(
        rows,
        column_order=(_OPEN_LINK_COLUMN, *LIST_COLUMNS),
        column_config={
            _OPEN_LINK_COLUMN: st.column_config.LinkColumn(
                label="Details",
                display_text="Open ->",
                help="Open the full conversation thread + judge reasoning.",
            ),
        },
        hide_index=True,
        width="stretch",
        key="list_table",
    )


__all__ = ["PAGE_SIZE", "render"]
