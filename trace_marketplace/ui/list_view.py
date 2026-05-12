"""Filterable + paginated list view.

Composition only -- pulls counts/options/rows from :mod:`queries` and
renders each row as a directory-style card via
:func:`components.render_directory_row`. Detail-view links go through
``?trace_id=...`` so bookmarkability is preserved without any JS state.

The 100-row page cap is the prompt's acceptance criterion; pagination
is manual (LIMIT/OFFSET driven by ``st.number_input("Page")``).
"""

from __future__ import annotations

import math
import sqlite3

import streamlit as st

from trace_marketplace.search.runner import NLSearchResult, run_nl_search
from trace_marketplace.ui.components import (
    render_directory_row,
    render_filters_summary,
)
from trace_marketplace.ui.queries import (
    HAS_ERROR_OPTIONS,
    ListFilters,
    bounds,
    distinct_values,
    list_traces,
    list_traces_by_ids,
    trace_counts,
)

PAGE_SIZE: int = 100
"""Acceptance-criteria-mandated cap on rows per page."""

_NL_QUERY_STATE_KEY: str = "_slice6_nl_query"
"""Where the most-recent NL search input lives across reruns. Stored
in ``st.session_state`` so the result panel survives sidebar tweaks
without re-issuing the LLM call -- :func:`_render_nl_panel` only
re-runs the search when this value changes."""

_NL_RESULT_STATE_KEY: str = "_slice6_nl_result"
"""Where the parsed :class:`NLSearchResult` is cached between reruns.
Avoids re-paying the parse + embed cost on every widget tweak."""

_PAGE_STATE_KEY: str = "_lv_page"
"""Current 1-based page index. Set by the Prev / Next buttons; reset
to 1 whenever the filter hash changes so a freshly-narrowed result
set doesn't strand the user on an empty page 7."""

_FILTER_HASH_KEY: str = "_lv_filter_hash"
"""Tracks the previous render's filter hash so we can detect filter
changes and snap the pager back to page 1 without a manual reset."""

_FILTER_WIDGET_KEYS: tuple[str, ...] = (
    "_lv_source_formats",
    "_lv_failure_labels",
    "_lv_has_error",
    "_lv_search",
    "_lv_agent_names",
    "_lv_models",
    "_lv_step_range",
    "_lv_has_atif",
    # Slice 10 outcome-aware filters.
    "_lv_has_unrecovered",
    "_lv_recovered_only",
    "_lv_clean_success",
)
"""Explicit session_state keys for every sidebar filter widget so the
empty-state 'Clear all filters' CTA can wipe them in one pass. Without
explicit keys the widgets bind to auto-generated names we can't enumerate."""

_HAS_ERROR_LABELS: dict[str, str] = {
    "any": "All",
    "errors": "Errors only",
    "successes": "Successes only",
    "unknown": "Unknown only",
}


def _reset_filters() -> None:
    """Pop every sidebar-filter widget's session_state entry.

    Called by the empty-state CTA so a user who's filtered themselves
    into a corner can recover in one click. ``st.rerun()`` is the
    caller's responsibility -- this just clears the keys.
    """
    for key in _FILTER_WIDGET_KEYS:
        st.session_state.pop(key, None)
    st.session_state.pop(_PAGE_STATE_KEY, None)
    st.session_state.pop(_FILTER_HASH_KEY, None)


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

    Layout (slice 9 declutter):

    * Primary filters always visible: ``source_format``,
      ``failure_label``, ``has_error``, and the free-text search.
      These are the four knobs the user reaches for in 90% of
      sessions.
    * Secondary filters tucked into an "Advanced filters" expander:
      ``agent_name``, ``model``, ``num_steps`` slider, ``ATIF
      column`` radio. They stay one click away without crowding the
      default view.

    Distinct-value lists are queried once per render. They're cheap
    (one DISTINCT scan apiece) and a stale cache would mean the user
    can't pick a freshly-ingested source format until they refresh.
    """
    st.sidebar.header("Filters")
    counts = trace_counts(conn)
    with st.sidebar:
        render_filters_summary(counts)
    st.sidebar.divider()

    source_formats = tuple(
        st.sidebar.multiselect(
            "Source format",
            options=distinct_values(conn, "source_format"),
            default=[],
            key="_lv_source_formats",
        )
    )
    failure_labels = tuple(
        st.sidebar.multiselect(
            "Failure label",
            options=distinct_values(conn, "failure_label"),
            default=[],
            help="Sonnet 4.6 taxonomy classification. Empty on rows the judge hasn't seen yet.",
            key="_lv_failure_labels",
        )
    )
    has_error_choice = st.sidebar.radio(
        "Outcome",
        options=HAS_ERROR_OPTIONS,
        index=0,
        format_func=lambda v: _HAS_ERROR_LABELS.get(v, v),
        key="_lv_has_error",
    )
    # Slice 10: outcome-aware filters. Live *outside* the Advanced
    # expander because finding failure-discovery / recovery-mode
    # traces is the marketplace's headline use case.
    has_unrecovered = st.sidebar.checkbox(
        "Has unrecovered failures",
        value=False,
        help=(
            "Show only traces with at least one failure event the agent "
            "did NOT escape within the session. Requires the slice-10 "
            "events extractor to have run."
        ),
        key="_lv_has_unrecovered",
    )
    recovered_only = st.sidebar.checkbox(
        "Recovered failures only",
        value=False,
        help=(
            "Show only failed traces (has_error=1) that ultimately "
            "succeeded -- the goldmine of recovery summaries."
        ),
        key="_lv_recovered_only",
    )
    clean_success = st.sidebar.checkbox(
        "Clean successes only",
        value=False,
        help=(
            "Show only clean successes (has_error=0 AND "
            "ultimately_succeeded=1) -- useful for finding golden "
            "reference traces."
        ),
        key="_lv_clean_success",
    )
    search = st.sidebar.text_input(
        "Search",
        value="",
        placeholder="id, agent name, or model",
        key="_lv_search",
    )

    with st.sidebar.expander("Advanced filters", expanded=False):
        agent_names = tuple(
            st.multiselect(
                "Agent name",
                options=distinct_values(conn, "agent_name"),
                default=[],
                key="_lv_agent_names",
            )
        )
        models = tuple(
            st.multiselect(
                "Model",
                options=distinct_values(conn, "model"),
                default=[],
                key="_lv_models",
            )
        )

        step_bounds = bounds(conn)
        lo = step_bounds["min_steps"]
        hi = max(step_bounds["max_steps"], lo + 1)
        selected_lo, selected_hi = st.slider(
            "Step count range",
            min_value=int(lo),
            max_value=int(hi),
            value=(int(lo), int(hi)),
            help=(
                "Leave at the full range to also include traces whose "
                "num_steps is NULL (unknown-format / raw-blob rows)."
            ),
            key="_lv_step_range",
        )
        min_steps, max_steps = _slider_to_step_bounds(
            (int(selected_lo), int(selected_hi)),
            (int(lo), int(hi)),
        )
        has_atif_choice = st.radio(
            "ATIF availability",
            options=("All", "Has ATIF", "Raw blob only"),
            index=0,
            help="Adapter-validated traces vs. raw-blob-only rows from unknown formats.",
            key="_lv_has_atif",
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
        has_unrecovered_failures=bool(has_unrecovered),
        recovered_failures_only=bool(recovered_only),
        clean_successes_only=bool(clean_success),
    )


def _render_nl_search_box() -> str:
    """Render the slice 6 NL-query text input and return its value.

    Wrapped in an ``st.form`` so the user can hit Enter to submit
    without burning an API call on every keystroke. The form has a
    "Clear" companion button that resets both the input and the
    cached result; sidebar tweaks reuse the cached parse so the
    LLM round-trip happens at most once per unique query.
    """
    with st.form(key="nl_search_form", clear_on_submit=False):
        col_input, col_submit, col_clear = st.columns([8, 1, 1])
        with col_input:
            query = st.text_input(
                "Natural-language search",
                value=st.session_state.get(_NL_QUERY_STATE_KEY, ""),
                placeholder=(
                    "e.g. 'find loops in claude_code' or "
                    "'agents that hallucinated APIs'"
                ),
                label_visibility="collapsed",
            )
        with col_submit:
            submitted = st.form_submit_button("Search", width="stretch")
        with col_clear:
            cleared = st.form_submit_button("Clear", width="stretch")

    if cleared:
        st.session_state.pop(_NL_QUERY_STATE_KEY, None)
        st.session_state.pop(_NL_RESULT_STATE_KEY, None)
        return ""
    if submitted and query.strip():
        st.session_state[_NL_QUERY_STATE_KEY] = query.strip()
        st.session_state.pop(_NL_RESULT_STATE_KEY, None)
    return st.session_state.get(_NL_QUERY_STATE_KEY, "")


def _run_or_reuse_nl_search(conn: sqlite3.Connection, query: str) -> NLSearchResult:
    """Memoise the NL search result in ``st.session_state``.

    The result is keyed off the query string so a sidebar tweak (or
    any incidental Streamlit rerun) doesn't re-issue the parser /
    embed calls. Calling :func:`run_nl_search` directly is still the
    primary cost-control gate -- this layer is the "don't double-pay
    for the same question" tier on top. The first-time path is
    wrapped in ``st.spinner`` so the user sees a "Searching..." badge
    instead of a 3-5s blank stare while the parser + embedding round-
    trip is in flight.
    """
    cached = st.session_state.get(_NL_RESULT_STATE_KEY)
    if isinstance(cached, tuple) and cached[0] == query:
        return cached[1]
    with st.spinner("Parsing query and searching..."):
        result = run_nl_search(conn, query)
    st.session_state[_NL_RESULT_STATE_KEY] = (query, result)
    return result


def _render_nl_results(conn: sqlite3.Connection, result: NLSearchResult) -> None:
    """Render a :class:`NLSearchResult` as the main panel.

    Three branches:

    * ``error`` populated -> show a Streamlit ``st.error`` with the
      configuration hint; the rest of the page is suppressed.
    * ``hits`` is empty -> show ``st.info`` and the parsed filters
      so the user can see what the parser made of their query.
    * Happy path -> show the "Search intent" caption + a dataframe
      of ranked traces with the similarity score in the leftmost
      column.
    """
    parsed = result.parsed
    if result.error:
        st.error(result.error)
        return
    if parsed is not None:
        st.caption(
            f"**Search intent**: {parsed.semantic_intent}  \n"
            f"**Explanation**: {parsed.explanation}"
        )
        filters_summary = []
        if parsed.failure_labels:
            filters_summary.append("labels=" + ",".join(parsed.failure_labels))
        if parsed.source_formats:
            filters_summary.append("formats=" + ",".join(parsed.source_formats))
        if parsed.agent_names:
            filters_summary.append("agents=" + ",".join(parsed.agent_names))
        if parsed.has_error is True:
            filters_summary.append("has_error=true")
        if parsed.has_error is False:
            filters_summary.append("has_error=false")
        if filters_summary:
            st.caption("Applied filters: " + " | ".join(filters_summary))
        else:
            st.caption("Applied filters: (none — pure semantic ranking)")

    if not result.hits:
        st.info(
            f"No traces matched. Candidate pool after filters: "
            f"{result.candidate_count} rows."
        )
        return

    rows_by_id = list_traces_by_ids(conn, [h.trace_id for h in result.hits])
    enriched: list[tuple[dict, float]] = []
    for hit in result.hits:
        row = rows_by_id.get(hit.trace_id)
        if row is None:
            continue
        enriched.append((dict(row), hit.similarity))

    st.markdown(
        f"Showing **top {len(enriched)}** of "
        f"**{result.candidate_count}** candidate traces, ranked by "
        f"cosine similarity to the search intent."
    )
    for row, similarity in enriched:
        render_directory_row(row, similarity=similarity)


def render(conn: sqlite3.Connection) -> None:
    """Render the list view. ``app.py`` calls this when no
    ``?trace_id=`` or ``?page=...`` query param is present."""
    st.title("Trace marketplace")
    st.caption(
        "A search directory of coding-agent traces. "
        "Type a question in the box below or use the sidebar filters."
    )

    nl_query = _render_nl_search_box()

    # NL mode REPLACES the standard filter + paginated table. The
    # sidebar filters are still rendered (and the counts header is
    # still useful) but they don't apply to the ranked results --
    # the parser owns its own filter set.
    filters = _sidebar_filters(conn)

    if nl_query:
        st.caption(
            "Natural-language search is active. Sidebar filters are "
            "ignored while a query is in the box; use **Clear** above "
            "to return to the standard filtered list."
        )
        result = _run_or_reuse_nl_search(conn, nl_query)
        _render_nl_results(conn, result)
        return

    # Snap back to page 1 whenever the filter set changes so a
    # freshly-narrowed result set doesn't strand the user on an empty
    # high-numbered page. ListFilters is a frozen dataclass so it
    # hashes deterministically.
    filter_hash = hash(filters)
    if st.session_state.get(_FILTER_HASH_KEY) != filter_hash:
        st.session_state[_PAGE_STATE_KEY] = 1
        st.session_state[_FILTER_HASH_KEY] = filter_hash
    current_page = int(st.session_state.get(_PAGE_STATE_KEY, 1))

    offset = (current_page - 1) * PAGE_SIZE
    rows, total = list_traces(conn, filters, limit=PAGE_SIZE, offset=offset)
    pages = max(1, math.ceil(total / PAGE_SIZE))

    # Defensive clamp: if a race / stale state put us past the last
    # page, snap forward and re-fetch on the next rerun.
    if current_page > pages:
        st.session_state[_PAGE_STATE_KEY] = pages
        st.rerun()

    if total == 0:
        col_msg, col_cta = st.columns([3, 1])
        with col_msg:
            st.info("No traces match these filters.")
        with col_cta:
            if st.button("Clear all filters", width="stretch", key="_lv_clear"):
                _reset_filters()
                st.rerun()
        return

    st.markdown(
        f"<div class='tm-pagecount'>Showing <b>{len(rows)}</b> of "
        f"<b>{total}</b> matching traces.</div>",
        unsafe_allow_html=True,
    )

    for row in rows:
        render_directory_row(row)

    # Compact pager only shows up when there's more than one page.
    if pages > 1:
        col_prev, col_label, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button(
                "<- Prev",
                width="stretch",
                disabled=current_page <= 1,
                key="_lv_prev",
            ):
                st.session_state[_PAGE_STATE_KEY] = current_page - 1
                st.rerun()
        with col_label:
            st.markdown(
                f"<div class='tm-pagelabel'>Page {current_page} of {pages}</div>",
                unsafe_allow_html=True,
            )
        with col_next:
            if st.button(
                "Next ->",
                width="stretch",
                disabled=current_page >= pages,
                key="_lv_next",
            ):
                st.session_state[_PAGE_STATE_KEY] = current_page + 1
                st.rerun()


__all__ = ["PAGE_SIZE", "render"]
