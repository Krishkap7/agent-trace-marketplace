"""Streamlit-based viewer for the trace marketplace.

The package is intentionally layered so the SQL helpers can be unit-tested
without importing Streamlit:

- :mod:`queries`    -- pure parameterised SQL helpers, no Streamlit import.
- :mod:`components` -- Streamlit rendering primitives for ATIF steps/headers.
- :mod:`list_view`  -- the filterable, paginated list view.
- :mod:`detail_view` -- the single-trace conversation thread view.
- :mod:`app`        -- 30-line router; entry point for ``streamlit run``.
"""
