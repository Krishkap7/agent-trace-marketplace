"""Streamlit router entry point.

Run with::

    uv run streamlit run trace_marketplace/ui/app.py

Routing is URL-driven: the presence of ``?trace_id=<id>`` in
``st.query_params`` selects the detail view, anything else lands on
the filterable list view. Keeping all routing in this one file makes
the link contract obvious -- the list view writes that param, the
detail view reads it, and the back button clears it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import streamlit as st

from trace_marketplace.storage.db import DEFAULT_DB_PATH, init_schema
from trace_marketplace.ui import detail_view, list_view


@st.cache_resource
def _get_conn(db_path: str) -> sqlite3.Connection:
    """One long-lived SQLite connection per Streamlit session.

    ``st.cache_resource`` is the documented way to hand long-lived
    resources to every rerun. Cache key is the DB path string so a
    user could in principle point at a different DB by editing the
    URL / env var; ``check_same_thread=False`` keeps Streamlit's
    background thread happy.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    return conn


def main() -> None:
    st.set_page_config(
        page_title="Trace marketplace",
        page_icon=":mag:",
        layout="wide",
    )

    if not DEFAULT_DB_PATH.exists():
        # Don't trip the user up with a sqlite "unable to open" error.
        st.error(
            f"No database found at `{DEFAULT_DB_PATH}`. "
            "Run the ingest scripts (see README) before launching the UI."
        )
        return

    conn = _get_conn(str(DEFAULT_DB_PATH))

    trace_id = st.query_params.get("trace_id")
    if isinstance(trace_id, list):
        # Streamlit returns a list when the param is repeated; take
        # the first occurrence for backwards compatibility.
        trace_id = trace_id[0] if trace_id else None

    if trace_id:
        detail_view.render(conn, trace_id)
    else:
        list_view.render(conn)


# Streamlit runs the script with ``__name__ == "__main__"`` (the runner
# execs the file rather than importing it), so this gate fires both for
# ``streamlit run`` and ``python trace_marketplace/ui/app.py``.
if __name__ == "__main__":
    main()
