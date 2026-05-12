"""Streamlit router entry point.

Run with::

    uv run streamlit run trace_marketplace/ui/app.py

Routing is URL-driven, decided by the order of checks here:

1. ``?page=upload``  -> :mod:`trace_marketplace.ui.upload_view` (slice 5)
2. ``?trace_id=<id>`` -> :mod:`trace_marketplace.ui.detail_view`
3. otherwise         -> :mod:`trace_marketplace.ui.list_view` (filterable)

``page`` is checked first so the upload form is reachable even if the
URL also carries a stale ``trace_id`` from a prior view (e.g. someone
edits the URL by hand after clicking into a detail page).

Keeping all routing in this one file makes the link contract obvious:
the list view writes ``?trace_id=...``, the upload-CTA writes
``?page=upload``, the detail view's back button clears both, and
nothing else mutates ``st.query_params``.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

import streamlit as st

from trace_marketplace.storage.db import DEFAULT_DB_PATH, init_schema
from trace_marketplace.ui import detail_view, list_view, upload_view
from trace_marketplace.ui.components import render_top_nav

log = logging.getLogger(__name__)


def _resolve_writable_db_path(source_db: Path) -> Path:
    """Return a SQLite-writable path, copying from ``source_db`` if needed.

    Streamlit Cloud checks the repo out into a **read-only filesystem**;
    only ``/tmp`` (and whatever ``tempfile.gettempdir()`` resolves to)
    is writable. SQLite happily READS from a read-only DB file -- the
    list view, detail view, NL search, and "similar traces" panel all
    work -- but any ``INSERT`` / ``UPDATE`` from the upload flow trips
    ``OperationalError: attempt to write a readonly database``. We
    detect the case once on startup and copy the seed DB into the OS
    tempdir so uploads work without any caller changes.

    Tradeoffs:

    * **Demo persistence model.** The writable copy lives in /tmp and
      gets wiped on container restart. For a demo deployment with a
      committed seed corpus this is the right tradeoff -- the seed
      reloads from the repo on every fresh container, and uploads done
      during a session survive until the container recycles. For
      durable persistence we'd want Postgres / external storage --
      out of scope for now.
    * **Detection.** Conservative: we use ``os.access(..., os.W_OK)``
      to check if the file itself is writable. False positives (we
      copy when we didn't have to) are harmless; false negatives (we
      skip the copy when we should have copied) trip the original
      bug, so we err toward copying.
    * **Idempotency.** If the copy already exists in /tmp we DO NOT
      overwrite it. Otherwise a Streamlit rerun during an active
      session would clobber the user's in-session uploads.
    """
    if os.access(source_db, os.W_OK):
        return source_db
    writable_path = Path(tempfile.gettempdir()) / source_db.name
    if not writable_path.exists():
        shutil.copy2(source_db, writable_path)
        log.info(
            "Read-only filesystem detected at %s; copied seed DB to %s "
            "for write access. Uploads will live in /tmp until the "
            "container recycles.",
            source_db,
            writable_path,
        )
    return writable_path


@st.cache_resource
def _get_conn(db_path: str) -> sqlite3.Connection:
    """One long-lived SQLite connection per Streamlit session.

    ``st.cache_resource`` is the documented way to hand long-lived
    resources to every rerun. Cache key is the DB path string so a
    user could in principle point at a different DB by editing the
    URL / env var; ``check_same_thread=False`` keeps Streamlit's
    background thread happy.

    The path passed in is the "logical" DB location (typically
    :data:`DEFAULT_DB_PATH` pointing at ``data/marketplace.db``).
    :func:`_resolve_writable_db_path` swaps it for a writable /tmp
    copy when the source is read-only (Streamlit Cloud); on local dev
    where ``data/`` is writable the resolver is a no-op.
    """
    source = Path(db_path)
    source.parent.mkdir(parents=True, exist_ok=True)
    actual_path = _resolve_writable_db_path(source)
    conn = sqlite3.connect(actual_path, check_same_thread=False)
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

    page = st.query_params.get("page")
    if isinstance(page, list):
        page = page[0] if page else None

    trace_id = st.query_params.get("trace_id")
    if isinstance(trace_id, list):
        # Streamlit returns a list when the param is repeated; take
        # the first occurrence for backwards compatibility.
        trace_id = trace_id[0] if trace_id else None

    # ``page`` is checked first: it's the explicit navigation knob,
    # while ``trace_id`` is a content reference that might linger in
    # the URL after the user navigates elsewhere by hand.
    if page == "upload":
        render_top_nav("upload")
        upload_view.render(conn)
    elif trace_id:
        render_top_nav("detail")
        detail_view.render(conn, trace_id)
    else:
        render_top_nav("browse")
        list_view.render(conn)


# Streamlit runs the script with ``__name__ == "__main__"`` (the runner
# execs the file rather than importing it), so this gate fires both for
# ``streamlit run`` and ``python trace_marketplace/ui/app.py``.
if __name__ == "__main__":
    main()
