"""Streamlit upload page.

Drag-drop a trace file (or paste raw content) -> stage to a tempfile ->
:func:`trace_marketplace.ingest.pipeline.ingest_file` -> inline rule pass via
:func:`trace_marketplace.enrich.failure_rules.persist_signals_for_trace` ->
per-file result with a link to the freshly-ingested trace.

Architectural decisions worth flagging:

* **No LLM-judge on the hot path.** Per-upload Sonnet calls would be 5-30s
  of user-facing latency and unbounded cost. Fresh uploads land with
  ``failure_label IS NULL``; the next ``scripts/judge_failures.py`` run
  picks them up via its existing ``failure_label IS NULL`` gate. The
  footer note tells the user this explicitly so the missing label
  doesn't look like a bug.

* **Inline rule pass.** The rule signals are pure-Python, deterministic,
  and free, so running them on upload gives the result block something
  useful to display ("signals: loop, ended_on_error") and means the new
  trace shows up in the list view with ``has_error`` populated -- not as
  an orphaned NULL row.

* **Per-file outcome isolation.** ``_process_upload`` wraps the whole
  ingest+rule sequence in try/except so one malformed file can't tank
  the rest of a multi-file submission. The exception is captured in the
  outcome and rendered as a red block.

* **Idempotency comes from the storage layer, not this view.**
  ``insert_trace`` does ``INSERT OR REPLACE`` on ``trace_id``; the trace
  id is derived deterministically (ATIF trajectory_id / session_id /
  raw-text identity peek). Re-uploading the same file produces a clean
  update, never a duplicate row.

* **No ``st.cache_data`` invalidation.** The list view doesn't decorate
  any reads with ``st.cache_data``; the shared ``@st.cache_resource``
  SQLite connection sees writes immediately. The "<- Back to all
  traces" link triggers a fresh render that reflects the new row.

* **Idempotent UI handling.** Streamlit reruns the script on every
  interaction. ``st.file_uploader`` keeps returning the same files
  across reruns until the user clears them, so naive auto-processing
  would re-ingest the same payload on every keystroke. We dedupe by
  ``UploadedFile.file_id`` cached in ``st.session_state`` so each
  upload only runs through the pipeline once per browser session.
"""

from __future__ import annotations

import logging
import sqlite3
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import streamlit as st

from trace_marketplace.enrich.failure_rules import persist_signals_for_trace
from trace_marketplace.ingest.pipeline import IngestResult, ingest_file

log = logging.getLogger(__name__)


_SESSION_PROCESSED_KEY: str = "upload_processed_ids"
"""``st.session_state`` slot mapping ``UploadedFile.file_id`` -> outcome.
Lets the auto-process-on-upload UX coexist with Streamlit's reruns-on-
everything model without re-ingesting the same bytes repeatedly. The
file_id is assigned by Streamlit per uploaded file and stable across
reruns within the same browser session."""

_SESSION_LAST_PASTE_KEY: str = "upload_last_paste_outcome"
"""``st.session_state`` slot for the most recent paste-tab outcome.
The paste tab is single-shot (one outcome at a time) so we don't need
a dict; we just store the latest. Survives reruns until the user
ingests another paste."""

_FORMAT_HELP: str = (
    "Format is detected from content, not file extension. Supported: "
    'ATIF (`.json` with `schema_version: "ATIF..."`), Claude Code '
    "(`.jsonl` with `sessionId`), Codex (`.jsonl` with `session_meta`-"
    "shaped events), Cursor (`.cursor.json` envelope), SWE-agent "
    "(`.json` with `instance_id` + `trajectory`). Unknown formats are "
    "still stored as raw blobs and remain findable via the list-view "
    "text search; they just lack structured filters (steps / model / "
    "has_error) until an adapter for that format is added."
)

_FOOTER_NOTE: str = (
    "**On `failure_label`.** Rule-based signals (`has_error`, "
    "`failure_signals`) fire inline on upload. The semantic "
    "`failure_label` -- the one-of-N taxonomy classification from "
    "Sonnet 4.6 -- is deliberately kept off the upload path so "
    "per-upload latency stays low and API costs stay bounded. Fresh "
    "uploads land with `failure_label = NULL`; the next batch run of "
    "`scripts/judge_failures.py` picks them up via its existing "
    "`WHERE failure_label IS NULL` gate."
)

_SAFE_FALLBACK_FILENAME: str = "upload.bin"
"""Last-resort filename used when the supplied name sanitises down to
something empty or otherwise unusable (e.g. ``..`` / ``.`` / pure
slashes). ``.bin`` keeps :func:`detect_format` on the unknown branch
rather than mis-classifying a sanitised payload."""

_DROPZONE_CSS: str = """
<style>
    /* Make Streamlit's stock file_uploader read as a big inviting
       drop zone rather than a flat 80px-tall rectangle with a small
       blue "Browse files" button. We target the public data-testid
       selectors that Streamlit guarantees as stable widget API; the
       streamlit>=1.40,<2.0 pin in pyproject.toml means these stay
       valid for our deployment.

       Three visual changes:

       1. ``min-height: 220px`` + ``padding: 2.5rem`` makes the
          drop-zone roughly 3x taller than the default, so dragging
          a file from a Finder window has an obvious target.
       2. Heavier dashed border + tinted background says "this is
          where files go", not "this is some metadata strip".
       3. Hover state gives interactive feedback so the user knows
          the zone IS the drop target, not just decoration.

       Scoped to the upload page (we only call _render_files_tab on
       ?page=upload) so the list-view sidebar's filter widgets are
       unaffected. */
    [data-testid="stFileUploaderDropzone"] {
        min-height: 220px;
        padding: 2.5rem 2rem;
        border: 2px dashed rgba(120, 130, 145, 0.55);
        border-radius: 12px;
        background: rgba(120, 130, 145, 0.06);
        transition: background 120ms ease, border-color 120ms ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        background: rgba(120, 130, 145, 0.14);
        border-color: rgba(120, 130, 145, 0.9);
    }
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        font-size: 1.05rem;
    }
</style>
"""
"""Scoped CSS injected at the top of the files tab. Lifts the
visibility of Streamlit's stock dropzone widget without forking it.
See block-comment above for the rationale on each rule."""


def _sanitise_upload_filename(raw: str) -> str:
    """Strip directory components and reject path-traversal payloads.

    Both upload entry points feed untrusted strings into ``tmp_dir /
    filename``: the paste tab pulls the filename from a free-text
    ``st.text_input``, and the drag-drop tab uses
    ``UploadedFile.name`` which is set by the browser / HTTP client
    and is therefore equally attacker-controllable. Without this
    sanitiser a filename like ``../../etc/cron.d/evil`` resolves
    OUTSIDE ``tmp_dir`` and ``write_bytes`` would clobber an arbitrary
    path on the host. (Flagged by Cursor bugbot on PR #8 commit
    bcea7cc.)

    Strategy:

    1. ``Path(raw).name`` -- drops every directory component, including
       chained ``../`` segments and absolute-path prefixes. Cross-
       platform: on POSIX the separator is ``/``; on Windows ``Path``
       handles both ``/`` and ``\\``. Streamlit Cloud is Linux so the
       POSIX behaviour is what runs in production.
    2. Strip whitespace and reject the empty string, ``.``, and ``..``
       -- ``Path("..").name`` returns ``".."`` which is a no-op against
       traversal but is still meaningless as a filename.
    3. Fall back to :data:`_SAFE_FALLBACK_FILENAME` when sanitisation
       leaves nothing usable; the upload still goes through and lands
       under that filename in the tempdir.
    """
    basename = Path(raw or "").name.strip()
    if basename in {"", ".", ".."}:
        return _SAFE_FALLBACK_FILENAME
    return basename


@dataclass(frozen=True)
class _UploadOutcome:
    """Result of processing one uploaded payload.

    Exactly one of ``result`` or ``error`` is non-None. ``signals`` is
    populated only when ``result.validated`` (the ingest pipeline
    produced a parseable ATIF blob); raw-blob and adapter-failure rows
    leave it as ``None``.
    """

    filename: str
    result: IngestResult | None
    signals: dict[str, bool] | None
    error: str | None


def _process_upload(
    conn: sqlite3.Connection, tmp_dir: Path, filename: str, payload: bytes
) -> _UploadOutcome:
    """Stage ``payload`` to ``tmp_dir/<safe-name>``, ingest, persist signals.

    Single try/except around each external step so we surface a useful
    error message to the user instead of crashing the page. The caller
    iterates over multiple files; isolating failures per file is what
    keeps a multi-file submission's good files from being lost when one
    bad file blows up.

    Filename is sanitised through :func:`_sanitise_upload_filename`
    before the tempdir join so a hostile path component
    (``../../etc/cron.d/evil``) can't escape the tempdir. We surface
    the *sanitised* name in the returned outcome so the per-file
    result block tells the user exactly what was written, not the
    payload they typed -- there's no point hiding sanitisation.
    """
    safe_name = _sanitise_upload_filename(filename)
    tmp_path = tmp_dir / safe_name
    try:
        tmp_path.write_bytes(payload)
    except OSError as exc:
        return _UploadOutcome(safe_name, None, None, f"Could not stage file: {exc}")

    try:
        ingest_result = ingest_file(tmp_path, conn)
    except Exception as exc:  # adapters can raise anything; be defensive
        log.exception("Ingest failed for %s", safe_name)
        return _UploadOutcome(
            safe_name, None, None, f"Ingest failed: {type(exc).__name__}: {exc}"
        )

    signals: dict[str, bool] | None = None
    if ingest_result.validated:
        # Pull the freshly-inserted ATIF text back and run the rule
        # pass. We re-read from the DB (rather than passing the ATIF
        # text through from ingest_file) so the upload path uses
        # exactly the bytes that ended up persisted -- if a future
        # storage-layer normaliser ever touches the column on write,
        # the rule pass sees the post-normalisation value.
        row = conn.execute(
            "SELECT atif FROM traces WHERE id = ?", (ingest_result.trace_id,)
        ).fetchone()
        atif_text = row["atif"] if row else None
        try:
            signals = persist_signals_for_trace(conn, ingest_result.trace_id, atif_text)
        except Exception:
            # Rule pass failures are NOT fatal -- the trace is already
            # ingested and visible. Log loudly so we notice in dev, but
            # keep the upload result green for the user.
            log.exception(
                "Rule pass failed for %s (%s); trace is still ingested",
                safe_name,
                ingest_result.trace_id,
            )

    conn.commit()
    return _UploadOutcome(safe_name, ingest_result, signals, None)


def _format_signals(signals: dict[str, bool] | None) -> str:
    """Human-readable summary of which rule signals fired."""
    if not signals:
        return "none fired"
    fired = sorted(name for name, hit in signals.items() if hit)
    return ", ".join(fired) if fired else "none fired"


def _render_outcome(outcome: _UploadOutcome) -> None:
    """One per-file result block."""
    if outcome.error is not None:
        st.error(f"`{outcome.filename}`: {outcome.error[:500]}")
        return

    result = outcome.result
    if result is None:
        # Defensive: dataclass invariant says result+error are mutually
        # exclusive but neither is None. If we ever hit this, fail loud.
        st.error(f"`{outcome.filename}`: internal error -- no result and no error")
        return

    trace_url = f"?trace_id={quote(result.trace_id, safe='')}"

    if not result.validated or result.source_format == "unknown":
        st.warning(
            f"`{outcome.filename}`: stored as raw blob "
            f"(detected format: `{result.source_format}`, "
            f"validated: `{result.validated}`). No structured search "
            "yet -- the trace is searchable by text only until an "
            "adapter for this format is added."
        )
        st.markdown(f"[Open `{result.trace_id}` ->]({trace_url})")
        return

    st.success(
        f"`{outcome.filename}` -> ingested as `{result.trace_id}` "
        f"(format: `{result.source_format}`, signals: "
        f"{_format_signals(outcome.signals)})"
    )
    st.markdown(f"[Open the trace ->]({trace_url})")


def _render_files_tab(conn: sqlite3.Connection) -> None:
    """The drag-drop tab: multi-file upload + per-file result rendering."""
    st.markdown(_DROPZONE_CSS, unsafe_allow_html=True)
    files = st.file_uploader(
        "Drop trace files here, or click to browse",
        accept_multiple_files=True,
        type=None,
        key="upload_files_widget",
        help=(
            "Multiple files OK; each is processed independently. "
            "Format detection is content-based so extensions don't "
            "matter -- a Claude Code log saved as `.txt` still works."
        ),
    )
    if not files:
        return

    processed: dict[str, _UploadOutcome] = st.session_state.setdefault(
        _SESSION_PROCESSED_KEY, {}
    )
    new_files = [f for f in files if f.file_id not in processed]

    if new_files:
        # One TemporaryDirectory per submission batch; cleaned up on
        # context exit. We stage every new file inside before any
        # ingest call so even if one write fails the others can still
        # try.
        with tempfile.TemporaryDirectory(prefix="upload-") as tmp:
            tmp_dir = Path(tmp)
            for uploaded in new_files:
                outcome = _process_upload(
                    conn, tmp_dir, uploaded.name, uploaded.getvalue()
                )
                processed[uploaded.file_id] = outcome

    st.subheader("Results")
    # Render in upload order so the UI matches what the user sees in
    # the uploader strip. Session_state dicts preserve insertion order
    # in Python 3.7+, but we iterate over ``files`` to be explicit.
    for uploaded in files:
        outcome = processed.get(uploaded.file_id)
        if outcome is not None:
            _render_outcome(outcome)


def _render_paste_tab(conn: sqlite3.Connection) -> None:
    """The paste-content tab: single-shot textarea + filename + ingest button.

    Differs from the files tab in two ways:

    1. **Explicit ingest button** -- pasting doesn't have a natural
       "I'm done" signal the way a file drop does, so we make the user
       click. Avoids ingesting half-typed content on every keystroke.
    2. **One outcome at a time** -- the paste tab is for single-trace
       demos / quick experiments. Multi-trace bulk loads belong in the
       files tab.
    """
    st.write(
        "Paste raw JSON or JSONL content. We save it to a temp file "
        "with the filename below (extension is a hint for the "
        "detector) and run the same pipeline."
    )
    default_name = f"pasted-{uuid.uuid4().hex[:8]}.json"
    filename = st.text_input(
        "Filename (used as a detector hint; must end in `.json` or `.jsonl`)",
        value=default_name,
        key="upload_paste_filename",
    )
    content = st.text_area(
        "Content",
        value="",
        height=300,
        key="upload_paste_content",
        placeholder='{"schema_version": "ATIF/1.7", "session_id": "...", ...}',
    )
    ingest_clicked = st.button(
        "Ingest pasted content",
        type="primary",
        disabled=not content.strip(),
        key="upload_paste_button",
    )

    if ingest_clicked:
        with tempfile.TemporaryDirectory(prefix="upload-paste-") as tmp:
            tmp_dir = Path(tmp)
            outcome = _process_upload(conn, tmp_dir, filename, content.encode("utf-8"))
        st.session_state[_SESSION_LAST_PASTE_KEY] = outcome

    last_outcome: _UploadOutcome | None = st.session_state.get(_SESSION_LAST_PASTE_KEY)
    if last_outcome is not None:
        st.subheader("Result")
        _render_outcome(last_outcome)


def render(conn: sqlite3.Connection) -> None:
    """Render the upload page. ``app.py`` calls this when ``?page=upload``."""
    st.title("Upload a trace")
    st.markdown("[<- Back to all traces](?)")
    st.caption(_FORMAT_HELP)
    st.divider()

    tab_files, tab_paste = st.tabs(["Upload files", "Paste content"])
    with tab_files:
        _render_files_tab(conn)
    with tab_paste:
        _render_paste_tab(conn)

    st.divider()
    st.markdown(_FOOTER_NOTE)


__all__ = ["render"]
