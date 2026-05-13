"""Streamlit upload page.

Drag-drop a trace file (or paste raw content) -> stage to a tempfile ->
:func:`trace_marketplace.ingest.pipeline.ingest_file` -> inline rule pass via
:func:`trace_marketplace.enrich.failure_rules.persist_signals_for_trace` ->
**inline Sonnet 4.6 judge + Opus 4.7 events extractor** ->
per-file result with a link to the freshly-ingested trace.

Architectural decisions worth flagging:

* **Full LLM classification on the upload hot path.** As of slice 10
  follow-up, every validated upload runs through the same
  classification pipeline as the batch scripts: rule pass ->
  ``persist_judge_for_trace`` (Sonnet 4.6, ~$0.01 + 5-10s) ->
  ``persist_events_for_trace`` (Opus 4.7, ~$0.10 + 10-20s). This
  trades ~15-30s of upload latency + ~$0.10 per upload for a
  fully-classified trace immediately visible in the UI. Earlier
  iterations kept the LLM passes off the hot path (batch only) but
  that confused users who expected the result block to surface a
  semantic label right away. Both helpers have a no-raise contract
  and degrade gracefully when ``ANTHROPIC_API_KEY`` is missing
  (Streamlit Cloud demo without secrets configured -> uploads still
  ingest, just without the semantic columns until the next batch
  run picks them up via the existing ``WHERE failure_label IS NULL``
  gate).

* **Inline rule pass.** The rule signals are pure-Python, deterministic,
  and free, so running them on upload gives the result block something
  useful to display ("signals: loop, ended_on_error") and means the new
  trace shows up in the list view with ``has_error`` populated -- not as
  an orphaned NULL row. Runs BEFORE the LLM passes so the judge prompt
  has the rule signals as a structural prior.

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
import os
import sqlite3
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import streamlit as st

from trace_marketplace.enrich.failure_events import (
    EventsResult,
    persist_events_for_trace,
)
from trace_marketplace.enrich.failure_judge import (
    JudgeResult,
    persist_judge_for_trace,
)
from trace_marketplace.enrich.failure_rules import persist_signals_for_trace
from trace_marketplace.enrich.recovered_recommendations import (
    persist_recommendations_for_trace,
)
from trace_marketplace.ingest.pipeline import IngestResult, ingest_file
from trace_marketplace.search.runner import (
    EmbedAndSimilarResult,
    embed_and_recommend_similar,
)

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
    "**Full classification runs on every upload.** Rule-based signals "
    "(`has_error`, `failure_signals`) fire first, followed by the "
    "Sonnet 4.6 judge (`failure_label` + reasoning, ~$0.01 + 5-10s) "
    "and the Opus 4.7 events extractor (`failure_events` + "
    "`ultimately_succeeded`, ~$0.10 + 10-20s). The events pass "
    "overwrites the Sonnet label with its own multi-event summary "
    "and may flip `has_error` to `1` if it sees an unrecovered "
    "failure the rule pass missed. Both LLM passes degrade "
    "gracefully when `ANTHROPIC_API_KEY` is unset (uploads still "
    "ingest; semantic columns stay NULL until the next batch run "
    "of `scripts/judge_failures.py` + `scripts/extract_failure_events.py`)."
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
        border: 2px dashed rgba(212, 165, 116, 0.45);
        border-radius: 12px;
        background: rgba(212, 165, 116, 0.04);
        transition: background 120ms ease, border-color 120ms ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        background: rgba(212, 165, 116, 0.1);
        border-color: rgba(212, 165, 116, 0.8);
    }
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        font-size: 1.05rem;
        color: #f4ebe1;
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
    leave it as ``None``. ``embed_and_similar`` is populated when the
    upload was a validated ATIF row AND
    :func:`~trace_marketplace.search.runner.embed_and_recommend_similar`
    ran (which is unconditional for validated rows but degrades
    gracefully when ``OPENAI_API_KEY`` is missing); a non-validated
    upload leaves it as ``None``.

    ``judge_result`` and ``events_result`` are the slice-10
    full-classification follow-up: every validated upload gets the
    Sonnet 4.6 judge run inline (writes ``failure_label`` +
    ``failure_label_reasoning``) followed by the Opus 4.7 events
    extractor (overwrites ``failure_label`` with the events-derived
    summary, writes ``failure_events`` + ``ultimately_succeeded``,
    and may flip ``has_error`` to 1 if Opus saw an unrecovered
    failure the rule pass missed). Both pass through ``None`` when
    the trace is out of scope (raw-blob / NULL atif / missing
    ANTHROPIC_API_KEY) or when the API call failed -- the upload
    pipeline guarantees a no-raise contract per file.

    ``recovered_recommendations`` is the final pass: for failing
    uploads (``has_error = 1``, possibly bumped from 0 by the events
    pass above) we additionally compute the top-k similar traces
    that recovered in-session and render their recovery summaries
    on the upload result page. Three-state semantics:

    * ``None``      -- skipped (success / unknown trace, or upload was
                       a raw blob without rule-pass results).
    * ``[]``        -- computed but the corpus has no qualifying
                       candidates yet (e.g. fresh DB before the events
                       extractor has run).
    * non-empty     -- top-k matches with ``recoveries`` per match.
    """

    filename: str
    result: IngestResult | None
    signals: dict[str, bool] | None
    error: str | None
    embed_and_similar: EmbedAndSimilarResult | None = None
    judge_result: JudgeResult | None = None
    events_result: EventsResult | None = None
    recovered_recommendations: list[dict[str, Any]] | None = None


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

    # Slice 6.5: embed the new row and look up neighbours so the
    # result block can offer "similar traces to explore". Done only
    # for validated ATIF uploads -- raw-blob rows have no text to
    # extract. The helper is responsible for graceful degradation
    # (missing OPENAI_API_KEY, network failures), so we don't need
    # an outer try/except.
    embed_and_similar: EmbedAndSimilarResult | None = None
    if ingest_result.validated:
        embed_and_similar = embed_and_recommend_similar(conn, ingest_result.trace_id)

    # Slice 10: full LLM classification on the upload hot path.
    #
    # Sonnet 4.6 judge first -- writes ``failure_label`` + reasoning
    # so the user gets a semantic label for every validated upload
    # immediately, regardless of whether the rule pass flagged it
    # as a failure. ~$0.01 + 5-10s per call. Then Opus 4.7 events
    # extractor -- writes ``failure_events`` (per-event timeline +
    # recovery info), overwrites ``failure_label`` with the events-
    # derived summary, sets ``ultimately_succeeded``, and may flip
    # ``has_error`` to 1 if Opus saw an unrecovered failure the rule
    # pass missed. ~$0.10 + 10-20s.
    #
    # Both helpers are no-raise by contract (they swallow API errors
    # and return None). Missing ANTHROPIC_API_KEY -> graceful skip
    # so local dev without secrets keeps working. The wrapping
    # try/except here is belt-and-braces against future helper bugs;
    # the existing helpers won't trip it.
    judge_result: JudgeResult | None = None
    events_result: EventsResult | None = None
    if ingest_result.validated:
        try:
            judge_result = persist_judge_for_trace(conn, ingest_result.trace_id)
        except Exception:
            log.exception(
                "Judge pass failed for %s (%s); trace is still ingested "
                "with rule-pass signals",
                safe_name,
                ingest_result.trace_id,
            )
        try:
            events_result = persist_events_for_trace(conn, ingest_result.trace_id)
        except Exception:
            log.exception(
                "Events pass failed for %s (%s); trace is still ingested "
                "with judge label (if available)",
                safe_name,
                ingest_result.trace_id,
            )

    # Slice 10 follow-up: pre-compute the "Recoveries from similar
    # failures" list. Runs AFTER the events pass because the events
    # pass may have flipped ``has_error`` from 0 to 1 -- and the
    # recoveries-panel gate (``has_error = 1``) is what decides
    # whether we compute the cache at all. Pure-local cosine NN over
    # the existing embeddings -- no API calls, no cost.
    recovered_recommendations: list[dict[str, Any]] | None = None
    if ingest_result.validated:
        try:
            recovered_recommendations = persist_recommendations_for_trace(
                conn, ingest_result.trace_id
            )
        except Exception:
            log.exception(
                "Recovered-recommendations pass failed for %s (%s); "
                "trace is still ingested",
                safe_name,
                ingest_result.trace_id,
            )

    conn.commit()
    return _UploadOutcome(
        filename=safe_name,
        result=ingest_result,
        signals=signals,
        error=None,
        embed_and_similar=embed_and_similar,
        judge_result=judge_result,
        events_result=events_result,
        recovered_recommendations=recovered_recommendations,
    )


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
    # Slice 10: surface the LLM classification (Sonnet judge label +
    # Opus events count) BEFORE the recoveries panel so the user sees
    # the headline label first, then the supporting evidence.
    _render_llm_classification(outcome.judge_result, outcome.events_result)
    # Slice 10: render recovered-failures recommendations BEFORE the
    # slice-6 similar-traces panel. For failing uploads the recoveries
    # are the higher-signal payload (concrete advice, not just "this
    # other trace is vaguely related") so they deserve top billing.
    _render_recovered_recommendations(outcome.recovered_recommendations, trace_url)
    _render_similar_traces(outcome.embed_and_similar)


def _render_llm_classification(
    judge_result: JudgeResult | None,
    events_result: EventsResult | None,
) -> None:
    """Inline LLM-classification summary on the upload result page.

    Three-state semantics (matched to the helpers' return contract):

    * Both ``None`` -- LLM passes were skipped (raw-blob upload, NULL
      atif, missing ``ANTHROPIC_API_KEY``, or both calls failed). Render
      a small caption that explains *why* it's missing rather than
      silently omitting -- on the hosted demo without secrets configured
      this is the expected state for every upload.
    * ``judge_result`` only -- Sonnet ran but the events extractor
      failed / was skipped. Show the Sonnet label + reasoning.
    * Both populated -- show the events-derived label (Opus is more
      detailed than Sonnet alone) plus the Sonnet reasoning paragraph,
      plus the recovered/unrecovered event count split.

    The label/reasoning rendering matches ``components.render_header``
    so users see consistent treatment across upload-result and detail-
    view pages.
    """
    if judge_result is None and events_result is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.caption(
                "_LLM classification skipped:_ ``ANTHROPIC_API_KEY`` "
                "is not configured. Rule-based ``has_error`` + "
                "``failure_signals`` are populated above; the semantic "
                "label requires an Anthropic key. Configure it in "
                "Streamlit Cloud secrets (or export it locally) to get "
                "Sonnet-judged labels and Opus-extracted recovery "
                "events on every upload."
            )
        else:
            st.caption(
                "_LLM classification failed for this upload._ Rule-"
                "based signals are still applied; check the server "
                "logs for the underlying error and re-upload to retry."
            )
        return

    summary_label: str | None = None
    if events_result is not None:
        # Events extractor's summary label is what's persisted to the
        # ``failure_label`` column (it overwrites Sonnet's). Mirror
        # that here so the upload page shows the same final label as
        # the detail view will.
        from trace_marketplace.enrich.failure_events import summarize_label

        summary_label = summarize_label(events_result.events).value
    elif judge_result is not None:
        summary_label = judge_result.label.value

    bits: list[str] = []
    if summary_label:
        bits.append(f"**Label:** `{summary_label}`")
    if events_result is not None:
        recovered = sum(1 for e in events_result.events if e.recovered)
        unrecovered = len(events_result.events) - recovered
        if events_result.events:
            bits.append(
                f"**Events:** {len(events_result.events)} total "
                f"(`{recovered}` recovered, `{unrecovered}` unrecovered)"
            )
        else:
            bits.append("**Events:** none detected (clean trajectory)")

    if bits:
        st.markdown("  ·  ".join(bits))

    if judge_result is not None and judge_result.reasoning.strip():
        st.info(
            f"**Judge reasoning** (Sonnet 4.6)\n\n> {judge_result.reasoning.strip()}",
            icon=":material/psychology:",
        )


def _render_recovered_recommendations(
    recommendations: list[dict[str, Any]] | None,
    trace_url: str,
) -> None:
    """Inline "Recoveries from similar failures" panel on upload result.

    Mirrors the detail-view panel but trimmed to a compact preview (top
    3 matches, with their recovery summaries inline) so the upload
    result page stays scannable. Users who want the full top-10 list
    click through to the detail view via ``trace_url``.

    Three states match :attr:`_UploadOutcome.recovered_recommendations`:

    * ``None``      -- the trace wasn't a failure; render nothing.
    * ``[]``        -- failing trace but no qualifying candidates yet;
                       render a small caption so the user knows the
                       feature ran but the corpus is too thin to help
                       (typical for fresh databases or very novel
                       failure shapes).
    * non-empty     -- render up to three matches with their first
                       recovery summary as a quoted blockquote, plus a
                       "see all N on the trace page" footer when there
                       are more than three.
    """
    if recommendations is None:
        return
    if not recommendations:
        st.caption(
            "_No recovery suggestions yet -- no similar failures in the "
            "corpus have a recorded recovery. Re-run the events "
            "extractor once more traces have been judged._"
        )
        return

    preview_count = 3
    with st.container(border=True):
        total = len(recommendations)
        st.markdown(
            f"**Recoveries from similar failures** "
            f"({total} trace{'s' if total != 1 else ''} hit similar walls and got out)"
        )
        for rec in recommendations[:preview_count]:
            sim_url = f"?trace_id={quote(str(rec['trace_id']), safe='')}"
            similarity = rec.get("similarity")
            sim_str = (
                f"{similarity:.3f}" if isinstance(similarity, (int, float)) else "n/a"
            )
            source_format = rec.get("source_format") or "unknown"
            st.markdown(
                f"- [`{rec['trace_id']}`]({sim_url}) -- similarity "
                f"**{sim_str}** · `{source_format}`"
            )
            recoveries = rec.get("recoveries") or []
            # One inline summary per match is enough for the preview;
            # full set of per-event summaries renders on detail view.
            if recoveries:
                first = recoveries[0]
                summary = first.get("summary")
                if isinstance(summary, str) and summary.strip():
                    st.caption(f"  > {summary.strip()}")
        if total > preview_count:
            st.markdown(f"[See all {total} on the trace page ->]({trace_url})")


def _render_similar_traces(
    embed_and_similar: EmbedAndSimilarResult | None,
) -> None:
    """Inline "similar traces" panel under each successful upload.

    Three states:

    * Helper was never called (raw-blob upload) -> render nothing.
    * Helper ran but short-circuited (no API key, malformed ATIF,
      OpenAI failure) -> render a small caption with the reason.
    * Helper succeeded -> render up to five ranked neighbours with
      bookmarkable links.
    """
    if embed_and_similar is None:
        return
    if not embed_and_similar.embedded:
        if embed_and_similar.skipped_reason:
            st.caption(f"_{embed_and_similar.skipped_reason}_")
        return
    if not embed_and_similar.similar:
        st.caption(
            "_Embedded, but no other traces are close enough to surface. "
            "(The neighbourhood is sparse for very short or very generic "
            "traces.)_"
        )
        return
    with st.container(border=True):
        st.markdown(
            "**Related traces you might want to check out** (cosine nearest neighbours)"
        )
        for hit in embed_and_similar.similar:
            sim_url = f"?trace_id={quote(str(hit.trace_id), safe='')}"
            st.markdown(
                f"- [`{hit.trace_id}`]({sim_url}) — similarity **{hit.similarity:.3f}**"
            )


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
    """Render the upload page. ``app.py`` calls this when ``?page=upload``.

    The top nav strip (Browse / Upload) is rendered by
    :mod:`trace_marketplace.ui.app` before this view runs, so we don't
    repeat the back-link CTA here. The long format-help text is tucked
    behind an expander to keep the dropzone the focal point.
    """
    st.title("Upload a trace")
    st.caption(
        "Drop a session file below and it'll be ingested, classified, "
        "and embedded for similarity search."
    )
    with st.expander("Which formats are supported?", expanded=False):
        st.markdown(_FORMAT_HELP)

    tab_files, tab_paste = st.tabs(["Upload files", "Paste content"])
    with tab_files:
        _render_files_tab(conn)
    with tab_paste:
        _render_paste_tab(conn)

    st.divider()
    st.markdown(_FOOTER_NOTE)


__all__ = ["render"]
