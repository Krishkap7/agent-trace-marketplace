"""Pure Streamlit rendering primitives for the trace viewer.

The detail view composes these into a conversation thread; the list view
uses :func:`render_filters_summary` for the sidebar header. None of the
helpers touch SQL -- they take already-fetched data and render it.

Why a separate module instead of inlining into ``detail_view.py``?

- It keeps the conversation-thread renderer reusable: the same
  ``render_step`` will eventually appear in a "compare traces" view.
- Tests can import the module to assert on its public API surface
  without needing a Streamlit runtime (the functions ARE called from
  Streamlit only, but importing the module shouldn't blow up).
- Slice 9 redesign moved global CSS injection, the top nav strip, and
  the directory-row card renderer in here too so list / detail / upload
  all share one visual vocabulary.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

import streamlit as st

_GLOBAL_CSS: str = """
<style>
    /* Slice 9 redesign: warm brown + cream palette + serif headers.
       The `.streamlit/config.toml` theme handles the base colours and
       widget surfaces; this stylesheet only adds the bits Streamlit's
       theme system doesn't expose (font stacks, header sizing, badge /
       directory-card visuals, top-nav layout).

       Selectors target Streamlit's documented `data-testid` attributes
       (stable since 1.30) and our own `tm-*` classes so future
       Streamlit upgrades only risk breaking the small surface they
       guarantee. */

    /* Streamlit Cloud runs on Linux, which doesn't ship Inter; without
       this import the page would fall back to DejaVu Sans and look
       noticeably heavier than the macOS dev preview. Loaded with
       display=swap so the page paints immediately with the system
       fallback while Inter is fetched. */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Base font + size on the app root only. We deliberately do NOT
       use a wildcard like [class*="st-emotion"] here because that
       would override the Material Icons / Material Symbols font that
       Streamlit uses for st.chat_message avatars and expander
       chevrons -- the browser would then render the icon names
       ("face", "smart_toy", "arrow_right") as literal text. CSS
       inheritance handles the rest. */
    html, body, .stApp {
        font-family: "Inter", -apple-system, BlinkMacSystemFont,
                     "Segoe UI", Helvetica, Arial, sans-serif;
        font-size: 16px;
        line-height: 1.55;
    }
    /* Belt-and-braces: pin Material Icon elements back to their
       glyph fonts so they survive any future blanket font rules.
       Streamlit ships these elements with multiple matching class
       names depending on the icon family in use. */
    .material-icons,
    .material-icons-outlined,
    [class*="material-icons"],
    [class*="material-symbols"] {
        font-family: "Material Symbols Outlined", "Material Icons",
                     "Material Icons Outlined" !important;
    }
    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        font-family: "Iowan Old Style", "Palatino", Georgia,
                     "Times New Roman", serif;
        letter-spacing: -0.01em;
    }
    h1, [data-testid="stMarkdownContainer"] h1 { font-size: 2.4rem; }
    h2, [data-testid="stMarkdownContainer"] h2 { font-size: 1.7rem; }

    .stApp { background: #2b1f17; }
    [data-testid="stSidebar"] { background: #251a13; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.95rem;
    }

    /* Top nav strip: anchor-tag tabs that switch pages via ?page=...
       so URLs stay bookmarkable. The active state is a thicker bottom
       border in the tan accent + bolder text; the others are muted
       cream. */
    .tm-topnav {
        display: flex;
        gap: 0.25rem;
        border-bottom: 1px solid rgba(244, 235, 225, 0.12);
        margin: 0 0 1.5rem 0;
        padding: 0;
    }
    .tm-topnav a {
        display: inline-block;
        padding: 0.7rem 1.2rem;
        margin-bottom: -1px;
        color: rgba(244, 235, 225, 0.55);
        text-decoration: none;
        font-family: "Iowan Old Style", "Palatino", Georgia, serif;
        font-size: 1.05rem;
        border-bottom: 2px solid transparent;
        transition: color 120ms ease, border-color 120ms ease;
    }
    .tm-topnav a:hover { color: #f4ebe1; }
    .tm-topnav a.tm-topnav-active {
        color: #f4ebe1;
        border-bottom-color: #d4a574;
        font-weight: 600;
    }

    /* Directory-row cards: one bordered block per trace. The whole
       card is a single <a> so the click target is the full surface --
       hovering anywhere lifts the border + chevron, clicking anywhere
       navigates to ?trace_id=... Middle-click still opens in a new
       tab, right-click "Copy link" works, keyboard focus reveals the
       hover state -- everything browsers give you for free on a real
       anchor. We strip the default anchor underline + colour so the
       inner title / badges keep their custom typography. */
    a.tm-row,
    a.tm-row:visited {
        display: block;
        border: 1px solid rgba(244, 235, 225, 0.12);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.65rem;
        background: rgba(58, 42, 30, 0.55);
        color: inherit;
        text-decoration: none;
        transition: border-color 120ms ease, background 120ms ease;
    }
    a.tm-row:hover {
        border-color: rgba(212, 165, 116, 0.55);
        background: rgba(58, 42, 30, 0.85);
    }
    a.tm-row:hover .tm-row-title {
        color: #fffdf9;
    }
    a.tm-row:hover .tm-row-arrow {
        color: #d4a574;
        transform: translateX(2px);
    }

    /* Compact pagination strip (slice 9 replacement for the
       st.number_input pager). Layout is three Streamlit columns;
       we just style the centre "Page X of Y" label and tighten
       the button row so it doesn't sit on its own giant strip. */
    .tm-pagelabel {
        text-align: center;
        font-family: "Iowan Old Style", "Palatino", Georgia, serif;
        font-size: 1rem;
        padding: 0.45rem 0;
        color: rgba(244, 235, 225, 0.78);
    }
    .tm-pagecount {
        text-align: right;
        color: rgba(244, 235, 225, 0.65);
        margin-bottom: 0.5rem;
    }
    /* Step-header row inside st.chat_message. The label gets serif
       weight + warm cream, the timestamp shrinks + dims so it reads
       as metadata rather than competing with the message body. */
    .tm-step-head {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.25rem;
    }
    .tm-step-label {
        font-family: "Iowan Old Style", "Palatino", Georgia, serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: #f4ebe1;
    }
    .tm-step-time {
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
        font-size: 0.78rem;
        color: rgba(244, 235, 225, 0.45);
        white-space: nowrap;
    }
    /* Compressed-thread elision marker. When detail_view skips middle
       steps to keep the websocket payload under Streamlit's message
       size limit, this is the placeholder block the user sees between
       the head and tail slices. Visual treatment matches the muted
       metadata styling of timestamps so it doesn't compete with
       actual step rows. */
    .tm-step-elision {
        margin: 1.25rem 0;
        padding: 0.85rem 1rem;
        text-align: center;
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
        font-size: 0.85rem;
        color: rgba(244, 235, 225, 0.55);
        background: rgba(244, 235, 225, 0.04);
        border-radius: 8px;
        border: 1px dashed rgba(244, 235, 225, 0.2);
    }
    .tm-row-head {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.2rem;
    }
    .tm-row-title {
        font-family: "Iowan Old Style", "Palatino", Georgia, serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: #f4ebe1;
        line-height: 1.35;
    }
    .tm-row-id {
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
        font-size: 0.78rem;
        color: rgba(244, 235, 225, 0.42);
        word-break: break-all;
        margin-bottom: 0.45rem;
    }
    .tm-row-arrow {
        color: rgba(244, 235, 225, 0.35);
        font-size: 1.4rem;
        line-height: 1;
        white-space: nowrap;
        transition: color 120ms ease, transform 120ms ease;
    }
    .tm-row-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.45rem;
    }
    .tm-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.82rem;
        background: rgba(244, 235, 225, 0.08);
        color: rgba(244, 235, 225, 0.85);
        border: 1px solid rgba(244, 235, 225, 0.1);
    }
    .tm-badge-error { color: #e8a08e; border-color: rgba(232, 160, 142, 0.4); }
    .tm-badge-ok    { color: #a8c9a3; border-color: rgba(168, 201, 163, 0.4); }
    .tm-badge-warn  { color: #d4a574; border-color: rgba(212, 165, 116, 0.5); }
    .tm-badge-sim   { color: #d4a574; background: rgba(212, 165, 116, 0.12); }

    /* Slice 10: outcome chip in the detail-view header. Five distinct
       states drive five visual variants (success / recovered failure /
       failed / subtle failure / unknown). The chip sits next to the
       summary label caption so the user can read process vs. outcome
       in one glance. */
    .tm-outcome {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-family: "Iowan Old Style", "Palatino", Georgia, serif;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.005em;
        border: 1px solid rgba(244, 235, 225, 0.15);
        background: rgba(244, 235, 225, 0.05);
        color: rgba(244, 235, 225, 0.9);
        margin-right: 0.5rem;
    }
    .tm-outcome-success {
        color: #b8d4ad; border-color: rgba(168, 201, 163, 0.5);
        background: rgba(168, 201, 163, 0.10);
    }
    .tm-outcome-recovered {
        color: #f0c47a; border-color: rgba(212, 165, 116, 0.55);
        background: rgba(212, 165, 116, 0.13);
    }
    .tm-outcome-failed {
        color: #f2a48e; border-color: rgba(232, 160, 142, 0.5);
        background: rgba(232, 160, 142, 0.10);
    }
    .tm-outcome-subtle {
        color: #c8b6e0; border-color: rgba(200, 182, 224, 0.5);
        background: rgba(200, 182, 224, 0.10);
    }
    .tm-outcome-unknown {
        color: rgba(244, 235, 225, 0.7);
    }
    .tm-outcome-caption {
        color: rgba(244, 235, 225, 0.6);
        font-size: 0.9rem;
        margin-left: 0.2rem;
    }

    /* Slice 10: failure-events timeline. Each event is its own row;
       recovered events get a muted accent + the summary quoted below;
       unrecovered events get a brighter accent line so the user's eye
       lands there first. */
    .tm-events {
        margin: 0.4rem 0 0.8rem 0;
    }
    .tm-event-row {
        display: block;
        padding: 0.55rem 0.8rem;
        margin-bottom: 0.45rem;
        border-left: 3px solid rgba(232, 160, 142, 0.8);
        background: rgba(232, 160, 142, 0.06);
        border-radius: 4px;
    }
    .tm-event-recovered {
        border-left-color: rgba(212, 165, 116, 0.6);
        background: rgba(212, 165, 116, 0.06);
    }
    .tm-event-label {
        font-weight: 600;
        color: #f4ebe1;
        margin-right: 0.5rem;
    }
    .tm-event-range {
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
        font-size: 0.85rem;
        color: rgba(244, 235, 225, 0.7);
    }
    .tm-event-status {
        font-size: 0.85rem;
        color: rgba(244, 235, 225, 0.55);
    }
    .tm-event-status-recovered { color: #f0c47a; }
    .tm-event-summary {
        margin-top: 0.3rem;
        margin-left: 0.5rem;
        padding-left: 0.6rem;
        border-left: 2px solid rgba(244, 235, 225, 0.15);
        font-style: italic;
        color: rgba(244, 235, 225, 0.85);
    }

    /* Slice 10: recoveries panel rendered for has_error=1 traces with
       populated recommendations. Tighter card variant of the directory
       row -- title (linked trace id) on top, recovery summaries quoted
       below. */
    .tm-recpanel-head {
        font-family: "Iowan Old Style", "Palatino", Georgia, serif;
        font-size: 1.15rem;
        font-weight: 600;
        margin: 0.4rem 0 0.6rem 0;
        color: #f4ebe1;
    }
    .tm-rec-row {
        display: block;
        padding: 0.65rem 0.95rem;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(244, 235, 225, 0.10);
        border-radius: 8px;
        background: rgba(58, 42, 30, 0.42);
        color: inherit;
        text-decoration: none;
        transition: border-color 120ms ease, background 120ms ease;
    }
    .tm-rec-row:hover {
        border-color: rgba(212, 165, 116, 0.45);
        background: rgba(58, 42, 30, 0.7);
    }
    .tm-rec-head {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 0.8rem;
        margin-bottom: 0.25rem;
    }
    .tm-rec-id {
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
        font-size: 0.85rem;
        color: #f4ebe1;
        word-break: break-all;
    }
    .tm-rec-meta {
        font-size: 0.82rem;
        color: rgba(244, 235, 225, 0.65);
        white-space: nowrap;
    }
    .tm-rec-blurb {
        font-size: 0.85rem;
        color: rgba(244, 235, 225, 0.7);
        margin-bottom: 0.35rem;
    }
    .tm-rec-quote {
        margin: 0.25rem 0 0.25rem 0.5rem;
        padding-left: 0.6rem;
        border-left: 2px solid rgba(212, 165, 116, 0.35);
        font-style: italic;
        color: rgba(244, 235, 225, 0.88);
        font-size: 0.92rem;
    }

    /* Caption / muted helper text -- bump from Streamlit's 0.875rem
       default so it stays readable at the bigger base font. */
    [data-testid="stCaptionContainer"] { font-size: 0.95rem; }
</style>
"""


def inject_global_styles() -> None:
    """Inject the slice 9 global stylesheet on every Streamlit rerun.

    Why no session_state guard: Streamlit rebuilds the DOM from
    scratch on every interaction (clicking a sidebar filter,
    submitting the NL form, etc.), so a "inject once per session"
    gate would skip CSS on every rerun after the first and leave the
    page rendering as un-styled raw HTML. The cost of re-injecting a
    ~5 KB ``<style>`` block per rerun is negligible. Called from
    :func:`render_top_nav` so any view that paints the nav strip
    automatically gets the styles too.
    """
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def render_top_nav(active: str) -> None:
    """Render the Browse / Upload tab strip + ensure global styles.

    ``active`` is one of ``"browse"`` / ``"upload"`` / ``"detail"`` and
    drives which anchor gets the ``tm-topnav-active`` class. Detail
    view counts as part of Browse (so the "Browse" tab stays
    highlighted while the user is reading a single trace).

    Anchors use ``?page=...`` so the existing URL routing in
    :mod:`trace_marketplace.ui.app` keeps working unchanged: the
    Browse anchor wipes ``page`` to land on the list view, Upload
    sets ``page=upload``. The href is a relative URL so it inherits
    the current host/path (works locally and on Streamlit Cloud
    without any extra config).
    """
    inject_global_styles()
    browse_class = "tm-topnav-active" if active in ("browse", "detail") else ""
    upload_class = "tm-topnav-active" if active == "upload" else ""
    st.markdown(
        f"""
        <nav class="tm-topnav">
          <a href="./" class="{browse_class}" target="_self">Browse</a>
          <a href="?page=upload" class="{upload_class}" target="_self">Upload</a>
        </nav>
        """,
        unsafe_allow_html=True,
    )


_TITLE_MAX_CHARS: int = 110
"""Soft cap on the directory-card title length. Beyond this we ellipsise
so a verbose first-user message doesn't blow out the card height."""

_SYSTEM_TITLE_MIN_CHARS: int = 80
"""A system-source step counts as a usable title only if its message is
at least this long. Bare framing snippets (one short line) get filtered
out; substantive bug reports / task descriptions pass through."""

_SYSTEM_FRAMING_PREFIXES: tuple[str, ...] = (
    "SETTING:",
    "You are an autonomous",
    "You are a helpful",
    "You are an AI",
)
"""Opening tokens that mark a system step as agent-harness framing,
not as the actual task description. SWE-agent traces in particular
start with a multi-paragraph 'SETTING: You are an autonomous
programmer...' instruction block that has to be skipped so the
following 'We're currently solving the following issue...' step
(which IS the task) becomes the title."""


_TASK_MARKERS: tuple[str, ...] = ("ISSUE:", "TASK:", "BUG:", "PROBLEM:")
"""Section markers that introduce the human-readable task summary
inside a system-step message. SWE-agent traces in particular use
``\\nISSUE:\\n<one-line title>\\n<long body>`` -- the line after the
marker is the actual GitHub issue title, which is the right thing
to show as a card title."""


def _strip_lead(text: str) -> str:
    """Trim a multi-line message down to its most informative first line.

    When the message contains a section marker from :data:`_TASK_MARKERS`,
    we return the first non-empty line *after* the marker (the issue /
    task title). Otherwise we return the message's first non-empty
    line.
    """
    lines = [line.strip() for line in text.splitlines()]
    for idx, line in enumerate(lines):
        if line in _TASK_MARKERS:
            for follow in lines[idx + 1 :]:
                if follow:
                    return follow
            break
    for line in lines:
        if line:
            return line
    return text.strip()


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "..."


def extract_task_title(row: dict[str, Any]) -> str | None:
    """Pull a human-readable title out of a row's ATIF blob.

    The directory-card title is the first thing a user reads, so we
    want something concrete -- "Add retry-with-backoff logic to the
    HTTP client" -- not an opaque UUID. The heuristic walks the trace's
    steps in order and prefers, in priority:

    1. The first ``source='user'`` step's message. This is the user's
       prompt for cursor / claude_code / codex / friendly formats.
    2. The first ``source='system'`` step whose message is at least
       :data:`_SYSTEM_TITLE_MIN_CHARS` long. SWE-agent's first system
       step is short framing ("SETTING: You are..."); its second is
       the actual bug report -- which IS the task, so we surface it.
    3. The first step with any non-empty message. Last-resort fallback
       for unusual formats.

    Returns ``None`` if nothing usable can be extracted -- caller is
    expected to fall back to the trace_id. Decoding failures are
    swallowed (returning ``None``) since a malformed ATIF column is a
    data issue, not a UI one.
    """
    atif_text = row.get("atif")
    if not atif_text or not isinstance(atif_text, str):
        return None
    try:
        atif = json.loads(atif_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(atif, dict):
        return None

    steps = atif.get("steps")
    if not isinstance(steps, list):
        return None

    long_system: str | None = None
    any_message: str | None = None
    for step in steps:
        if not isinstance(step, dict):
            continue
        msg = step.get("message")
        if not isinstance(msg, str) or not msg.strip():
            continue
        source = (step.get("source") or "").lower()
        if source == "user":
            return _truncate(_strip_lead(msg), _TITLE_MAX_CHARS)
        if source == "system" and len(msg) >= _SYSTEM_TITLE_MIN_CHARS:
            stripped = msg.lstrip()
            if any(stripped.startswith(prefix) for prefix in _SYSTEM_FRAMING_PREFIXES):
                continue  # skip agent-harness framing; keep looking
            if long_system is None:
                long_system = _strip_lead(msg)
        if any_message is None:
            any_message = _strip_lead(msg)

    chosen = long_system or any_message
    return _truncate(chosen, _TITLE_MAX_CHARS) if chosen else None


_BADGE_CLASS_FOR_LABEL: dict[str, str] = {
    "loop": "tm-badge tm-badge-warn",
    "gave_up": "tm-badge tm-badge-error",
    "hallucinated_api": "tm-badge tm-badge-error",
    "misread_error": "tm-badge tm-badge-error",
    "environment_issue": "tm-badge tm-badge-warn",
    "wrong_approach": "tm-badge tm-badge-warn",
    "partial_success": "tm-badge tm-badge-warn",
    "success": "tm-badge tm-badge-ok",
}
"""Map the slice-4 failure-label taxonomy to coloured badge classes.
Unknown labels fall back to the neutral ``tm-badge`` styling."""


def _escape(value: Any) -> str:
    """Minimal HTML escape for badge / title content.

    We only escape the four characters that can break the surrounding
    ``unsafe_allow_html=True`` markdown block. None / "" returns "-"
    so empty badges still render with a visible placeholder rather
    than collapsing to a zero-width pill.
    """
    if value is None or value == "":
        return "-"
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_directory_row(
    row: dict[str, Any], *, similarity: float | None = None
) -> None:
    """Render one trace as a directory-style card.

    Slice 9 replacement for the column-based ``st.dataframe`` rows.
    Card layout:

    * **Title** -- :func:`extract_task_title` pulls a sentence-length
      summary from the trace's first user (or substantive system)
      step. Falls back to "Untitled trace" if extraction fails.
    * **Trace ID** -- shown small + monospace + muted underneath the
      title; useful for copy/paste but not the focal point.
    * **Badges** -- source_format, agent_name, model, num_steps,
      failure_label, has_error. Optional similarity score for NL /
      similar-trace panels.
    * **Chevron** -- small right-pointing glyph on the trailing edge
      that nudges right on hover. The whole card surface is a single
      ``<a href="?trace_id=...">`` so a click anywhere navigates
      to the detail view (and middle-click opens a new tab, etc.).

    The whole card is one ``st.markdown`` call with
    ``unsafe_allow_html=True`` so it lays out as flex/grid CSS rather
    than as a stack of Streamlit columns (which can't be styled per-
    row).
    """
    trace_id = row.get("id", "")
    trace_id_safe = _escape(trace_id)
    title = extract_task_title(row) or "Untitled trace"
    title_safe = _escape(title)
    open_href = f"?trace_id={quote(str(trace_id), safe='')}"

    badges: list[str] = []
    if similarity is not None:
        badges.append(
            f'<span class="tm-badge tm-badge-sim">similarity {similarity:.3f}</span>'
        )
    for key in ("source_format", "agent_name", "model"):
        value = row.get(key)
        if value:
            badges.append(f'<span class="tm-badge">{_escape(value)}</span>')
    num_steps = row.get("num_steps")
    if num_steps is not None:
        badges.append(f'<span class="tm-badge">{_escape(num_steps)} steps</span>')

    failure_label = row.get("failure_label")
    if failure_label:
        cls = _BADGE_CLASS_FOR_LABEL.get(str(failure_label), "tm-badge")
        badges.append(f'<span class="{cls}">{_escape(failure_label)}</span>')

    has_error = row.get("has_error")
    if has_error == 1:
        badges.append('<span class="tm-badge tm-badge-error">has_error</span>')
    elif has_error == 0:
        badges.append('<span class="tm-badge tm-badge-ok">clean</span>')

    badges_html = "".join(badges) or '<span class="tm-badge">no metadata</span>'
    st.markdown(
        f"""
        <a class="tm-row" href="{open_href}" target="_self">
          <div class="tm-row-head">
            <div class="tm-row-title">{title_safe}</div>
            <span class="tm-row-arrow" aria-hidden="true">&rsaquo;</span>
          </div>
          <div class="tm-row-id">{trace_id_safe}</div>
          <div class="tm-row-meta">{badges_html}</div>
        </a>
        """,
        unsafe_allow_html=True,
    )


SOURCE_LABELS: dict[str, str] = {
    "user": "User",
    "agent": "Agent",
    "system": "System (harness)",
    "tool": "Tool result",
}
"""Friendly labels for ATIF ``Step.source`` values. Unknown sources
render with the raw string in title-case.

``User`` and ``Agent`` are obvious from chat-message left/right
alignment so they stay short. ``System`` and ``Tool`` get a small
parenthetical so a SWE-agent harness step doesn't get mistaken for
"our marketplace's system" or "Sonnet's commentary" -- that
ambiguity bit real users on the previous iteration. The detail
view also conditionally renders a fuller legend for SWE-agent
traces specifically (see :mod:`trace_marketplace.ui.detail_view`)."""


def _pretty_arguments(arguments: Any) -> str:
    """Render tool-call arguments as readable text.

    The ATIF schema lets ``arguments`` be a dict OR a string (some
    agents stringify their JSON). Both are dumped to indented JSON
    for display; non-JSON strings pass through verbatim.
    """
    if isinstance(arguments, str):
        try:
            return json.dumps(json.loads(arguments), indent=2)
        except json.JSONDecodeError:
            return arguments
    try:
        return json.dumps(arguments, indent=2, default=str)
    except TypeError:
        return repr(arguments)


def _result_text(result: dict[str, Any]) -> str:
    """Extract the displayable content of an ``ObservationResult``.

    ATIF allows ``content`` to be a string OR a list of content blocks
    (Anthropic shape). We try to flatten lists into text; otherwise the
    block is JSON-dumped so nothing is silently lost.
    """
    content = result.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                chunks.append(block["text"])
            else:
                chunks.append(json.dumps(block, default=str))
        return "\n".join(chunks)
    if content is None:
        return ""
    return json.dumps(content, indent=2, default=str)


def _is_encrypted_reasoning(text: str) -> bool:
    """Slice 2.7 introduced ``"[encrypted reasoning, N bytes]"`` and
    ``"[encrypted thinking, signature N bytes]"`` placeholders for
    opaque reasoning blocks. We badge them rather than rendering them
    as if they were normal italic chain-of-thought."""
    return isinstance(text, str) and text.startswith("[encrypted ")


def render_step(step: dict[str, Any], index: int) -> None:
    """Render one ATIF step as a chat-style message.

    Visual rules:

    * ``source="user"`` -> chat message on the right ("user").
    * ``source="agent"`` / ``"assistant"`` -> chat message on the left
      ("assistant").
    * ``source="system"`` / ``"tool"`` -> plain caption + boxed text so
      these don't get mistaken for conversation turns.
    * Reasoning content is rendered italicised under the main message;
      encrypted-reasoning placeholders get a small badge instead.
    * ``tool_calls`` and ``observation.results`` go into ``st.expander``
      blocks so the thread stays scannable.
    """
    source = (step.get("source") or "").lower() or "agent"

    if source == "user":
        chat_role = "user"
    elif source in ("agent", "assistant"):
        chat_role = "assistant"
    else:
        chat_role = None  # system / tool / unknown -> plain block.

    label = SOURCE_LABELS.get(source, source.title() or "Step")
    label_safe = _escape(label)
    timestamp = step.get("timestamp")
    timestamp_html = (
        f'<span class="tm-step-time">{_escape(timestamp)}</span>' if timestamp else ""
    )
    header = (
        f'<div class="tm-step-head">'
        f'<span class="tm-step-label">{label_safe} &middot; Step {index + 1}</span>'
        f"{timestamp_html}"
        f"</div>"
    )

    def _body() -> None:
        st.markdown(header, unsafe_allow_html=True)
        message = step.get("message")
        if message:
            st.markdown(message)

        reasoning = step.get("reasoning_content")
        if reasoning:
            if _is_encrypted_reasoning(reasoning):
                st.caption(f":lock: {reasoning}")
            else:
                st.markdown(f"*{reasoning}*")

        tool_calls = step.get("tool_calls") or []
        for tc in tool_calls:
            name = tc.get("function_name", "<unknown>")
            call_id = tc.get("tool_call_id", "")
            title = f"Tool call: `{name}`"
            if call_id:
                title += f" (id: `{call_id}`)"
            with st.expander(title, expanded=False):
                st.code(_pretty_arguments(tc.get("arguments")), language="json")

        observation = step.get("observation")
        results = (observation or {}).get("results") or []
        for r_idx, result in enumerate(results):
            source_call_id = result.get("source_call_id")
            label_obs = f"Observation #{r_idx + 1}"
            if source_call_id:
                label_obs += f" -> call `{source_call_id}`"
            with st.expander(label_obs, expanded=False):
                text = _result_text(result)
                if text:
                    st.code(text, language=None)
                else:
                    st.caption("_(empty observation)_")

    if chat_role is None:
        with st.container(border=True):
            _body()
    else:
        with st.chat_message(chat_role):
            _body()


_OUTCOME_VARIANTS: dict[tuple[int | None, int | None], tuple[str, str, str]] = {
    # (has_error, ultimately_succeeded) -> (css_class, glyph, label)
    (0, 1): ("tm-outcome-success", "\u2713", "Success"),
    (1, 1): ("tm-outcome-recovered", "\u26a1", "Recovered failure"),
    (1, 0): ("tm-outcome-failed", "\u2717", "Failed"),
    # The interesting cell: process said "clean" (no rules fired) but
    # the outcome column says otherwise. Surface explicitly so the
    # operator can spot rule misses rather than burying it in "Unknown".
    (0, 0): ("tm-outcome-subtle", "\u26a0", "Subtle failure"),
}
"""(has_error, ultimately_succeeded) -> outcome chip variant. Anything
not in this map renders as the neutral "Unknown" variant -- that's the
fall-through for NULLs in either column."""


def _render_outcome_chip(trace_row: dict[str, Any]) -> None:
    """Render the slice-10 outcome chip + summary label caption.

    Single chip with one of 5 variants:

    * Success            (has_error=0, ultimately_succeeded=1)
    * Recovered failure  (has_error=1, ultimately_succeeded=1)
    * Failed             (has_error=1, ultimately_succeeded=0)
    * Subtle failure     (has_error=0, ultimately_succeeded=0)
    * Unknown            (NULL in either column)

    The summary label caption sits to the right of the chip when
    ``failure_label`` is populated, so the reader gets the one-word
    summary (e.g. "hallucinated_api") without the chip having to
    encode it.
    """
    has_error = trace_row.get("has_error")
    succeeded = trace_row.get("ultimately_succeeded")
    key = (
        int(has_error) if has_error is not None else None,
        int(succeeded) if succeeded is not None else None,
    )
    variant = _OUTCOME_VARIANTS.get(key)
    if variant is None:
        css_class, glyph, label = "tm-outcome-unknown", "\u003f", "Unknown"
    else:
        css_class, glyph, label = variant
    failure_label = trace_row.get("failure_label")
    caption = (
        f'<span class="tm-outcome-caption">Summary: <code>{_escape(failure_label)}</code></span>'
        if failure_label
        else ""
    )
    st.markdown(
        f'<div><span class="tm-outcome {css_class}">{glyph} {label}</span>'
        f"{caption}</div>",
        unsafe_allow_html=True,
    )


def render_header(trace_row: dict[str, Any]) -> None:
    """Render the trace-detail header strip.

    Shows the primary key (copy-friendly via ``st.code``), source format,
    agent/model badges, the slice-10 outcome chip, and the ingest
    timestamp. Pure presentation; never queries SQL.
    """
    st.subheader(f"Trace {trace_row.get('id', '?')}")
    st.code(trace_row.get("id", ""), language=None)

    badges = [
        ("Source", trace_row.get("source_format")),
        ("Agent", trace_row.get("agent_name")),
        ("Model", trace_row.get("model")),
        ("Steps", trace_row.get("num_steps")),
        ("Tool calls", trace_row.get("num_tool_calls")),
        ("Ingested", trace_row.get("ingested_at")),
    ]
    cols = st.columns(len(badges))
    for col, (k, v) in zip(cols, badges, strict=False):
        col.metric(label=k, value=str(v if v is not None else "-"))

    _render_outcome_chip(trace_row)

    # Slice 4 LLM-judge reasoning. Promoted from a quiet caption to a
    # full st.info() callout so the reasoning -- the whole point of the
    # judge pass -- doesn't get visually buried below the metric row.
    # Title-line + blockquote keeps the callout scannable even when the
    # reasoning is long; ``icon=":material/psychology:"`` reads as
    # "this is a model's interpretation" without needing words.
    reasoning = trace_row.get("failure_label_reasoning")
    has_error = trace_row.get("has_error")
    if isinstance(reasoning, str) and reasoning.strip():
        st.info(
            f"**Judge reasoning** (Sonnet 4.6)\n\n> {reasoning.strip()}",
            icon=":material/psychology:",
        )
    elif has_error == 0:
        # Clean success path: the judge only labels a 10% sample of
        # has_error=0 traces to keep batch costs down, so 90% of clean
        # successes legitimately don't have a label and that's not a
        # bug. Mention the sample rate explicitly so the user knows
        # the lever to pull if they want full coverage.
        st.caption(
            "_Not classified by the LLM judge._  This row is a clean "
            "success (`has_error=0`). The judge only labels a "
            "deterministic 10% sample of clean successes (seed 42) "
            "to keep batch costs down; re-run `scripts/judge_failures.py` "
            "with a larger sample rate to include it."
        )
    elif has_error == 1:
        # Failed path: this row SHOULD have been judged but wasn't.
        # Almost always means it was added (uploaded, auto-upload
        # daemon, or backfill) after the last batch run of
        # ``scripts/judge_failures.py``. Fresh uploads land with
        # ``failure_label = NULL`` by design -- per-upload Sonnet
        # calls would be 5-30 s of user-facing latency and unbounded
        # cost, so the judge stays a batch job. The next batch run
        # picks this row up via its existing ``WHERE failure_label
        # IS NULL`` gate.
        st.caption(
            "_Not classified by the LLM judge yet._  This row is "
            "flagged as a failure (`has_error=1`) but the Sonnet 4.6 "
            "judge hasn't seen it. The judge runs as a batch job "
            "(`scripts/judge_failures.py`) -- fresh uploads land with "
            "`failure_label = NULL` and get picked up on the next "
            "batch run via the script's existing `WHERE failure_label "
            "IS NULL` gate."
        )


_EVENT_LABEL_GLYPHS: dict[str, str] = {
    "loop": "\U0001f501",
    "tool_spamming": "\U0001f4a5",
    "gave_up": "\U0001f3f3",
    "incomplete": "\u23f8",
    "wrong_approach": "\U0001f9ed",
    "hallucinated_api": "\U0001f47b",
    "misread_error": "\U0001f50d",
    "environment_issue": "\U0001f527",
    "success": "\u2713",
    "partial_success": "\u00bd",
    "unclear": "\u003f",
}
"""Per-label glyph for the failure-events timeline. Falls back to a
generic warning triangle for unknown labels."""


def _render_one_event(event: dict[str, Any]) -> None:
    label = event.get("label") or "unknown"
    glyph = _EVENT_LABEL_GLYPHS.get(label, "\u26a0")
    step_start = event.get("step_start")
    step_end = event.get("step_end")
    if isinstance(step_start, int) and isinstance(step_end, int):
        if step_start == step_end:
            range_str = f"step {step_start}"
        else:
            range_str = f"steps {step_start}-{step_end}"
    else:
        range_str = "step ?"
    recovered = bool(event.get("recovered"))
    row_classes = "tm-event-row"
    if recovered:
        row_classes += " tm-event-recovered"
        recovery_step = event.get("recovery_step")
        if isinstance(recovery_step, int):
            status_html = (
                f'<span class="tm-event-status tm-event-status-recovered">'
                f"recovered at step {recovery_step}</span>"
            )
        else:
            status_html = (
                '<span class="tm-event-status tm-event-status-recovered">'
                "recovered</span>"
            )
    else:
        status_html = '<span class="tm-event-status">not recovered</span>'

    summary_html = ""
    summary = event.get("recovery_summary")
    if recovered and isinstance(summary, str) and summary.strip():
        summary_html = f'<div class="tm-event-summary">{_escape(summary.strip())}</div>'

    st.markdown(
        f'<div class="{row_classes}">'
        f'<span class="tm-event-label">{glyph} {_escape(label)}</span>'
        f'<span class="tm-event-range">{_escape(range_str)}</span>'
        f"&nbsp;&middot;&nbsp;{status_html}"
        f"{summary_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_failure_events_timeline(events_blob: str | None) -> None:
    """Render the slice-10 failure-events timeline.

    ``events_blob`` is the raw JSON string from the
    ``traces.failure_events`` column (or ``None`` when the extractor
    hasn't run yet). The renderer hides itself entirely on:

    * NULL / empty input,
    * malformed JSON,
    * empty events list (legitimate "clean trace" output).

    The detail view doesn't need to special-case any of these branches;
    it just calls this function unconditionally for every trace.
    """
    if not events_blob:
        return
    try:
        events = json.loads(events_blob)
    except json.JSONDecodeError:
        return
    if not isinstance(events, list) or not events:
        return
    st.markdown(
        '<div class="tm-recpanel-head">Failure events</div>',
        unsafe_allow_html=True,
    )
    for event in events:
        if isinstance(event, dict):
            _render_one_event(event)


def render_recovered_recommendations_panel(
    trace_row: dict[str, Any],
) -> None:
    """Render the slice-10 "Recoveries from similar failures" panel.

    Reads the pre-computed ``recovered_recommendations`` JSON column.
    Renders only when:

    * the trace itself failed (``has_error = 1``), AND
    * the column is populated with a non-empty list.

    Each row is a single ``<a>`` element so the whole card is one click
    target (mirroring the directory-card affordance from the list
    view). The card surfaces the matched trace's recovered events as
    quoted recovery summaries -- the concrete actionable advice users
    visit this panel to read.
    """
    if trace_row.get("has_error") != 1:
        return
    blob = trace_row.get("recovered_recommendations")
    if not blob:
        return
    try:
        recs = json.loads(blob)
    except json.JSONDecodeError:
        return
    if not isinstance(recs, list) or not recs:
        return

    st.markdown(
        f'<div class="tm-recpanel-head">'
        f"\U0001f4a1 Recoveries from similar failures "
        f"({len(recs)} trace{'s' if len(recs) != 1 else ''} hit similar "
        f"walls and got out)</div>",
        unsafe_allow_html=True,
    )
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        hit_id = rec.get("trace_id")
        if not isinstance(hit_id, str):
            continue
        similarity = rec.get("similarity")
        sim_str = (
            f"similarity {float(similarity):.2f}"
            if isinstance(similarity, (int, float))
            else ""
        )
        source = rec.get("source_format") or ""
        agent = rec.get("agent_name") or ""
        meta_bits = " &middot; ".join(b for b in (sim_str, source, agent) if b)
        recoveries = rec.get("recoveries") or []
        recoveries = [r for r in recoveries if isinstance(r, dict)]
        if not recoveries:
            continue
        blurb_text = (
            f"Hit similar issue, recovered ({len(recoveries)} events):"
            if len(recoveries) > 1
            else "Hit similar issue, recovered:"
        )
        quotes_html = "".join(
            f'<div class="tm-rec-quote">{_escape(r.get("summary") or "")}</div>'
            for r in recoveries
            if r.get("summary")
        )
        href = f"?trace_id={quote(str(hit_id), safe='')}"
        st.markdown(
            f'<a class="tm-rec-row" href="{href}" target="_self">'
            f'<div class="tm-rec-head">'
            f'<span class="tm-rec-id">{_escape(hit_id)}</span>'
            f'<span class="tm-rec-meta">{meta_bits}</span>'
            f"</div>"
            f'<div class="tm-rec-blurb">{blurb_text}</div>'
            f"{quotes_html}"
            f"</a>",
            unsafe_allow_html=True,
        )


def render_filters_summary(counts: dict[str, int]) -> None:
    """Sidebar header that tells the user how many rows live in the DB.

    Plain text on purpose -- this is meant to be a constant point of
    reference while the user fiddles with filter controls; ``st.metric``
    would compete with the actual filter widgets for visual weight.
    """
    st.markdown(
        f"**Total traces:** {counts.get('total', 0)}  \n"
        f"with ATIF: {counts.get('with_atif', 0)}  \n"
        f":red[errors:] {counts.get('errors', 0)}  ·  "
        f":green[successes:] {counts.get('successes', 0)}  ·  "
        f":gray[unknown:] {counts.get('unknown_outcome', 0)}"
    )


__all__ = [
    "extract_task_title",
    "inject_global_styles",
    "render_directory_row",
    "render_failure_events_timeline",
    "render_filters_summary",
    "render_header",
    "render_recovered_recommendations_panel",
    "render_step",
    "render_top_nav",
]
