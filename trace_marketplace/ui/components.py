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


def render_header(trace_row: dict[str, Any]) -> None:
    """Render the trace-detail header strip.

    Shows the primary key (copy-friendly via ``st.code``), source format,
    agent/model badges, failure flags, and the ingest timestamp. Pure
    presentation; never queries SQL.
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

    failure_bits: list[str] = []
    has_error = trace_row.get("has_error")
    if has_error == 1:
        failure_bits.append(":red[Labelled FAILURE (`has_error=1`)]")
    elif has_error == 0:
        failure_bits.append(":green[Labelled SUCCESS (`has_error=0`)]")
    else:
        failure_bits.append(":gray[Outcome label: unknown (`has_error IS NULL`)]")
    failure_label = trace_row.get("failure_label")
    if failure_label:
        failure_bits.append(f"Failure label: `{failure_label}`")
    st.markdown("  \n".join(failure_bits))

    # Slice 4 LLM-judge reasoning. Promoted from a quiet caption to a
    # full st.info() callout so the reasoning -- the whole point of the
    # judge pass -- doesn't get visually buried below the metric row.
    # Title-line + blockquote keeps the callout scannable even when the
    # reasoning is long; ``icon=":material/psychology:"`` reads as
    # "this is a model's interpretation" without needing words.
    reasoning = trace_row.get("failure_label_reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        st.info(
            f"**Judge reasoning** (Sonnet 4.6)\n\n> {reasoning.strip()}",
            icon=":material/psychology:",
        )
    elif has_error in (0, 1):
        # Tell the user *why* there's no judge output here so a clean
        # success doesn't read as "the feature is broken". The judge
        # only ran on all has_error=1 + a 10% sample of has_error=0,
        # so 91% of clean successes legitimately don't have a label.
        st.caption(
            "_Not classified by the LLM judge._  Only `has_error=1` "
            "traces plus a deterministic 10% sample of clean successes "
            "(seed 42) were sent to Sonnet 4.6 -- this row wasn't "
            "in the sample. Re-run `scripts/judge_failures.py` with a "
            "larger sample rate to include it."
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
    "render_filters_summary",
    "render_header",
    "render_step",
    "render_top_nav",
]
