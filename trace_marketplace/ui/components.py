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
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

SOURCE_LABELS: dict[str, str] = {
    "user": "User (human / task setup)",
    "agent": "Agent (the model under study)",
    "system": "System (harness / tooling)",
    "tool": "Tool (function result)",
}
"""Friendly labels for ATIF ``Step.source`` values. Unknown sources
render with the raw string in title-case.

The parenthetical disambiguators are deliberate: real users on the
deployed UI repeatedly read ``System`` as "the marketplace's system"
or "the LLM judge" rather than "the agent harness that fed this
prompt to the model under study". Naming the role concretely once
in the step header makes the conversation thread self-describing
without a separate legend.
"""


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
    timestamp = step.get("timestamp")
    header = f"**Step {index + 1} - {label}**"
    if timestamp:
        header += f"  \n*{timestamp}*"

    def _body() -> None:
        st.markdown(header)
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


__all__ = ["render_filters_summary", "render_header", "render_step"]
