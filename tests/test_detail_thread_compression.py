"""Regression test for detail-view thread compression on huge traces.

Streamlit Cloud rejected the full render of a 1169-step claude_code
trace with ``RangeError: index out of range: 99 + 5272104 > 3646309``
(~5 MB message vs ~3.5 MB websocket frame budget). The fix in
``detail_view._render_thread`` caps the inline render to first 30 +
last 100 steps for traces over ``_THREAD_RENDER_SOFT_CAP``; the
omitted middle is replaced by a placeholder block, and a button lets
the user opt into the full render at their own risk.

We pin two invariants:

* Traces under the soft cap render every step inline (no behavioural
  change for the common case).
* Traces over the cap call ``render_step`` exactly
  ``_THREAD_HEAD_COUNT + _THREAD_TAIL_COUNT`` times and skip the
  middle entirely -- otherwise we'd re-trip the original websocket
  overflow.

We use ``unittest.mock.patch`` to stub the actual Streamlit calls
(``st.button``, ``st.caption``, ``st.markdown``, ``st.session_state``,
``st.info``) and the imported ``render_step`` symbol, so we can count
invocations without spinning up Streamlit's runtime.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch


def _make_step(idx: int) -> dict[str, Any]:
    return {"source": "user" if idx % 2 == 0 else "agent", "message": f"step {idx}"}


def _make_atif(num_steps: int) -> dict[str, Any]:
    return {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "test-agent", "model_name": "test-model"},
        "steps": [_make_step(i) for i in range(num_steps)],
    }


def _render_with_stubs(num_steps: int, *, opt_in_full: bool = False) -> int:
    """Invoke ``_render_thread`` against a fake trace, return the number
    of ``render_step`` calls.

    ``opt_in_full=True`` simulates the user having clicked the "render
    all" button on this trace (the toggle is keyed by trace_id in
    ``st.session_state``).
    """
    fake_state: dict[str, Any] = {}
    if opt_in_full:
        # Module under test reads ``st.session_state.get(KEY) == trace_id``
        # to decide whether to render the full thread; populate it.
        from trace_marketplace.ui.detail_view import _THREAD_RENDER_ALL_KEY

        fake_state[_THREAD_RENDER_ALL_KEY] = "trace-under-test"

    render_step_mock = MagicMock()

    with (
        patch("trace_marketplace.ui.detail_view.render_step", render_step_mock),
        patch("trace_marketplace.ui.detail_view.st") as st_mock,
    ):
        # st.session_state must behave like a dict for .get(...) lookups.
        st_mock.session_state = fake_state
        # st.button returns False -> user didn't click "render all" this
        # turn (orthogonal to the pre-existing opt_in_full state which
        # is read via session_state).
        st_mock.button.return_value = False
        # Everything else (caption / info / markdown / rerun) is a no-op.
        # MagicMock will absorb the calls without complaint.

        from trace_marketplace.ui.detail_view import _render_thread

        _render_thread(_make_atif(num_steps), trace_id="trace-under-test")

    return render_step_mock.call_count


def test_short_trace_renders_every_step_inline() -> None:
    """Traces under the soft cap render in full -- no compression."""
    from trace_marketplace.ui.detail_view import _THREAD_RENDER_SOFT_CAP

    # One step below the cap -> render everything.
    n = _THREAD_RENDER_SOFT_CAP - 1
    assert _render_with_stubs(n) == n


def test_trace_at_soft_cap_still_renders_everything_inline() -> None:
    """Boundary: traces exactly at the cap still get the full render
    (the compression only kicks in *above* the cap)."""
    from trace_marketplace.ui.detail_view import _THREAD_RENDER_SOFT_CAP

    assert _render_with_stubs(_THREAD_RENDER_SOFT_CAP) == _THREAD_RENDER_SOFT_CAP


def test_huge_trace_renders_only_head_and_tail() -> None:
    """The original bug: a 1169-step trace must NOT render 1169 steps
    inline. The compressed render emits exactly head + tail steps."""
    from trace_marketplace.ui.detail_view import (
        _THREAD_HEAD_COUNT,
        _THREAD_TAIL_COUNT,
    )

    rendered = _render_with_stubs(1169)
    assert rendered == _THREAD_HEAD_COUNT + _THREAD_TAIL_COUNT


def test_huge_trace_with_user_opt_in_renders_everything() -> None:
    """When the user clicks "render all N steps inline" (we simulate
    by pre-populating the session_state slot), the compression yields
    to the explicit user choice and every step renders."""
    rendered = _render_with_stubs(1169, opt_in_full=True)
    assert rendered == 1169


def test_zero_step_trace_does_not_call_render_step() -> None:
    """Defensive: an empty steps list short-circuits with an info
    message and does NOT touch ``render_step``."""
    assert _render_with_stubs(0) == 0
