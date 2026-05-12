"""Tests for :mod:`trace_marketplace.search.text_extraction`.

The function under test is the input side of the slice 6 similarity
loop, so we hold it to four invariants:

1. **Determinism.** Same dict in -> identical bytes out, repeatable.
   This is what makes ``source_text_hash`` a valid idempotency key
   for the backfill script.
2. **Hard size cap.** The cap from the plan (4 KB default) must be
   respected even on adversarially long inputs.
3. **Label propagation.** When a ``failure_label`` is supplied it
   appears in the rendered blob -- this is the strongest semantic
   signal we feed into the embedding.
4. **Degradation, not crashes.** Empty / malformed trajectories
   must produce a plain string, never raise; the backfill script
   runs against 547+ rows and one weird record shouldn't tank it.
"""

from __future__ import annotations

from trace_marketplace.search.text_extraction import (
    DEFAULT_MAX_CHARS,
    extract_searchable_text,
)


def _make_step(idx: int, *, source: str, message: str, tool: str | None = None):
    step: dict = {
        "step_id": f"s{idx}",
        "source": source,
        "message": message,
    }
    if tool:
        step["tool_calls"] = [{"function_name": tool, "arguments": {}}]
        step["observation"] = {"results": [{"content": f"ran {tool} on step {idx}"}]}
    return step


def _make_atif(*, agent="claude-code", model="sonnet-4", steps=None):
    return {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": agent, "model_name": model},
        "steps": steps or [],
    }


def test_extract_searchable_text_is_deterministic():
    atif = _make_atif(
        steps=[
            _make_step(1, source="user", message="please fix the bug"),
            _make_step(2, source="agent", message="checking", tool="bash"),
            _make_step(3, source="agent", message="done"),
        ]
    )
    first = extract_searchable_text(atif, failure_label="loop")
    second = extract_searchable_text(atif, failure_label="loop")
    assert first == second
    # Sanity: header tokens we care about are present.
    assert "Agent: claude-code" in first
    assert "Model: sonnet-4" in first
    assert "Failure label: loop" in first
    assert "Tools used: bash" in first


def test_extract_searchable_text_respects_max_chars_cap():
    # 500 steps with 400-char messages each = ~200 KB raw input,
    # far above the 4 KB cap. The blob must still come back under it.
    huge_steps = [
        _make_step(i, source="agent", message="x" * 400, tool="bash")
        for i in range(1, 501)
    ]
    atif = _make_atif(steps=huge_steps)
    blob = extract_searchable_text(atif)
    assert len(blob) <= DEFAULT_MAX_CHARS
    # The header (agent / step count) is always preserved.
    assert "Agent: claude-code" in blob
    assert "Step count: 500" in blob


def test_extract_searchable_text_includes_label_and_reasoning():
    atif = _make_atif(
        steps=[
            _make_step(1, source="user", message="run the tests"),
            _make_step(2, source="agent", message="ok", tool="pytest"),
        ]
    )
    blob = extract_searchable_text(
        atif,
        failure_label="gave_up",
        failure_label_reasoning=(
            "Agent abandoned the task after 3 failed attempts "
            "and explicitly said 'I cannot help with this'."
        ),
    )
    assert "Failure label: gave_up" in blob
    assert "Label reasoning: Agent abandoned" in blob
    # The task line picks up the first non-agent message.
    assert "Task: run the tests" in blob


def test_extract_searchable_text_handles_empty_and_malformed_inputs():
    # No agent, no steps, no nothing -- still returns a non-empty
    # string and doesn't raise.
    blob = extract_searchable_text({})
    assert isinstance(blob, str)
    assert "Agent: unknown" in blob

    # Steps key is the wrong type.
    blob = extract_searchable_text({"agent": "not-a-dict", "steps": "wat"})
    assert "Agent: unknown" in blob
    assert "Conversation: (no steps)" in blob

    # Step dicts contain garbage entries mixed with valid ones.
    atif = _make_atif(
        steps=[
            "this is not a dict",  # type: ignore[list-item]
            {"step_id": "s1"},  # no source, no message
            _make_step(2, source="agent", message="ok"),
        ]
    )
    blob = extract_searchable_text(atif)
    assert "Agent: claude-code" in blob


def test_tool_names_collected_uniquely_in_header():
    atif = _make_atif(
        steps=[
            _make_step(1, source="agent", message="a", tool="bash"),
            _make_step(2, source="agent", message="b", tool="grep"),
            _make_step(3, source="agent", message="c", tool="bash"),
        ]
    )
    blob = extract_searchable_text(atif)
    # Order preserved, no duplicates.
    assert "Tools used: bash, grep" in blob
