"""Tests for :mod:`trace_marketplace.synth.prompts`.

Pin the *structural* contract with Sonnet:

1. Each format has its own static spec block.
2. The system message has a cache-control marker on the static spec
   (this is the optimisation the user is paying for at scale).
3. The tool schema enforces a string ``file_bytes`` field.
4. Per-trace briefs embed the task, primary failure, and all secondary
   failures.
5. ``extract_file_bytes`` round-trips a valid tool_use block.
6. ``extract_file_bytes`` returns None on wrong-tool / empty input.
"""

from __future__ import annotations

import json

import pytest

from trace_marketplace.synth.prompts import (
    FORMAT_SPECS,
    OUTPUT_TOOL_NAME,
    OUTPUT_TOOL_SCHEMA,
    build_prompt,
    dump_tool_schema_for_test,
    extract_file_bytes,
    extract_notes,
)
from trace_marketplace.synth.spec import (
    FORMAT_CLAUDE_CODE,
    FORMAT_CODEX,
    FORMAT_CURSOR,
    TraceSpec,
)


def _make_spec(fmt: str = FORMAT_CODEX, failures: tuple[str, ...] = ()) -> TraceSpec:
    return TraceSpec(
        output_format=fmt,
        primary_failure="loop",
        secondary_failures=failures,
        task="Fix the failing test in tests/test_api.py",
        target_step_count=25,
    )


def test_all_formats_have_specs() -> None:
    assert set(FORMAT_SPECS) == {FORMAT_CLAUDE_CODE, FORMAT_CODEX, FORMAT_CURSOR}
    for spec in FORMAT_SPECS.values():
        assert len(spec) > 100


def test_format_specs_are_distinct() -> None:
    """Sanity: the three format specs should not be byte-identical."""
    specs = list(FORMAT_SPECS.values())
    assert len({s for s in specs}) == 3


def test_build_prompt_marks_static_block_for_caching() -> None:
    built = build_prompt(_make_spec())
    cached = [b for b in built.system if b.get("cache_control")]
    assert len(cached) == 1
    assert cached[0]["cache_control"] == {"type": "ephemeral"}


def test_build_prompt_includes_format_spec_in_static_block() -> None:
    built = build_prompt(_make_spec(fmt=FORMAT_CLAUDE_CODE))
    static_block = next(b for b in built.system if b.get("cache_control"))
    assert "Claude Code session JSONL" in static_block["text"]


def test_build_prompt_brief_embeds_task_and_label() -> None:
    spec = _make_spec(failures=("misread_error",))
    built = build_prompt(spec)
    assert spec.task in built.user_brief
    assert "loop" in built.user_brief
    assert "misread_error" in built.user_brief


def test_build_prompt_brief_lists_no_secondary_when_empty() -> None:
    spec = _make_spec()
    built = build_prompt(spec)
    assert "No secondary failure modes" in built.user_brief


def test_build_prompt_rejects_unknown_format() -> None:
    with pytest.raises(ValueError):
        build_prompt(
            TraceSpec(
                output_format="atif",
                primary_failure="loop",
                task="x",
                target_step_count=5,
            )
        )


def test_tool_schema_requires_file_bytes_string() -> None:
    schema = OUTPUT_TOOL_SCHEMA["input_schema"]
    assert schema["properties"]["file_bytes"]["type"] == "string"
    assert "file_bytes" in schema["required"]


def test_extract_file_bytes_happy_path() -> None:
    block = {
        "type": "tool_use",
        "name": OUTPUT_TOOL_NAME,
        "input": {"file_bytes": "raw\nbytes\n"},
    }
    assert extract_file_bytes(block) == "raw\nbytes\n"


def test_extract_file_bytes_rejects_wrong_tool() -> None:
    block = {
        "type": "tool_use",
        "name": "some_other_tool",
        "input": {"file_bytes": "raw"},
    }
    assert extract_file_bytes(block) is None


def test_extract_file_bytes_rejects_empty_string() -> None:
    block = {
        "type": "tool_use",
        "name": OUTPUT_TOOL_NAME,
        "input": {"file_bytes": "   "},
    }
    assert extract_file_bytes(block) is None


def test_extract_file_bytes_rejects_non_tool_use_block() -> None:
    assert (
        extract_file_bytes({"type": "text", "text": "raw bytes"}) is None
    )


def test_extract_notes_round_trips() -> None:
    block = {
        "type": "tool_use",
        "name": OUTPUT_TOOL_NAME,
        "input": {"file_bytes": "x", "notes": "  loop trace with 7 reps  "},
    }
    assert extract_notes(block) == "loop trace with 7 reps"


def test_extract_notes_returns_none_when_missing() -> None:
    block = {
        "type": "tool_use",
        "name": OUTPUT_TOOL_NAME,
        "input": {"file_bytes": "x"},
    }
    assert extract_notes(block) is None


def test_dump_tool_schema_for_test_is_stable_json() -> None:
    rendered = dump_tool_schema_for_test()
    parsed = json.loads(rendered)
    assert parsed["name"] == OUTPUT_TOOL_NAME
