"""Per-format prompt + tool-schema builders for the Sonnet synth pipeline.

Each format gets its own prompt because the on-disk shapes are too
different to share one template (Claude Code JSONL with Anthropic
content-blocks, Codex Responses-API JSONL, Cursor's v1 envelope JSON).

The unified contract is the tool schema: regardless of format, Sonnet
returns its trace as one ``file_bytes`` string under the
:data:`OUTPUT_TOOL_SCHEMA` tool. The orchestrator strips that string,
saves it verbatim to disk under ``data/raw/<format>/`` with the
matching extension, and lets the existing ingest scripts pick it up
unchanged. No JSON parsing in the hot loop -- the adapter is the only
parser we trust.

Why "one tool with a string field" rather than a richly typed schema?

* Claude's tool-use mode guarantees valid JSON for the *tool args*,
  not for the inner string content. A typed schema would force the
  model to fill out our typing, then we'd still have to validate the
  emitted JSONL against the adapter. Net result: two failure
  surfaces instead of one.
* The string-only schema means we get one retry-able failure mode
  ("adapter didn't accept the bytes") plus a free escape hatch to
  iterate on prompts without touching schemas.

Prompt caching: the static portion of each prompt (format spec,
examples, behavioural guidance) is wrapped with
``cache_control={"type": "ephemeral"}`` so the first run pays the
input-token cost and subsequent runs in the same window get a >90%
discount on the cached prefix. With ~500 traces this is the
difference between $30 and $7 on the input side.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from trace_marketplace.enrich.failure_taxonomy import (
    LABEL_DESCRIPTIONS,
    FailureLabel,
)
from trace_marketplace.synth.spec import (
    FORMAT_CLAUDE_CODE,
    FORMAT_CODEX,
    FORMAT_CURSOR,
    TraceSpec,
)

# ---------------------------------------------------------------------------
# Anthropic tool schema -- one universal "submit raw bytes" tool.
# ---------------------------------------------------------------------------

OUTPUT_TOOL_NAME = "submit_trace"
"""Name Sonnet must use when emitting the trace. The validator looks
for ``tool_use`` blocks with this exact name; anything else is
treated as a parse failure and triggers a retry."""

OUTPUT_TOOL_SCHEMA: dict[str, Any] = {
    "name": OUTPUT_TOOL_NAME,
    "description": (
        "Submit the complete synthetic trace as a single string of "
        "file bytes (JSONL for Claude Code / Codex, JSON for Cursor). "
        "Do NOT include surrounding markdown fences, do NOT pretty-"
        "print -- emit the file exactly as it should land on disk."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_bytes": {
                "type": "string",
                "description": (
                    "The full file contents. For JSONL formats, "
                    "newline-separated JSON objects with no blank "
                    "lines and no trailing newline. For Cursor, a "
                    "single JSON object."
                ),
            },
            "notes": {
                "type": "string",
                "description": (
                    "(Optional) one-line summary of which failure "
                    "mode(s) this trace exhibits, for the dataset "
                    "manifest. Not embedded in the file itself."
                ),
            },
        },
        "required": ["file_bytes"],
    },
}


# ---------------------------------------------------------------------------
# Per-format format descriptions.
# ---------------------------------------------------------------------------
#
# These are the chunks marked as cache-friendly: same string every
# call for a given format, so prompt caching reuses them after the
# first request.

_CLAUDE_CODE_SPEC = """\
=== Output format: Claude Code session JSONL ===

A Claude Code session is a newline-delimited file. Each line is ONE
JSON object representing one event. Required event types in order:

1. First event: a `permission-mode` event (metadata, optional but
   recommended for realism):
   {"type":"permission-mode","sessionId":"<uuid>","cwd":"/Users/dev/project","mode":"acceptEdits","timestamp":"2026-05-11T10:00:00.000Z","uuid":"<uuid>"}

2. A `user` event with the initial task. ``message.content`` is a
   STRING for the initial human turn:
   {"type":"user","sessionId":"<uuid>","timestamp":"2026-05-11T10:00:01.000Z","uuid":"<uuid>","message":{"role":"user","content":"Fix the failing test in tests/test_api.py"}}

3. Then alternating turns. An `assistant` event carries one model
   turn; ``message.content`` is an ARRAY of typed blocks:
     - {"type":"text","text":"I'll start by looking at the test file."}
     - {"type":"thinking","thinking":"The error suggests the import is broken..."}  (optional)
     - {"type":"tool_use","id":"toolu_01abc...","name":"Bash","input":{"command":"cat tests/test_api.py"}}

   Example assistant event:
   {"type":"assistant","sessionId":"<uuid>","timestamp":"2026-05-11T10:00:02.500Z","uuid":"<uuid>","message":{"id":"msg_01...","role":"assistant","model":"claude-sonnet-4-5","content":[{"type":"text","text":"Let me read it"},{"type":"tool_use","id":"toolu_01abc","name":"Bash","input":{"command":"cat tests/test_api.py"}}],"usage":{"input_tokens":1200,"output_tokens":80}}}

4. A `user` event whose ``message.content`` is an ARRAY of
   ``tool_result`` blocks responding to the prior assistant turn.
   The ``tool_use_id`` MUST match the prior ``tool_use.id`` exactly:
   {"type":"user","sessionId":"<uuid>","timestamp":"2026-05-11T10:00:03.000Z","uuid":"<uuid>","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_01abc","content":"def test_api():\\n    response = client.get('/api')\\n    assert response.status_code == 200"}]}}

5. Loop steps 3-4 until the agent completes (or, for failing traces,
   gives up / loops / etc).

Tool names commonly used by Claude Code: ``Bash``, ``Read``, ``Edit``,
``Write``, ``Grep``, ``Glob``, ``TodoWrite``, ``WebFetch``,
``WebSearch``. Use these realistically.

RULES (the adapter will reject the file otherwise):
- One JSON object per line; no blank lines; no trailing comma.
- Every ``tool_use.id`` MUST be referenced by exactly one
  ``tool_result.tool_use_id`` later in the file.
- Timestamps must be ISO 8601 strings; advance them by 0.5-30s per
  step for realism.
- ``sessionId`` and per-event ``uuid`` should be valid UUIDv4-ish
  strings (8-4-4-4-12 hex layout) and be UNIQUE per event.
"""

_CODEX_SPEC = """\
=== Output format: Codex (OpenAI codex CLI) session JSONL ===

A Codex session is a newline-delimited file in the OpenAI Responses
API event-stream shape. Required structure:

1. First line: ``session_meta`` event with payload identity.
   {"type":"session_meta","payload":{"id":"019f...","timestamp":"2026-05-11T10:00:00.000Z","cli_version":"0.20.0","originator":"codex_cli","cwd":"/Users/dev/project","instructions":""}}

2. Second line: ``turn_context`` with the model:
   {"type":"turn_context","payload":{"cwd":"/Users/dev/project","approval_policy":"on-request","sandbox_policy":{"mode":"workspace-write"},"model":"gpt-5.5","summary":"auto"}}

3. Then a ``response_item`` for the initial user message:
   {"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Fix the failing test in tests/test_api.py"}]}}

4. Then assistant-side `response_item`s in this rough order:
   - reasoning (optional):
     {"type":"response_item","payload":{"type":"reasoning","summary":[{"type":"summary_text","text":"I should first read the test file..."}]}}
   - message (model talking to user):
     {"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Let me read the file."}]}}
   - function_call (tool invocation):
     {"type":"response_item","payload":{"type":"function_call","name":"shell","call_id":"call_abc123","arguments":"{\\"command\\":[\\"cat\\",\\"tests/test_api.py\\"]}"}}
   - function_call_output (tool response, MUST match the prior
     ``call_id``):
     {"type":"response_item","payload":{"type":"function_call_output","call_id":"call_abc123","output":"{\\"output\\":\\"def test_api():\\\\n    ...\\",\\"metadata\\":{\\"exit_code\\":0,\\"duration_seconds\\":0.05}}"}}

5. Loop function_call / function_call_output / message until done.

6. Final line: token usage:
   {"type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":5000,"output_tokens":1200,"cached_input_tokens":2000}}}}

Tool names commonly used by Codex: ``shell`` (the dominant one,
takes a ``command`` array), ``apply_patch``, ``view``. Use these
realistically.

RULES:
- One JSON object per line.
- ``function_call.arguments`` and ``function_call_output.output`` are
  themselves JSON-encoded STRINGS, not raw objects. (Note the
  embedded ``\\"`` escapes in the examples above.)
- ``call_id`` MUST be unique per call and match its paired output.
- No timestamps required on response_items (Codex doesn't emit them).
"""

_CURSOR_SPEC = """\
=== Output format: Cursor v1 export envelope (single JSON object) ===

Cursor's v1 export is ONE JSON object (not JSONL). Shape:

{
  "session_id": "<uuid>",
  "cursor_version": "0.45.0",
  "workspace_path": "/Users/dev/project",
  "model": "claude-3.5-sonnet",
  "synthetic": true,
  "synthetic_failure_mode": "<primary_failure_label>",
  "messages": [
    {"role": "user", "content": "Fix the failing test", "timestamp": "2026-05-11T10:00:00Z"},
    {
      "role": "assistant",
      "content": "Let me read it first.",
      "timestamp": "2026-05-11T10:00:01Z",
      "tool_calls": [
        {"id": "tc_1", "name": "read_file", "arguments": {"path": "tests/test_api.py"}}
      ]
    },
    {
      "role": "user",
      "content": "",
      "tool_results": [
        {"tool_call_id": "tc_1", "content": "def test_api():\\n    ..."}
      ]
    }
  ]
}

Tool names commonly used by Cursor: ``read_file``, ``edit_file``,
``codebase_search``, ``grep_search``, ``run_terminal_cmd``,
``list_dir``, ``file_search``. Use these realistically.

RULES:
- Output is ONE JSON object (not JSONL). Single line is fine; pretty-
  print also fine -- the adapter handles both.
- Every assistant ``tool_calls[i].id`` MUST be referenced by exactly
  one ``tool_results[j].tool_call_id`` in a later message.
- ``synthetic`` must be ``true`` and ``synthetic_failure_mode`` must
  be set to the primary failure label assigned in the brief.
"""

FORMAT_SPECS: dict[str, str] = {
    FORMAT_CLAUDE_CODE: _CLAUDE_CODE_SPEC,
    FORMAT_CODEX: _CODEX_SPEC,
    FORMAT_CURSOR: _CURSOR_SPEC,
}


# ---------------------------------------------------------------------------
# Failure-mode guidance.
# ---------------------------------------------------------------------------

_BEHAVIOURAL_GUIDANCE = """\
=== How to act out a given failure mode ===

You are playing the part of a coding agent whose trajectory exhibits
specific failures. The trajectory must look organic -- the failure
should emerge from realistic-looking mistakes, not from sentences
that say "I am now exhibiting LOOP behaviour". Reviewers will read
the trace and reason about what went wrong; if the failure mode is
hard to spot, that's fine. If it's labelled like a Halloween costume,
that's bad.

Per-label playbooks:

- loop: Repeat the same tool call (or trivial variants) 6+ times.
  The agent keeps getting the same unhelpful output and keeps trying
  it again. Don't show learning between attempts.
- tool_spamming: Many (15+) low-value tool calls. The agent thrashes
  -- reads 10 unrelated files, greps for various unrelated symbols,
  never zeroes in.
- gave_up: The agent eventually says something like "I don't think I
  can fix this", "you may want to consult a human", "this exceeds my
  capabilities", and STOPS calling tools. The final message is a
  resignation.
- incomplete: The session ends mid-action: an unfinished tool call,
  a truncated message ending in "...", or an obvious cliff. Do NOT
  add a wrap-up message after the cut-off.
- wrong_approach: The agent confidently pursues a plan that won't
  work (e.g. user wanted async/await refactor, agent does
  threading.Thread instead; user wanted SQL fix, agent edits the
  ORM model). The agent never realises.
- hallucinated_api: The agent calls a function / imports a module /
  reads a file that DOESN'T EXIST in the project. The tool output
  reflects this (ModuleNotFoundError, AttributeError, file not
  found, etc.) but the agent doesn't catch on -- it tries another
  imaginary thing.
- misread_error: The error message says X (e.g. "ImportError: cannot
  import name 'foo'"); the agent treats it as Y (e.g. installs a
  package, when the actual fix is renaming an attribute). The agent
  acts on its misreading.
- environment_issue: Tools return realistic infrastructure failures
  -- ECONNRESET, "permission denied", network timeouts,
  ``pip install`` failing because PyPI is down, docker daemon not
  running, etc. The agent isn't at fault but can't proceed.
- unclear: The session is short / cryptic / contains so many
  intermixed signals that a human couldn't say what went wrong. Use
  this label sparingly -- 2-3 ambiguous tool calls and a vague
  final message.
- success: Agent solves the task cleanly. Reads relevant files,
  makes the right edit, verifies (runs tests / re-greps), reports
  done. Should be the most plausible "good agent" you can produce.
- partial_success: Agent makes real progress but stops short. E.g.
  fixes the main bug but doesn't add the requested test; addresses
  3 of 4 review comments; refactors module A but leaves module B
  untouched. The final message should honestly acknowledge what
  wasn't completed.

For multi-mode failures (when secondary failures are listed in the
brief), weave them in. Example: a trace tagged ``misread_error +
loop`` might misread an error early, then loop on the wrong fix
repeatedly. Make the modes related, not stacked.

Variety guidance:
- Use realistic file paths matching the task (``src/`` / ``tests/`` /
  ``app/`` / ``api/``). Don't always use ``foo.py``.
- Errors should be specific (``AttributeError: 'NoneType' object has
  no attribute 'get'``, not ``Error occurred``).
- Tool outputs should look like real ``stdout``/``stderr`` (multi-
  line, mixed warnings + content).
- Avoid lazy filler like ``"...output truncated..."`` everywhere --
  emit actual content most of the time.
"""

_GENERAL_QUALITY_RULES = """\
=== Quality bar ===

- Aim for the step count given in the brief. Trajectories that are
  way too short (< 5 steps) or way too long (> 60 steps) will be
  rejected.
- Always include at least one assistant turn with tool_calls. A
  no-tool-call trace is degenerate.
- Tool arguments must look usable -- a ``Read`` should have a real
  path; a ``Bash`` should have a real command; an ``edit_file``
  should show a believable diff snippet.
- Each tool_result content should be 50-1500 characters of plausible
  output for that command. Don't return empty strings.
- All cross-references (tool_call_id <-> tool_use_id, call_id <->
  call_id) MUST line up. Mismatches will fail validation and burn a
  retry.
- Output the file ONLY through the ``submit_trace`` tool; do not
  write any prose response.
"""


# ---------------------------------------------------------------------------
# Brief construction.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuiltPrompt:
    """Bundle returned by :func:`build_prompt`.

    Fields:

    * ``system``: list of system-content blocks, the last of which is
      marked with ``cache_control={"type": "ephemeral"}`` so prompt
      caching kicks in for the static format spec.
    * ``user_brief``: a short per-trace instruction string (the
      non-cacheable, varying part of the prompt).
    * ``tools``: the single-tool list the orchestrator passes to
      ``messages.create``.
    """

    system: list[dict[str, Any]]
    user_brief: str
    tools: list[dict[str, Any]]


def _label_doc(label: str) -> str:
    """Return the human description for ``label`` or a fallback.

    Used in per-trace briefs so Sonnet knows what each label *means*
    without us re-typing the taxonomy on every call.
    """
    try:
        member = FailureLabel(label)
    except ValueError:
        return label
    return LABEL_DESCRIPTIONS.get(member, label)


def _format_failure_modes(spec: TraceSpec) -> str:
    """Render the spec's failure modes as a bullet list for the brief."""
    lines = [f"- Primary failure mode: **{spec.primary_failure}** -- {_label_doc(spec.primary_failure)}"]
    for sec in spec.secondary_failures:
        lines.append(f"- Secondary failure mode: **{sec}** -- {_label_doc(sec)}")
    if not spec.secondary_failures and spec.is_failure:
        lines.append("- No secondary failure modes (single-mode failure).")
    return "\n".join(lines)


def _build_user_brief(spec: TraceSpec) -> str:
    """One-trace task brief -- the only per-call varying portion."""
    if spec.is_failure:
        intent = "exhibits the failure mode(s) below"
    elif spec.primary_failure == FailureLabel.PARTIAL_SUCCESS.value:
        intent = "shows partial success (some progress, task NOT fully completed)"
    else:
        intent = "shows a clean successful completion"
    return (
        f"Generate a synthetic agent trace in **{spec.output_format}** format.\n\n"
        f"Task the agent was asked to perform:\n"
        f"  > {spec.task}\n\n"
        f"This trace {intent}.\n\n"
        f"Failure-mode brief:\n"
        f"{_format_failure_modes(spec)}\n\n"
        f"Target around **{spec.target_step_count}** trajectory steps "
        f"(model turns + tool calls combined). Submit ONLY via the "
        f"``{OUTPUT_TOOL_NAME}`` tool."
    )


def build_prompt(spec: TraceSpec) -> BuiltPrompt:
    """Assemble the full prompt + tool schema for one :class:`TraceSpec`.

    Layout:
    * ``system[0]``: short role-setting message.
    * ``system[1]``: format spec for ``spec.output_format`` +
      behavioural guidance + quality bar -- marked
      ``cache_control={"type": "ephemeral"}`` so subsequent calls in
      the same window pay only the cache-read price (~10x cheaper).
    * ``user_brief``: per-trace brief (task topic, failure modes,
      target step count).
    * ``tools``: single ``submit_trace`` tool.

    The orchestrator is responsible for actually wiring these into
    ``client.messages.create``; we keep the assembly side-effect
    free here so tests can assert on the structure.
    """
    if spec.output_format not in FORMAT_SPECS:
        raise ValueError(
            f"Unknown output format: {spec.output_format!r}. "
            f"Expected one of {sorted(FORMAT_SPECS)}."
        )

    static_spec = (
        FORMAT_SPECS[spec.output_format]
        + "\n\n"
        + _BEHAVIOURAL_GUIDANCE
        + "\n\n"
        + _GENERAL_QUALITY_RULES
    )

    system_blocks: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "You are a synthetic-data engine that generates realistic "
                "coding-agent trajectories for evaluating failure-mode "
                "search. You write traces in the exact on-disk format the "
                "downstream adapter expects."
            ),
        },
        {
            "type": "text",
            "text": static_spec,
            "cache_control": {"type": "ephemeral"},
        },
    ]

    return BuiltPrompt(
        system=system_blocks,
        user_brief=_build_user_brief(spec),
        tools=[OUTPUT_TOOL_SCHEMA],
    )


def extract_file_bytes(tool_use_block: dict[str, Any]) -> str | None:
    """Pull the ``file_bytes`` field out of a Claude ``tool_use`` block.

    Returns ``None`` if the block isn't the expected ``submit_trace``
    tool or the field is missing/empty. The orchestrator turns that
    None into a "model didn't emit usable output" retry signal.
    """
    if not isinstance(tool_use_block, dict):
        return None
    if tool_use_block.get("type") != "tool_use":
        return None
    if tool_use_block.get("name") != OUTPUT_TOOL_NAME:
        return None
    raw_input = tool_use_block.get("input")
    if not isinstance(raw_input, dict):
        return None
    bytes_val = raw_input.get("file_bytes")
    if isinstance(bytes_val, str) and bytes_val.strip():
        return bytes_val
    return None


def extract_notes(tool_use_block: dict[str, Any]) -> str | None:
    """Pull the optional ``notes`` field out of a ``tool_use`` block.

    Used for the dataset manifest only -- never persisted alongside
    the trace file itself. Falsy values return ``None``.
    """
    if not isinstance(tool_use_block, dict):
        return None
    raw_input = tool_use_block.get("input")
    if not isinstance(raw_input, dict):
        return None
    notes = raw_input.get("notes")
    if isinstance(notes, str) and notes.strip():
        return notes.strip()
    return None


def dump_tool_schema_for_test() -> str:
    """Return a stable JSON dump of the tool schema for snapshot tests."""
    return json.dumps(OUTPUT_TOOL_SCHEMA, sort_keys=True, indent=2)


__all__ = [
    "BuiltPrompt",
    "FORMAT_SPECS",
    "OUTPUT_TOOL_NAME",
    "OUTPUT_TOOL_SCHEMA",
    "build_prompt",
    "dump_tool_schema_for_test",
    "extract_file_bytes",
    "extract_notes",
]
