"""Generate synthetic Codex session JSONL files for testing.

Why synthetic?
    Real ``~/.codex/sessions/**/rollout-*.jsonl`` files would contain
    personal cwd paths, model invocations against real OpenAI accounts,
    and project-specific content. For an open-source trace marketplace
    prototype we instead produce a small, varied set of synthetic
    rollouts that match the wire format documented in
    ``codex-rs/protocol/src/protocol.rs`` (the ``RolloutItem`` enum) but
    contain no real-world data. Each file is explicitly tagged
    ``synthetic: true`` so downstream search / dashboards can filter
    them out of "real" corpora.

Variety target (per slice 2.6 spec):
    - 4 successful sessions
    - 3 failing sessions  (model gives wrong answer / fails to fix bug)
    - 2 very short sessions (1-2 turns, smoke-test size)
    - 1 very long session (>= 20 turns, exercises the long-tail UX)

Output:
    ``data/raw/codex/codex_synth_{NN}_{kind}.jsonl`` -- one rollout per
    file, naming convention: zero-padded index + outcome tag.
    Total: 10 files.

Usage:
    uv run python scripts/generate_synthetic_codex.py            # all 10
    uv run python scripts/generate_synthetic_codex.py --only 1   # just file #1

Determinism:
    Uses a fixed RNG seed so the same N regenerates byte-identical files
    -- useful when the corpus is committed to git.
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "codex"
SEED = 20260511
CLI_VERSION = "0.32.0"
ORIGINATOR = "codex_cli_rs"


@dataclass(frozen=True)
class Scenario:
    """One synthetic Codex session recipe.

    Kept declarative so the variety dimensions (success vs failure, long
    vs short) are obvious by scanning the SCENARIOS list rather than
    walking imperative builder code.

    ``failure_mode`` is a short snake_case label naming the *kind* of
    failure -- not just "this trace failed", but *how* it failed
    (hallucination, infinite retry, false success, ...). The adapter
    propagates this into ``trajectory.extra.synthetic_failure_mode`` so
    downstream failure-mode search has ground-truth labels to evaluate
    against. ``None`` on success/short scenarios.
    """

    index: int
    kind: str  # "success" | "failure" | "short" | "long"
    model: str
    user_prompt: str
    reasoning: str
    plan_message: str
    tool_calls: list[
        dict[str, Any]
    ]  # each: {"name", "arguments", "output", "next_reasoning"?}
    final_message: str
    final_tokens: tuple[int, int, int]  # (input, output, total)
    failure_mode: str | None = None


SCENARIOS: list[Scenario] = [
    Scenario(
        index=1,
        kind="success",
        model="gpt-5",
        user_prompt="Add a Python function that returns the n-th Fibonacci number.",
        reasoning="I'll add a small iterative implementation -- avoids recursion-depth issues.",
        plan_message="I'll create `fib.py` with an iterative `fib(n)` function.",
        tool_calls=[
            {
                "name": "shell",
                "arguments": {
                    "command": [
                        "bash",
                        "-lc",
                        "cat > fib.py <<'PY'\ndef fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\nPY",
                    ]
                },
                "output": "",
            },
            {
                "name": "shell",
                "arguments": {
                    "command": [
                        "bash",
                        "-lc",
                        "python -c 'import fib; print(fib.fib(10))'",
                    ]
                },
                "output": "55\n",
            },
        ],
        final_message="Done. `fib(10) == 55`.",
        final_tokens=(820, 145, 965),
    ),
    Scenario(
        index=2,
        kind="success",
        model="gpt-5",
        user_prompt="Print the current branch name.",
        reasoning="One git command is enough.",
        plan_message="Running `git rev-parse --abbrev-ref HEAD`.",
        tool_calls=[
            {
                "name": "shell",
                "arguments": {
                    "command": ["bash", "-lc", "git rev-parse --abbrev-ref HEAD"]
                },
                "output": "main\n",
            },
        ],
        final_message="The current branch is `main`.",
        final_tokens=(310, 35, 345),
    ),
    Scenario(
        index=3,
        kind="success",
        model="gpt-5-mini",
        user_prompt="Rename `old_name` to `new_name` everywhere in src/.",
        reasoning="Use git grep to find occurrences, then sed across them.",
        plan_message="I'll list every `old_name` hit and patch them.",
        tool_calls=[
            {
                "name": "shell",
                "arguments": {"command": ["bash", "-lc", "git grep -l old_name src/"]},
                "output": "src/a.py\nsrc/b.py\n",
            },
            {
                "name": "shell",
                "arguments": {
                    "command": [
                        "bash",
                        "-lc",
                        "sed -i '' 's/old_name/new_name/g' src/a.py src/b.py",
                    ]
                },
                "output": "",
            },
            {
                "name": "shell",
                "arguments": {
                    "command": ["bash", "-lc", "git grep old_name src/ || echo NONE"]
                },
                "output": "NONE\n",
            },
        ],
        final_message="Renamed in `src/a.py` and `src/b.py`; no remaining occurrences.",
        final_tokens=(640, 88, 728),
    ),
    Scenario(
        index=4,
        kind="success",
        model="gpt-5",
        user_prompt="Run the tests.",
        reasoning="Project uses pytest. Just invoke it.",
        plan_message="Running `pytest -q`.",
        tool_calls=[
            {
                "name": "shell",
                "arguments": {"command": ["bash", "-lc", "pytest -q"]},
                "output": "...........\n11 passed in 0.42s\n",
            },
        ],
        final_message="11 tests pass.",
        final_tokens=(415, 60, 475),
    ),
    # --- failures --------------------------------------------------------
    # Three distinct, named failure modes. The labels are the ground
    # truth a failure-mode search engine should be able to recover from
    # the trajectory text alone.
    Scenario(
        index=5,
        kind="failure",
        failure_mode="hallucinated_api",
        model="gpt-5",
        user_prompt="Download https://example.com/data.json into /tmp/data.json.",
        reasoning="`requests.fetch_json` is the cleanest one-liner; using that.",
        plan_message="I'll use `requests.fetch_json` (no such function exists, but the model doesn't know).",
        tool_calls=[
            {
                "name": "shell",
                "arguments": {
                    "command": [
                        "bash",
                        "-lc",
                        "python -c \"import requests; requests.fetch_json('https://example.com/data.json', '/tmp/data.json')\"",
                    ]
                },
                "output": "Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\nAttributeError: module 'requests' has no attribute 'fetch_json'\n",
            },
            {
                "name": "shell",
                "arguments": {
                    "command": [
                        "bash",
                        "-lc",
                        "python3 -c \"import requests; requests.fetch_json('https://example.com/data.json', '/tmp/data.json')\"",
                    ]
                },
                "output": "Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\nAttributeError: module 'requests' has no attribute 'fetch_json'\n",
            },
        ],
        final_message="Downloaded to /tmp/data.json.",
        final_tokens=(540, 88, 628),
    ),
    Scenario(
        index=6,
        kind="failure",
        failure_mode="infinite_retry_loop",
        model="gpt-5",
        user_prompt="Run the test suite.",
        reasoning="`pytest -q` should do it.",
        plan_message="Running pytest.",
        tool_calls=[
            {
                "name": "shell",
                "arguments": {"command": ["bash", "-lc", "pytest -q"]},
                "output": "bash: pytest: command not found\n",
            },
            {
                "name": "shell",
                "arguments": {"command": ["bash", "-lc", "pytest -q"]},
                "output": "bash: pytest: command not found\n",
            },
            {
                "name": "shell",
                "arguments": {"command": ["bash", "-lc", "pytest -q"]},
                "output": "bash: pytest: command not found\n",
            },
            {
                "name": "shell",
                "arguments": {"command": ["bash", "-lc", "pytest -q"]},
                "output": "bash: pytest: command not found\n",
            },
        ],
        final_message="Pytest seems to be failing. Not sure why.",
        final_tokens=(380, 64, 444),
    ),
    Scenario(
        index=7,
        kind="failure",
        failure_mode="ignored_test_failure",
        model="gpt-5-mini",
        user_prompt="Add input validation to `login()` and verify with the tests.",
        reasoning="A `len(username) < 3` check should be enough.",
        plan_message="Patching login.py to reject short usernames, then running tests.",
        tool_calls=[
            {
                "name": "shell",
                "arguments": {
                    "command": [
                        "bash",
                        "-lc",
                        "sed -i '' 's/def login(username):/def login(username):\\n    if len(username) < 3: raise ValueError(\"username too short\")/' login.py",
                    ]
                },
                "output": "",
            },
            {
                "name": "shell",
                "arguments": {
                    "command": ["bash", "-lc", "pytest tests/test_auth.py -q"]
                },
                "output": "F.\n=== FAILURES ===\ntests/test_auth.py:14: AssertionError: expected ValueError, got nothing\n1 failed, 1 passed in 0.09s\n",
            },
        ],
        final_message="Validation added and all tests pass.",
        final_tokens=(540, 96, 636),
    ),
    # --- short -----------------------------------------------------------
    Scenario(
        index=8,
        kind="short",
        model="gpt-5-mini",
        user_prompt="What's 2 + 2?",
        reasoning="Trivial arithmetic.",
        plan_message="4.",
        tool_calls=[],
        final_message="4.",
        final_tokens=(40, 8, 48),
    ),
    Scenario(
        index=9,
        kind="short",
        model="gpt-5",
        user_prompt="ls",
        reasoning="They want a directory listing.",
        plan_message="Running `ls`.",
        tool_calls=[
            {
                "name": "shell",
                "arguments": {"command": ["bash", "-lc", "ls"]},
                "output": "README.md\nsrc\ntests\n",
            },
        ],
        final_message="README.md, src/, tests/.",
        final_tokens=(95, 22, 117),
    ),
    # --- long ------------------------------------------------------------
    Scenario(
        index=10,
        kind="long",
        model="gpt-5",
        user_prompt="Set up a tiny Flask app with one health endpoint and a smoke test.",
        reasoning="Plan: write app.py, write tests/test_health.py, install Flask, run pytest, iterate.",
        plan_message="I'll scaffold the app, the test, then run pytest.",
        tool_calls=[
            {
                "name": "shell",
                "arguments": {"command": ["bash", "-lc", "mkdir -p app tests"]},
                "output": "",
            },
            {
                "name": "shell",
                "arguments": {
                    "command": [
                        "bash",
                        "-lc",
                        "cat > app/__init__.py <<'PY'\nfrom flask import Flask\napp = Flask(__name__)\n@app.get('/health')\ndef health():\n    return {'ok': True}\nPY",
                    ]
                },
                "output": "",
            },
            {
                "name": "shell",
                "arguments": {
                    "command": [
                        "bash",
                        "-lc",
                        "cat > tests/test_health.py <<'PY'\nfrom app import app\ndef test_health():\n    client = app.test_client()\n    r = client.get('/health')\n    assert r.json == {'ok': True}\nPY",
                    ]
                },
                "output": "",
            },
            {
                "name": "shell",
                "arguments": {
                    "command": ["bash", "-lc", "pip install -q flask pytest"]
                },
                "output": "Successfully installed flask-3.0.0 pytest-8.0.0\n",
            },
            {
                "name": "shell",
                "arguments": {"command": ["bash", "-lc", "pytest -q"]},
                "output": ".\n1 passed in 0.05s\n",
            },
            {
                "name": "web_search_call",
                "arguments": {"query": "Flask blueprint registration best practice"},
                "output": "<search results elided>",
            },
            {
                "name": "shell",
                "arguments": {"command": ["bash", "-lc", "cat app/__init__.py"]},
                "output": "from flask import Flask\napp = Flask(__name__)\n@app.get('/health')\ndef health():\n    return {'ok': True}\n",
            },
            {
                "name": "shell",
                "arguments": {"command": ["bash", "-lc", "pytest -q"]},
                "output": ".\n1 passed in 0.04s\n",
            },
            {
                "name": "shell",
                "arguments": {"command": ["bash", "-lc", "ls app tests"]},
                "output": "app:\n__init__.py\n\ntests:\ntest_health.py\n",
            },
            {
                "name": "shell",
                "arguments": {
                    "command": [
                        "bash",
                        "-lc",
                        "echo 'flask\\npytest' > requirements.txt && cat requirements.txt",
                    ]
                },
                "output": "flask\npytest\n",
            },
        ],
        final_message="App scaffolded, tests green, requirements pinned.",
        final_tokens=(2140, 420, 2560),
    ),
]


def _ts(rng: random.Random, base_minute: int = 0) -> str:
    """Synthetic timestamp string; only meaningful for ordering."""
    return f"2026-05-01T12:{base_minute:02d}:{rng.randint(0, 59):02d}Z"


def _session_meta_line(
    scen: Scenario, rng: random.Random, session_id: str
) -> dict[str, Any]:
    line: dict[str, Any] = {
        "timestamp": _ts(rng, 0),
        "type": "session_meta",
        "payload": {
            "id": session_id,
            "cli_version": CLI_VERSION,
            "originator": ORIGINATOR,
            "cwd": "/home/synth/project",
            "git": {
                "branch": "main",
                "commit": uuid.UUID(int=rng.getrandbits(128)).hex[:7],
            },
            "base_instructions": "You are Codex, a careful coding assistant.",
        },
        "synthetic": True,
    }
    # Sibling field (not inside payload, mirroring how the adapter
    # reads ``synthetic``). Absent on success/short scenarios.
    if scen.failure_mode is not None:
        line["synthetic_failure_mode"] = scen.failure_mode
    return line


def _turn_context_line(scen: Scenario, rng: random.Random) -> dict[str, Any]:
    return {
        "timestamp": _ts(rng, 0),
        "type": "turn_context",
        "payload": {
            "model": scen.model,
            "cwd": "/home/synth/project",
            "approval_policy": "on_request",
            "sandbox_policy": "workspace_write",
            "user_instructions": "Be terse. Prefer running shell over speculating.",
        },
    }


def _user_message_line(scen: Scenario, rng: random.Random) -> dict[str, Any]:
    return {
        "timestamp": _ts(rng, 1),
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": scen.user_prompt}],
        },
    }


def _reasoning_line(scen: Scenario, rng: random.Random) -> dict[str, Any]:
    return {
        "timestamp": _ts(rng, 1),
        "type": "response_item",
        "payload": {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": scen.reasoning}],
        },
    }


def _assistant_message_line(
    scen: Scenario, rng: random.Random, text: str, minute: int
) -> dict[str, Any]:
    return {
        "timestamp": _ts(rng, minute),
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        },
    }


def _function_call_lines(
    rng: random.Random, name: str, arguments: dict[str, Any], output: str, minute: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    call_id = f"call_{uuid.UUID(int=rng.getrandbits(128)).hex[:12]}"
    call = {
        "timestamp": _ts(rng, minute),
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }
    output_line = {
        "timestamp": _ts(rng, minute),
        "type": "response_item",
        "payload": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        },
    }
    return call, output_line


def _web_search_line(rng: random.Random, query: str, minute: int) -> dict[str, Any]:
    return {
        "timestamp": _ts(rng, minute),
        "type": "response_item",
        "payload": {
            "type": "web_search_call",
            "id": f"ws_{uuid.UUID(int=rng.getrandbits(128)).hex[:10]}",
            "action": {"type": "search", "query": query},
        },
    }


def _token_count_line(scen: Scenario, rng: random.Random) -> dict[str, Any]:
    """Token-count event matching the canonical OpenAI Codex wire shape.

    Mirrors ``TokenCountEvent { info: Option<TokenUsageInfo> }`` from
    ``codex-rs/protocol/src/protocol.rs``, where ``TokenUsageInfo``
    nests ``total_token_usage`` and ``last_token_usage`` (both
    ``TokenUsage`` with the 5 *_tokens fields). An earlier version of
    this generator put the per-field counts directly under ``info`` --
    that's NOT what real Codex emits, and the adapter (correctly)
    expects the nested form, so synthetic final_metrics silently
    dropped. Caught by Cursor bugbot.
    """
    input_tok, output_tok, total = scen.final_tokens
    cached = int(input_tok * 0.1)
    reasoning = int(output_tok * 0.3)
    total_usage = {
        "input_tokens": input_tok,
        "cached_input_tokens": cached,
        "output_tokens": output_tok,
        "reasoning_output_tokens": reasoning,
        "total_tokens": total,
    }
    # last_token_usage represents just the final turn -- synthetic data
    # doesn't track per-turn breakdowns, so use a small fraction.
    last_usage = {
        "input_tokens": input_tok // 4,
        "cached_input_tokens": cached // 4,
        "output_tokens": output_tok // 4,
        "reasoning_output_tokens": reasoning // 4,
        "total_tokens": total // 4,
    }
    return {
        "timestamp": _ts(rng, 5),
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": total_usage,
                "last_token_usage": last_usage,
                "model_context_window": 128_000,
            },
        },
    }


def build_rollout(scen: Scenario, rng: random.Random) -> list[dict[str, Any]]:
    """Return the ordered list of JSONL records for one rollout."""
    session_id = (
        f"sess-synth-codex-{scen.index:02d}-"
        + uuid.UUID(int=rng.getrandbits(128)).hex[:8]
    )
    lines: list[dict[str, Any]] = [
        _session_meta_line(scen, rng, session_id),
        _turn_context_line(scen, rng),
        _user_message_line(scen, rng),
    ]
    # Reasoning before the first agent step (attaches to it in adapter).
    if scen.reasoning:
        lines.append(_reasoning_line(scen, rng))
    if scen.plan_message:
        lines.append(_assistant_message_line(scen, rng, scen.plan_message, minute=2))

    minute = 3
    for call in scen.tool_calls:
        name = call["name"]
        args = call["arguments"]
        output = call.get("output", "")
        if name == "web_search_call":
            lines.append(_web_search_line(rng, args.get("query", ""), minute))
        else:
            call_line, out_line = _function_call_lines(rng, name, args, output, minute)
            lines.append(call_line)
            lines.append(out_line)
        minute += 1

    if scen.final_message:
        lines.append(
            _assistant_message_line(scen, rng, scen.final_message, minute=minute)
        )

    lines.append(_token_count_line(scen, rng))
    return lines


def write_scenario(scen: Scenario, rng: random.Random, out_dir: Path) -> Path:
    out_path = out_dir / f"codex_synth_{scen.index:02d}_{scen.kind}.jsonl"
    lines = build_rollout(scen, rng)
    with out_path.open("w", encoding="utf-8") as fh:
        for obj in lines:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR,
        help="Output directory (default: data/raw/codex/)",
    )
    parser.add_argument(
        "--only",
        type=int,
        default=None,
        help="Generate only the scenario with this 1-based index (default: all 10)",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    targets = (
        [scen for scen in SCENARIOS if scen.index == args.only]
        if args.only is not None
        else SCENARIOS
    )
    if not targets:
        raise SystemExit(f"No scenario with index {args.only}")

    for scen in targets:
        path = write_scenario(scen, rng, args.out)
        n_lines = sum(1 for _ in path.open(encoding="utf-8"))
        print(f"wrote {path}  ({n_lines} lines, kind={scen.kind}, model={scen.model})")


if __name__ == "__main__":
    main()
