"""Round-trip integration tests for the synthetic Codex/Cursor generators.

The unit test fixtures in ``tests/fixtures/`` are intentionally small
and use canonical wire shapes. They don't catch generator-vs-adapter
drift -- e.g. a generator that emits a slightly-wrong shape that
satisfies validation but loses information silently.

This module closes that gap: it runs the *actual* generator output
through detection + adapter and asserts the high-value invariants
hold (non-None Trajectory, non-empty steps, populated final_metrics
for Codex, propagated synthetic markers, etc). This is the safety
net that would have caught the two bugbot findings on PR #4:

  1. reasoning summaries silently dropped (dict-block form ignored)
  2. final_metrics silently None (info.* vs info.total_token_usage.*)

Both bugs survived the fixture-driven unit tests because the fixtures
happened to use a different wire shape than the generators.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trace_marketplace.adapters import ADAPTERS
from trace_marketplace.ingest.detect import detect_format

REPO_ROOT = Path(__file__).resolve().parents[1]
CODEX_DIR = REPO_ROOT / "data" / "raw" / "codex"
CURSOR_DIR = REPO_ROOT / "data" / "raw" / "cursor"


def _codex_files() -> list[Path]:
    return sorted(CODEX_DIR.glob("codex_synth_*.jsonl"))


def _cursor_files() -> list[Path]:
    return sorted(CURSOR_DIR.glob("cursor_synth_*.cursor.json"))


# --------------------------------------------------------------------------- #
# Codex
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", _codex_files(), ids=lambda p: p.name)
def test_codex_synth_file_detected_as_codex(path: Path) -> None:
    assert detect_format(path) == "codex"


@pytest.mark.parametrize("path", _codex_files(), ids=lambda p: p.name)
def test_codex_synth_file_round_trips(path: Path) -> None:
    """Each generated Codex rollout must produce a valid Trajectory."""
    traj = ADAPTERS["codex"].to_atif(path.read_text(encoding="utf-8"), path)
    assert traj is not None, f"{path.name} failed to convert"
    assert len(traj.steps) >= 1, f"{path.name} produced 0 steps"
    assert traj.agent.name == "codex"
    assert traj.agent.model_name is not None
    extra = traj.extra or {}
    assert extra.get("synthetic") is True, f"{path.name} lost synthetic marker"


@pytest.mark.parametrize("path", _codex_files(), ids=lambda p: p.name)
def test_codex_synth_final_metrics_populated(path: Path) -> None:
    """Regression for bugbot finding #2 on PR #4: the generator was
    emitting tokens flat under ``info.*`` while the adapter (correctly,
    per ``codex-rs/protocol/src/protocol.rs``) reads
    ``info.total_token_usage.*``. Every generated file MUST produce
    non-None final_metrics with real numbers."""
    traj = ADAPTERS["codex"].to_atif(path.read_text(encoding="utf-8"), path)
    assert traj is not None
    assert traj.final_metrics is not None, (
        f"{path.name}: final_metrics is None -- generator likely "
        f"using wrong token_count shape again"
    )
    assert traj.final_metrics.total_prompt_tokens is not None
    assert traj.final_metrics.total_prompt_tokens > 0
    assert traj.final_metrics.total_completion_tokens is not None
    assert traj.final_metrics.total_completion_tokens > 0


@pytest.mark.parametrize("path", _codex_files(), ids=lambda p: p.name)
def test_codex_synth_first_agent_step_has_reasoning(path: Path) -> None:
    """Regression for bugbot finding #1 on PR #4: every generated
    rollout includes a reasoning event before the plan_message, and
    the adapter must attach it to the next agent step. The previous
    string-only summary filter silently dropped the canonical
    dict-block form."""
    traj = ADAPTERS["codex"].to_atif(path.read_text(encoding="utf-8"), path)
    assert traj is not None
    agent_steps = [s for s in traj.steps if s.source == "agent"]
    assert agent_steps, f"{path.name} has no agent steps"
    assert agent_steps[0].reasoning_content, (
        f"{path.name}: first agent step lost its reasoning_content "
        f"-- generator/adapter summary shape drift?"
    )


def test_codex_synth_failure_modes_all_distinct() -> None:
    """All 3 failure files must carry a distinct named failure mode."""
    modes: dict[str, str] = {}
    for path in _codex_files():
        if "_failure" not in path.name:
            continue
        traj = ADAPTERS["codex"].to_atif(path.read_text(encoding="utf-8"), path)
        assert traj is not None
        fm = (traj.extra or {}).get("synthetic_failure_mode")
        assert isinstance(fm, str) and fm
        modes[path.name] = fm
    assert len(modes) == 3
    assert len(set(modes.values())) == 3, f"failure modes collide: {modes}"


# --------------------------------------------------------------------------- #
# Cursor
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", _cursor_files(), ids=lambda p: p.name)
def test_cursor_synth_file_detected_as_cursor(path: Path) -> None:
    assert detect_format(path) == "cursor"


@pytest.mark.parametrize("path", _cursor_files(), ids=lambda p: p.name)
def test_cursor_synth_file_round_trips(path: Path) -> None:
    traj = ADAPTERS["cursor"].to_atif(path.read_text(encoding="utf-8"), path)
    assert traj is not None, f"{path.name} failed to convert"
    assert len(traj.steps) >= 1, f"{path.name} produced 0 steps"
    assert traj.agent.name == "cursor"
    assert traj.agent.model_name is not None
    extra = traj.extra or {}
    assert extra.get("synthetic") is True, f"{path.name} lost synthetic marker"


def test_cursor_synth_failure_modes_all_distinct() -> None:
    modes: dict[str, str] = {}
    for path in _cursor_files():
        if "_failure" not in path.name:
            continue
        traj = ADAPTERS["cursor"].to_atif(path.read_text(encoding="utf-8"), path)
        assert traj is not None
        fm = (traj.extra or {}).get("synthetic_failure_mode")
        assert isinstance(fm, str) and fm
        modes[path.name] = fm
    assert len(modes) == 3
    assert len(set(modes.values())) == 3, f"failure modes collide: {modes}"
