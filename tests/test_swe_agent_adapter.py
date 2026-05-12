"""Fixture-driven tests for ``SWEAgentAdapter``.

The synthetic fixture in ``tests/fixtures/swe_agent_sample.json`` carries one
entry per behaviour we care about:

  * trajectory[0] = system entry (text empty, system_prompt populated)
  * trajectory[1, 3, 5, 7] = ``role="user"`` env observations
  * trajectory[2, 4, 6, 8] = ``role="ai"`` agent turns
  * trajectory[7-8]        = empty text/system_prompt -> should be skipped
  * trajectory[9]          = unrecognised ``role`` -> should be tracked
                             in extra and skipped
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harbor.models.trajectories import Trajectory
from trace_marketplace.adapters.swe_agent import (
    EVAL_LOGS_TRUNCATE_BYTES,
    SWEAgentAdapter,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def adapter() -> SWEAgentAdapter:
    return SWEAgentAdapter()


@pytest.fixture(scope="module")
def converted(adapter: SWEAgentAdapter, swe_agent_sample_path: Path) -> Trajectory:
    raw_text = swe_agent_sample_path.read_text(encoding="utf-8")
    traj = adapter.to_atif(raw_text, source_path=swe_agent_sample_path)
    assert traj is not None, "fixture must convert successfully"
    return traj


# --------------------------------------------------------------------------- #
# Identity / metadata
# --------------------------------------------------------------------------- #


def test_session_id_comes_from_instance_id(converted: Trajectory) -> None:
    assert converted.session_id == "fictional__test-repo-42"


def test_agent_identity_is_swe_agent(converted: Trajectory) -> None:
    assert converted.agent.name == "swe-agent"
    assert converted.agent.model_name == "swe-agent-llama-70b"
    assert converted.agent.version == "unknown"
    assert converted.agent.extra == {"original_format": "swe-agent-hf-row"}


def test_schema_version_is_v1_7(converted: Trajectory) -> None:
    assert converted.schema_version == "ATIF-v1.7"


# --------------------------------------------------------------------------- #
# Step shape
# --------------------------------------------------------------------------- #


def test_step_ids_are_sequential_starting_at_one(converted: Trajectory) -> None:
    ids = [s.step_id for s in converted.steps]
    assert ids == list(range(1, len(ids) + 1))


def test_drops_empty_text_and_unknown_role_entries(converted: Trajectory) -> None:
    """Fixture has 10 entries; 2 are empty (text+system_prompt blank) and 1
    has an unrecognised role. Adapter should produce 7 steps.
    """
    assert len(converted.steps) == 7


def test_step_one_is_system_prompt_with_source_system(
    converted: Trajectory,
) -> None:
    """The system entry uses ``system_prompt``, NOT ``text``. Adapter must
    still produce a step from it and tag source=system."""
    step = converted.steps[0]
    assert step.source == "system"
    assert "autonomous programmer" in step.message
    assert step.model_name is None


def test_user_role_entries_become_source_system(converted: Trajectory) -> None:
    """SWE-agent's ``role="user"`` entries are env observations (file editor
    state, command stdout). Mapping to source="system" is the honest call."""
    sources = [s.source for s in converted.steps]
    # Expected order from the fixture (skipping empty entries):
    # system, system, agent, system, agent, system, agent
    assert sources == [
        "system",
        "system",
        "agent",
        "system",
        "agent",
        "system",
        "agent",
    ]


def test_agent_steps_carry_model_name(converted: Trajectory) -> None:
    for step in converted.steps:
        if step.source == "agent":
            assert step.model_name == "swe-agent-llama-70b"
        else:
            assert step.model_name is None


def test_agent_step_message_preserves_full_text(converted: Trajectory) -> None:
    """No regex parsing of Thought:/Action: -- the raw block stays intact."""
    first_agent = next(s for s in converted.steps if s.source == "agent")
    assert "Let me start by exploring" in first_agent.message
    assert "```" in first_agent.message
    assert "ls -la" in first_agent.message


def test_step_extra_carries_mask_and_cutoff_date(converted: Trajectory) -> None:
    """system entry's cutoff_date should propagate; ai entries' mask=True
    should propagate."""
    sys_step = converted.steps[0]
    assert sys_step.extra is not None
    assert sys_step.extra.get("cutoff_date") == "01.01.2023"
    assert sys_step.extra.get("mask") is False

    first_agent = next(s for s in converted.steps if s.source == "agent")
    assert first_agent.extra is not None
    assert first_agent.extra.get("mask") is True


# --------------------------------------------------------------------------- #
# Trajectory-level extras (these are what the pipeline uses for has_error)
# --------------------------------------------------------------------------- #


def test_trajectory_extra_carries_resolved_from_target(
    converted: Trajectory,
) -> None:
    """target=False in the fixture -> resolved=False. Pipeline turns that
    into has_error=1."""
    extra = converted.extra or {}
    assert extra.get("resolved") is False


def test_trajectory_extra_carries_exit_status(converted: Trajectory) -> None:
    extra = converted.extra or {}
    assert extra.get("exit_status") == "submitted (exit_context)"


def test_trajectory_extra_carries_generated_patch(converted: Trajectory) -> None:
    extra = converted.extra or {}
    patch = extra.get("generated_patch") or ""
    assert "diff --git" in patch
    assert "x = 2" in patch


def test_trajectory_extra_carries_eval_logs_with_size_metadata(
    converted: Trajectory,
) -> None:
    extra = converted.extra or {}
    assert "eval_logs" in extra
    assert extra.get("eval_logs_full_bytes") == len(extra["eval_logs"])


def test_trajectory_extra_tracks_unknown_roles(converted: Trajectory) -> None:
    extra = converted.extra or {}
    assert extra.get("unknown_role_counts") == {"wat-is-this": 1}


# --------------------------------------------------------------------------- #
# Truncation behaviour for oversized eval_logs
# --------------------------------------------------------------------------- #


def test_eval_logs_truncated_when_oversized(
    adapter: SWEAgentAdapter, swe_agent_sample_path: Path
) -> None:
    """Patch the fixture in memory with a giant eval_logs and confirm the
    silver column gets capped while the original size is retained as
    ``eval_logs_full_bytes``."""
    raw_dict = json.loads(swe_agent_sample_path.read_text(encoding="utf-8"))
    big_logs = "X" * (EVAL_LOGS_TRUNCATE_BYTES * 3)
    raw_dict["eval_logs"] = big_logs
    raw_text = json.dumps(raw_dict)

    traj = adapter.to_atif(raw_text)
    assert traj is not None
    extra = traj.extra or {}
    assert extra["eval_logs_full_bytes"] == len(big_logs)
    assert len(extra["eval_logs"]) < len(big_logs)
    assert extra["eval_logs"].startswith("X" * 100)
    assert "truncated" in extra["eval_logs"]


# --------------------------------------------------------------------------- #
# Defensive failure paths
# --------------------------------------------------------------------------- #


def test_returns_none_on_invalid_json(adapter: SWEAgentAdapter) -> None:
    assert adapter.to_atif("not json at all") is None


def test_returns_none_when_top_level_is_list(adapter: SWEAgentAdapter) -> None:
    assert adapter.to_atif(json.dumps([1, 2, 3])) is None


def test_returns_none_when_required_keys_missing(
    adapter: SWEAgentAdapter,
) -> None:
    """Missing ``trajectory`` -- the most likely real-world malformation."""
    assert (
        adapter.to_atif(
            json.dumps({"instance_id": "x", "model_name": "y", "target": True})
        )
        is None
    )


def test_returns_none_when_trajectory_empty(adapter: SWEAgentAdapter) -> None:
    assert (
        adapter.to_atif(
            json.dumps(
                {
                    "instance_id": "x",
                    "model_name": "y",
                    "target": True,
                    "trajectory": [],
                }
            )
        )
        is None
    )


def test_returns_none_when_all_entries_skipped(
    adapter: SWEAgentAdapter,
) -> None:
    """Adapter must not produce a 0-step Trajectory (Pydantic would reject
    it anyway, but we want a clean None instead of a ValidationError)."""
    assert (
        adapter.to_atif(
            json.dumps(
                {
                    "instance_id": "x",
                    "model_name": "y",
                    "target": True,
                    "trajectory": [
                        {"role": "user", "text": "", "system_prompt": ""},
                        {"role": "wat", "text": "blah", "system_prompt": ""},
                    ],
                }
            )
        )
        is None
    )


def test_resolved_only_set_when_target_is_actual_bool(
    adapter: SWEAgentAdapter,
) -> None:
    """If a future row has target=null somehow, we don't want to silently
    label it as failed."""
    raw = {
        "instance_id": "x",
        "model_name": "y",
        "target": None,
        "trajectory": [
            {"role": "system", "system_prompt": "hello", "text": "", "mask": False},
        ],
    }
    traj = adapter.to_atif(json.dumps(raw))
    assert traj is not None
    extra = traj.extra or {}
    assert "resolved" not in extra
