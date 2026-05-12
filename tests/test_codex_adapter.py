"""Fixture-driven tests for ``CodexAdapter``.

The 12-event synthetic fixture exercises every event-routing branch:

  * session_meta (identity seeding)
  * turn_context (model_name seeding)
  * response_item.message (user + assistant)
  * response_item.reasoning (buffered, attached to next agent step)
  * response_item.function_call + .function_call_output (paired by call_id)
  * response_item.web_search_call (standalone tool call)
  * event_msg.token_count (final_metrics)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harbor.models.trajectories import Trajectory
from trace_marketplace.adapters.codex import CodexAdapter


@pytest.fixture(scope="module")
def adapter() -> CodexAdapter:
    return CodexAdapter()


@pytest.fixture(scope="module")
def converted(adapter: CodexAdapter, codex_sample_path: Path) -> Trajectory:
    raw_text = codex_sample_path.read_text(encoding="utf-8")
    traj = adapter.to_atif(raw_text, source_path=codex_sample_path)
    assert traj is not None, "fixture must convert successfully"
    return traj


# --------------------------------------------------------------------------- #
# Identity / metadata
# --------------------------------------------------------------------------- #


def test_session_id_from_session_meta(converted: Trajectory) -> None:
    assert converted.session_id == "sess-fixture-codex-1"


def test_agent_identity_is_codex(converted: Trajectory) -> None:
    assert converted.agent.name == "codex"
    assert converted.agent.version == "0.32.0"
    assert converted.agent.model_name == "gpt-5"


def test_agent_extra_carries_session_meta_payload(converted: Trajectory) -> None:
    extra = converted.agent.extra or {}
    assert extra.get("originator") == "codex_cli_rs"
    assert extra.get("cwd") == "/home/me/fictional"
    # Current Codex emits `base_instructions` on session_meta and
    # `user_instructions` on turn_context (see codex-rs/protocol/src/protocol.rs).
    assert extra.get("base_instructions") == "You are Codex."
    assert extra.get("user_instructions") == "Be terse."
    assert isinstance(extra.get("git"), dict)


def test_synthetic_failure_mode_propagates_to_trajectory_extra(
    adapter: CodexAdapter,
) -> None:
    """Generators set `synthetic_failure_mode` as a sibling of
    `synthetic` on the session_meta line. The adapter must carry it
    through to ``trajectory.extra`` so it survives as a ground-truth
    label in the DB."""
    payload = "\n".join(
        [
            json.dumps(
                {
                    "type": "session_meta",
                    "payload": {"id": "fm-test", "cli_version": "0.32.0"},
                    "synthetic": True,
                    "synthetic_failure_mode": "hallucinated_api",
                }
            ),
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"text": "go"}],
                    },
                }
            ),
        ]
    )
    traj = adapter.to_atif(payload)
    assert traj is not None
    extra = traj.extra or {}
    assert extra.get("synthetic") is True
    assert extra.get("synthetic_failure_mode") == "hallucinated_api"


def test_legacy_instructions_field_falls_back_to_base_instructions(
    adapter: CodexAdapter,
) -> None:
    """Older Codex rollouts (and Harbor's adapter) wrote `instructions`
    instead of `base_instructions`. The adapter must accept both."""
    payload = "\n".join(
        [
            json.dumps(
                {
                    "type": "session_meta",
                    "payload": {
                        "id": "legacy",
                        "cli_version": "0.20.0",
                        "instructions": "legacy text",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"text": "hi"}],
                    },
                }
            ),
        ]
    )
    traj = adapter.to_atif(payload)
    assert traj is not None
    extra = traj.agent.extra or {}
    assert extra.get("base_instructions") == "legacy text"


def test_schema_version_is_v1_7(converted: Trajectory) -> None:
    assert converted.schema_version == "ATIF-v1.7"


def test_synthetic_marker_propagates(converted: Trajectory) -> None:
    extra = converted.extra or {}
    assert extra.get("synthetic") is True


# --------------------------------------------------------------------------- #
# Step shape
# --------------------------------------------------------------------------- #


def test_step_ids_are_sequential_starting_at_one(converted: Trajectory) -> None:
    ids = [s.step_id for s in converted.steps]
    assert ids == list(range(1, len(ids) + 1))


def test_message_user_becomes_source_user_step(converted: Trajectory) -> None:
    user_steps = [s for s in converted.steps if s.source == "user"]
    assert len(user_steps) == 1
    assert "List the files" in user_steps[0].message


def test_message_assistant_becomes_source_agent_step(converted: Trajectory) -> None:
    """The final assistant message in the fixture should land as a Step."""
    matches = [
        s
        for s in converted.steps
        if s.source == "agent" and s.message and "changed `x` to 2" in s.message
    ]
    assert len(matches) == 1
    assert matches[0].model_name == "gpt-5"


def test_function_call_and_output_collapse_into_one_step(
    converted: Trajectory,
) -> None:
    """Harbor's pattern: each call+output pair = ONE step with both
    tool_calls AND observation populated. The fixture has 2 such pairs."""
    paired = [s for s in converted.steps if s.tool_calls and s.observation]
    assert len(paired) == 2
    for step in paired:
        assert len(step.tool_calls) == 1
        assert step.tool_calls[0].function_name == "shell"
        assert step.observation is not None
        assert len(step.observation.results) == 1
        assert step.observation.results[0].source_call_id in {"fc_1", "fc_2"}


def test_function_call_arguments_parsed_to_dict(converted: Trajectory) -> None:
    """Codex serialises function-call arguments as a JSON string; the
    adapter must parse them into a dict."""
    paired = [s for s in converted.steps if s.tool_calls and s.observation]
    args_first = paired[0].tool_calls[0].arguments
    assert isinstance(args_first, dict)
    assert args_first.get("cmd") == ["ls", "-la"]


def test_function_call_raw_arguments_preserved_in_extra(
    converted: Trajectory,
) -> None:
    """We parse arguments AND keep the original raw string in extra for
    debuggability."""
    paired = [s for s in converted.steps if s.tool_calls and s.observation]
    tc_extra = paired[0].tool_calls[0].extra or {}
    assert tc_extra.get("raw_arguments") == '{"cmd":["ls","-la"]}'
    assert tc_extra.get("status") == "completed"


def test_tool_output_text_extracted_from_json_blob(converted: Trajectory) -> None:
    paired = [s for s in converted.steps if s.tool_calls and s.observation]
    first_output = paired[0].observation.results[0].content
    assert "foo.py" in first_output
    assert "README.md" in first_output


def test_tool_output_metadata_lands_in_observation_extra(
    converted: Trajectory,
) -> None:
    paired = [s for s in converted.steps if s.tool_calls and s.observation]
    obs_result_extra = paired[0].observation.results[0].extra or {}
    meta = obs_result_extra.get("tool_metadata") or {}
    assert meta.get("exit_code") == 0


def test_web_search_call_becomes_standalone_tool_step(
    converted: Trajectory,
) -> None:
    """web_search_call has no paired output -- emits a single Step with
    tool_calls but no observation."""
    web_searches = [
        s
        for s in converted.steps
        if s.tool_calls
        and any(tc.function_name == "web_search_call" for tc in s.tool_calls)
    ]
    assert len(web_searches) == 1
    step = web_searches[0]
    args = step.tool_calls[0].arguments
    assert args.get("query") == "python module export pattern"
    assert step.observation is None


# --------------------------------------------------------------------------- #
# Reasoning routing (the buffered pending_reasoning pattern)
# --------------------------------------------------------------------------- #


def test_reasoning_attaches_to_next_agent_step(converted: Trajectory) -> None:
    """The fixture has two reasoning events, each immediately followed by a
    tool_call. The reasoning text should land on the matching paired
    step's reasoning_content."""
    paired = [s for s in converted.steps if s.tool_calls and s.observation]
    assert paired[0].reasoning_content is not None
    assert "listing the repo files" in paired[0].reasoning_content.lower()
    assert paired[1].reasoning_content is not None
    assert "found foo.py" in paired[1].reasoning_content.lower()


# --------------------------------------------------------------------------- #
# Final metrics
# --------------------------------------------------------------------------- #


def test_final_metrics_extracted_from_token_count(converted: Trajectory) -> None:
    fm = converted.final_metrics
    assert fm is not None
    assert fm.total_prompt_tokens == 1234
    assert fm.total_completion_tokens == 567
    assert fm.total_cached_tokens == 100
    assert fm.total_cost_usd is None  # we intentionally don't compute cost
    extra = fm.extra or {}
    assert extra.get("reasoning_output_tokens") == 50
    assert extra.get("total_tokens") == 1801


# --------------------------------------------------------------------------- #
# Defensive failure paths
# --------------------------------------------------------------------------- #


def test_returns_none_on_empty_input(adapter: CodexAdapter) -> None:
    assert adapter.to_atif("") is None


def test_returns_none_when_all_lines_malformed(adapter: CodexAdapter) -> None:
    assert adapter.to_atif("not json\nstill not json\n") is None


def test_returns_none_when_no_useful_events(adapter: CodexAdapter) -> None:
    """Just a session_meta yields no Steps; ATIF requires >=1, so None."""
    payload = json.dumps(
        {"type": "session_meta", "payload": {"id": "x", "cli_version": "0.0.1"}}
    )
    assert adapter.to_atif(payload + "\n") is None


def test_returns_none_when_input_is_not_jsonl_objects(
    adapter: CodexAdapter,
) -> None:
    """Top-level non-dict lines are skipped; if NONE survive, return None."""
    assert adapter.to_atif('[1, 2, 3]\n"a string"\n42\n') is None


def test_session_id_falls_back_to_filename_stem(
    adapter: CodexAdapter, tmp_path: Path
) -> None:
    """A Codex JSONL with no session_meta must still get a session_id."""
    p = tmp_path / "fallback-id.jsonl"
    p.write_text(
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"text": "hi"}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    traj = adapter.to_atif(p.read_text(encoding="utf-8"), source_path=p)
    assert traj is not None
    assert traj.session_id == "fallback-id"


def test_unmatched_tool_output_counted(adapter: CodexAdapter) -> None:
    """A function_call_output with no preceding function_call should NOT
    crash; it should produce a step (best effort) and increment the
    unmatched_tool_outputs counter."""
    payload = "\n".join(
        [
            json.dumps(
                {
                    "type": "session_meta",
                    "payload": {"id": "orphan-test", "cli_version": "0.0.0"},
                }
            ),
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call_output",
                        "call_id": "orphan_1",
                        "output": '{"output":"stranded"}',
                    },
                }
            ),
        ]
    )
    traj = adapter.to_atif(payload)
    assert traj is not None
    extra = traj.extra or {}
    assert extra.get("unmatched_tool_outputs") == 1


def test_pending_tool_call_without_output_still_emitted(
    adapter: CodexAdapter,
) -> None:
    """A function_call with no matching output gets flushed at the end as
    a step with tool_calls but no observation."""
    payload = "\n".join(
        [
            json.dumps(
                {
                    "type": "session_meta",
                    "payload": {"id": "pending-test", "cli_version": "0.0.0"},
                }
            ),
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "call_id": "pending_1",
                        "name": "shell",
                        "arguments": "{}",
                    },
                }
            ),
        ]
    )
    traj = adapter.to_atif(payload)
    assert traj is not None
    pending_steps = [s for s in traj.steps if s.tool_calls and s.observation is None]
    assert len(pending_steps) == 1
    tc_extra = pending_steps[0].tool_calls[0].extra or {}
    assert tc_extra.get("pending") is True
