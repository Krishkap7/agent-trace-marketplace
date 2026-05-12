"""Unit tests for the deterministic structural signals.

Each test crafts a minimal ATIF-shaped dict (just the fields the signal
under test consumes) so it's clear which input shape triggers which
output. No DB, no fixtures, no Streamlit -- the functions in
:mod:`trace_marketplace.enrich.failure_rules` are pure dict-in / bool-out.
"""

from __future__ import annotations

from typing import Any

from trace_marketplace.enrich.failure_rules import (
    LOOP_REPEAT_THRESHOLD,
    MAX_TURNS_THRESHOLD,
    any_signal_fired,
    compute_signals,
    signal_abandonment,
    signal_ended_on_error,
    signal_hit_max_turns,
    signal_incomplete_session,
    signal_repeated_tool_call_loop,
)


def _agent_step(
    *,
    message: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    step: dict[str, Any] = {"source": "agent"}
    if message is not None:
        step["message"] = message
    if tool_calls is not None:
        step["tool_calls"] = tool_calls
    if observation is not None:
        step["observation"] = observation
    return step


def _tool_call(name: str, **arguments: Any) -> dict[str, Any]:
    return {"function_name": name, "arguments": dict(arguments)}


def _error_observation(text: str) -> dict[str, Any]:
    return {"observation": {"results": [{"content": text}]}}


# ---------------- signal_hit_max_turns ----------------


def test_hit_max_turns_fires_exactly_at_threshold() -> None:
    traj = {"steps": [_agent_step(message="x") for _ in range(MAX_TURNS_THRESHOLD)]}
    assert signal_hit_max_turns(traj) is True


def test_hit_max_turns_below_threshold_does_not_fire() -> None:
    traj = {"steps": [_agent_step(message="x") for _ in range(MAX_TURNS_THRESHOLD - 1)]}
    assert signal_hit_max_turns(traj) is False


def test_hit_max_turns_handles_missing_steps_field() -> None:
    assert signal_hit_max_turns({}) is False


def test_hit_max_turns_handles_non_list_steps() -> None:
    assert signal_hit_max_turns({"steps": "not-a-list"}) is False


# ---------------- signal_repeated_tool_call_loop ----------------


def test_repeated_tool_call_loop_fires_on_consecutive_identical_calls() -> None:
    call = _tool_call("shell", cmd="ls")
    traj = {
        "steps": [_agent_step(tool_calls=[call]) for _ in range(LOOP_REPEAT_THRESHOLD)]
    }
    assert signal_repeated_tool_call_loop(traj) is True


def test_repeated_tool_call_loop_does_not_fire_below_threshold() -> None:
    call = _tool_call("shell", cmd="ls")
    traj = {
        "steps": [
            _agent_step(tool_calls=[call]) for _ in range(LOOP_REPEAT_THRESHOLD - 1)
        ]
    }
    assert signal_repeated_tool_call_loop(traj) is False


def test_repeated_tool_call_loop_breaks_streak_on_different_args() -> None:
    # 4 identical, then different args, then 4 identical again -- never
    # 5 in a row, so the signal must NOT fire.
    same = [_agent_step(tool_calls=[_tool_call("shell", cmd="ls")]) for _ in range(4)]
    interrupt = [_agent_step(tool_calls=[_tool_call("shell", cmd="cat")])]
    traj = {"steps": [*same, *interrupt, *same]}
    assert signal_repeated_tool_call_loop(traj) is False


def test_repeated_tool_call_loop_ignores_arg_key_ordering() -> None:
    # ``json.dumps(..., sort_keys=True)`` means {"a": 1, "b": 2} and
    # {"b": 2, "a": 1} hash to the same key.
    a = _agent_step(tool_calls=[{"function_name": "f", "arguments": {"a": 1, "b": 2}}])
    b = _agent_step(tool_calls=[{"function_name": "f", "arguments": {"b": 2, "a": 1}}])
    traj = {"steps": [a, b, a, b, a]}
    assert signal_repeated_tool_call_loop(traj) is True


def test_repeated_tool_call_loop_ignores_user_steps_in_between() -> None:
    # User echoes between agent calls shouldn't reset the streak.
    agent_call = _agent_step(tool_calls=[_tool_call("shell", cmd="ls")])
    user_step = {"source": "user", "message": "ok"}
    traj = {
        "steps": [
            agent_call,
            user_step,
            agent_call,
            user_step,
            agent_call,
            user_step,
            agent_call,
            user_step,
            agent_call,
        ]
    }
    assert signal_repeated_tool_call_loop(traj) is True


def test_repeated_tool_call_loop_catches_swe_agent_system_message_repetition() -> None:
    # SWE-agent shape: agent + system alternate, agent messages
    # paraphrase, but the system response is templated and identical.
    # The signal must catch that the SYSTEM side is looping.
    system_template = (
        "Your proposed edit has introduced new syntax error(s). "
        "Please read this error message and try again."
    )
    agent_paraphrases = [
        "The syntax errors persist due to the incorrect use of parentheses.",
        "The syntax errors need to be fixed carefully to ensure correctness.",
        "It seems that the syntax errors are due to indentation issues.",
        "Let me try a different approach to the syntax error.",
        "Attempting yet another fix for the syntax error.",
    ]
    steps: list[dict[str, Any]] = []
    for paraphrase in agent_paraphrases:
        steps.append(_agent_step(message=paraphrase))
        steps.append({"source": "system", "message": system_template})
    assert signal_repeated_tool_call_loop({"steps": steps}) is True


def test_repeated_tool_call_loop_message_only_agent_pattern() -> None:
    # When an agent emits the same plain-text message 5 times in a row,
    # the signal should also fire. (SWE-agent agents occasionally do this.)
    same_msg = "The syntax errors persist due to the indentation."
    traj = {"steps": [_agent_step(message=same_msg) for _ in range(5)]}
    assert signal_repeated_tool_call_loop(traj) is True


# ---------------- signal_ended_on_error ----------------


def test_ended_on_error_fires_on_traceback_in_final_observation() -> None:
    traj = {
        "steps": [
            _agent_step(
                tool_calls=[_tool_call("shell", cmd="python broken.py")],
                **_error_observation("Traceback (most recent call last):\n  File ..."),
            ),
        ]
    }
    assert signal_ended_on_error(traj) is True


def test_ended_on_error_does_not_fire_when_agent_recovers() -> None:
    traj = {
        "steps": [
            _agent_step(
                tool_calls=[_tool_call("shell", cmd="ls /nope")],
                **_error_observation("Error: No such file or directory"),
            ),
            _agent_step(message="Looks like the path was wrong, let me retry."),
        ]
    }
    assert signal_ended_on_error(traj) is False


def test_ended_on_error_ignores_innocuous_text() -> None:
    traj = {
        "steps": [
            _agent_step(
                tool_calls=[_tool_call("shell", cmd="ls")],
                **_error_observation(
                    "file1.txt\nfile2.txt\nthis was erroneous in 1998"
                ),
            ),
        ]
    }
    # Lowercase "erroneous" must not match -- only the explicit
    # error markers should.
    assert signal_ended_on_error(traj) is False


def test_ended_on_error_handles_anthropic_style_content_blocks() -> None:
    # ATIF allows content to be a list of {"type": "text", "text": "..."}
    # blocks. Make sure we walk into the list.
    traj = {
        "steps": [
            _agent_step(
                observation={
                    "results": [
                        {
                            "content": [
                                {"type": "text", "text": "exit code 2"},
                            ]
                        }
                    ]
                }
            ),
        ]
    }
    assert signal_ended_on_error(traj) is True


def test_ended_on_error_fires_on_swe_agent_system_message_template() -> None:
    # SWE-agent error path: the final system step's MESSAGE field
    # contains the error template, not observation.results.
    traj = {
        "steps": [
            _agent_step(message="Let me apply a fix."),
            {
                "source": "system",
                "message": (
                    "Your proposed edit has introduced new syntax error(s). "
                    "SyntaxError: invalid syntax"
                ),
            },
        ]
    }
    assert signal_ended_on_error(traj) is True


def test_ended_on_error_fires_on_exit_context_status_even_without_step_error() -> None:
    # SWE-agent often labels context-window failures via
    # ``trajectory.extra["exit_status"]`` while the final agent step
    # reads like an innocuous summary.
    traj = {
        "extra": {"exit_status": "submitted (exit_context)"},
        "steps": [_agent_step(message="Submitting the patch.")],
    }
    assert signal_ended_on_error(traj) is True


def test_ended_on_error_does_not_fire_when_exit_status_is_clean_submission() -> None:
    traj = {
        "extra": {"exit_status": "submitted"},
        "steps": [_agent_step(message="Submitting the patch.")],
    }
    assert signal_ended_on_error(traj) is False


# ---------------- signal_abandonment ----------------


def test_abandonment_fires_on_explicit_give_up() -> None:
    traj = {
        "steps": [
            _agent_step(message="Let me try one more thing."),
            _agent_step(
                message="I cannot solve this within the available tools. I'll stop here."
            ),
        ]
    }
    assert signal_abandonment(traj) is True


def test_abandonment_only_inspects_final_agent_message() -> None:
    # Earlier "I cannot" should not trigger if the agent later recovers
    # and the final message is a clean wrap-up.
    traj = {
        "steps": [
            _agent_step(message="I cannot do X without a tool."),
            _agent_step(message="Wait -- I'll use the shell tool instead. Done."),
        ]
    }
    assert signal_abandonment(traj) is False


def test_abandonment_handles_smart_quote_apostrophe() -> None:
    # Real Claude / Codex exports normalise straight ' to curly '.
    traj = {
        "steps": [
            _agent_step(message="I\u2019m sorry, but I can\u2019t complete this.")
        ]
    }
    assert signal_abandonment(traj) is True


# ---------------- signal_incomplete_session ----------------


def test_incomplete_session_fires_when_last_call_has_no_observation() -> None:
    traj = {
        "steps": [
            _agent_step(message="thinking"),
            _agent_step(tool_calls=[_tool_call("shell", cmd="long_running")]),
        ]
    }
    assert signal_incomplete_session(traj) is True


def test_incomplete_session_does_not_fire_when_paired_in_same_step() -> None:
    # Codex pattern: the function_call + function_call_output are
    # fused into one Step with both tool_calls AND observation.
    traj = {
        "steps": [
            _agent_step(
                tool_calls=[_tool_call("shell", cmd="ls")],
                observation={"results": [{"content": "file.txt"}]},
            ),
        ]
    }
    assert signal_incomplete_session(traj) is False


def test_incomplete_session_does_not_fire_when_next_step_has_observation() -> None:
    traj = {
        "steps": [
            _agent_step(tool_calls=[_tool_call("shell", cmd="ls")]),
            {
                "source": "tool",
                "observation": {"results": [{"content": "file.txt"}]},
            },
        ]
    }
    assert signal_incomplete_session(traj) is False


def test_incomplete_session_handles_no_tool_calls_at_all() -> None:
    traj = {
        "steps": [
            _agent_step(message="hello"),
            _agent_step(message="goodbye"),
        ]
    }
    assert signal_incomplete_session(traj) is False


# ---------------- compute_signals + any_signal_fired ----------------


def test_compute_signals_returns_one_bool_per_signal() -> None:
    sigs = compute_signals({"steps": []})
    expected_keys = {
        "hit_max_turns",
        "repeated_tool_call_loop",
        "ended_on_error",
        "abandonment",
        "incomplete_session",
    }
    assert set(sigs.keys()) == expected_keys
    assert all(isinstance(v, bool) for v in sigs.values())


def test_compute_signals_all_false_on_empty_trajectory() -> None:
    sigs = compute_signals({"steps": []})
    assert all(v is False for v in sigs.values())
    assert any_signal_fired(sigs) is False


def test_any_signal_fired_true_when_at_least_one_signal() -> None:
    assert any_signal_fired({"a": False, "b": True, "c": False}) is True


def test_any_signal_fired_handles_truthy_non_bool_safely() -> None:
    # Defensive: any_signal_fired should coerce, not blow up, if
    # something other than a bool ever slips in. compute_signals
    # only emits bools, but pipe-through paths might not.
    assert any_signal_fired({"a": 0, "b": 1}) is True  # type: ignore[dict-item]


def test_compute_signals_integration_on_obvious_failure_trace() -> None:
    """Multi-signal sanity: a trajectory that should light up several rules."""
    call = _tool_call("shell", cmd="pytest")
    steps: list[dict[str, Any]] = []
    # 5 identical pytest calls in a row -> loop
    for _ in range(LOOP_REPEAT_THRESHOLD):
        steps.append(
            _agent_step(
                tool_calls=[call],
                observation={
                    "results": [{"content": "Traceback (most recent call last):"}]
                },
            )
        )
    # Final agent give-up.
    steps.append(_agent_step(message="I cannot fix this within the available tools."))
    sigs = compute_signals({"steps": steps})
    assert sigs["repeated_tool_call_loop"] is True
    assert sigs["abandonment"] is True
    assert any_signal_fired(sigs) is True
