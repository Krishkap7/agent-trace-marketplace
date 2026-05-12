"""Codex (OpenAI ``codex`` CLI) session JSONL -> ATIF adapter.

A Codex session is a newline-delimited JSON file in the OpenAI Responses
API event-stream shape. Real Codex stores them at
``~/.codex/sessions/<session-id>.jsonl``. One session = one file = one
ATIF :class:`Trajectory`.

Each line is one event with a top-level ``type`` field. The four outer
types we care about:

* ``session_meta`` (first event) -- carries ``payload.id``,
  ``payload.cli_version``, ``payload.originator``, ``payload.cwd``,
  ``payload.git``, ``payload.instructions``. Seeds the trajectory
  identity.
* ``turn_context`` -- carries ``payload.model``. Seeds the agent's model
  name.
* ``response_item`` -- the actual conversation events. The inner
  ``payload.type`` determines the kind:
    - ``reasoning`` (summary text)
    - ``message`` (with ``role`` and ``content``)
    - ``function_call`` / ``custom_tool_call`` (tool invocations)
    - ``function_call_output`` / ``custom_tool_call_output`` (tool results)
    - ``web_search_call`` (built-in web search)
* ``event_msg`` -- usage metrics, errors, lifecycle. We extract the last
  ``payload.type=token_count`` event to populate ``final_metrics``.

Cross-reference: Harbor's installed Codex adapter at
[harbor/agents/installed/codex.py](https://github.com/harbor-framework/harbor/blob/main/src/harbor/agents/installed/codex.py).
We mirror its event-walking strategy. Differences:

* Harbor uses ``schema_version="ATIF-v1.5"``; we use ``"ATIF-v1.7"`` for
  consistency with the other adapters in this repo.
* Harbor computes cost via LiteLLM pricing; we omit it (would pull a
  heavy dep for a single field).
* Harbor reads from its own logs dir layout; we take a single on-disk
  JSONL path. Same mapping, different I/O.

Like the other adapters, this is a pure function: no DB, no network, no
LLM. Malformed input returns ``None``; the raw blob is preserved in the
``traces`` table either way.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from harbor.models.trajectories import Trajectory
from trace_marketplace.adapters.base import BaseAdapter

log = logging.getLogger(__name__)

SCHEMA_VERSION = "ATIF-v1.7"
AGENT_NAME = "codex"

OUTER_EVENT_TYPES: frozenset[str] = frozenset(
    {"session_meta", "turn_context", "response_item", "event_msg", "compacted"}
)

# Inner payload.type values for response_item events. Anything outside
# this set is logged at DEBUG (not WARNING -- Codex adds new event types
# faster than we can chase them) and skipped.
RESPONSE_ITEM_TYPES: frozenset[str] = frozenset(
    {
        "reasoning",
        "message",
        "function_call",
        "custom_tool_call",
        "function_call_output",
        "custom_tool_call_output",
        "web_search_call",
    }
)

# Map Codex message role -> ATIF Step source. Codex follows the OpenAI
# Responses API roles.
ROLE_TO_SOURCE: dict[str, str] = {
    "user": "user",
    "assistant": "agent",
    "system": "system",
    "developer": "system",
}


def _parse_jsonl(raw_text: str) -> list[dict[str, Any]]:
    """Return one dict per non-empty line of ``raw_text``. Skip malformed."""
    events: list[dict[str, Any]] = []
    for lineno, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError as exc:
            log.warning(
                "CodexAdapter: line %d not valid JSON (skipping): %s",
                lineno,
                exc.msg,
            )
            continue
        if not isinstance(event, dict):
            log.warning(
                "CodexAdapter: line %d JSON is not an object (skipping)", lineno
            )
            continue
        events.append(event)
    return events


def _extract_message_text(content: Any) -> str:
    """Concatenate ``text`` fields from a Codex content-blocks array.

    Codex messages use the OpenAI Responses shape:
    ``content: [{"type": "...", "text": "..."}]``. We're lenient: any block
    with a string ``text`` field contributes, regardless of its ``type``.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _extract_reasoning_summary(summary: Any) -> str | None:
    """Join a reasoning event's ``summary`` field into one string.

    Real Codex emits the OpenAI Responses-API canonical block form:

        "summary": [{"type": "summary_text", "text": "..."}]

    Older / simplified emitters sometimes use a plain list of strings
    (``["..."]``). We accept both so the adapter doesn't silently drop
    reasoning on real production sessions just because the wire shape
    is the canonical one.
    """
    if not isinstance(summary, list) or not summary:
        return None
    parts: list[str] = []
    for entry in summary:
        if isinstance(entry, str) and entry:
            parts.append(entry)
        elif isinstance(entry, dict):
            text = entry.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    if not parts:
        return None
    return "\n".join(parts)


def _parse_function_arguments(raw: Any) -> tuple[dict[str, Any], Any]:
    """Coerce Codex's stringy function-call ``arguments`` into a dict.

    Codex serialises the model's JSON-string arguments verbatim, so we
    parse them here. Returns ``(parsed_dict, raw_value_for_extra)``. The
    raw value is preserved so adapter consumers can still see exactly what
    the model emitted (e.g. for debugging shell-injection-style bugs).
    """
    if isinstance(raw, dict):
        return raw, None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"input": raw}, raw
        if isinstance(parsed, dict):
            return parsed, raw
        return {"value": parsed}, raw
    if raw is None:
        return {}, None
    return {"value": raw}, raw


def _parse_tool_output(raw: Any) -> tuple[str, dict[str, Any] | None]:
    """Extract textual output + optional metadata from a tool result blob.

    Codex tool outputs are typically a JSON string with an ``output`` field
    (e.g. ``{"output": "ls\\nfoo.txt", "metadata": {"exit_code": 0}}``).
    Fall back to ``str(raw)`` on any parse hiccup; we never lose data.
    """
    if raw is None:
        return "", None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw, None
    else:
        parsed = raw
    if isinstance(parsed, dict):
        output = parsed.get("output")
        if output is None:
            output = json.dumps(parsed, ensure_ascii=False)
        elif not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False)
        meta = parsed.get("metadata")
        return output, (meta if isinstance(meta, dict) else None)
    return str(parsed), None


def _build_final_metrics(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return a FinalMetrics dict built from the last ``token_count`` event.

    Codex emits a ``token_count`` event_msg after every turn; the LAST one
    holds the cumulative totals. Cost is left ``None`` (we don't pull
    LiteLLM); ATIF allows that.
    """
    target_info: dict[str, Any] | None = None
    for event in reversed(events):
        if event.get("type") != "event_msg":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "token_count":
            continue
        info = payload.get("info")
        if isinstance(info, dict):
            target_info = info
            break
    if target_info is None:
        return None

    total_usage = target_info.get("total_token_usage")
    if not isinstance(total_usage, dict):
        return None

    metrics: dict[str, Any] = {}
    if (v := total_usage.get("input_tokens")) is not None:
        metrics["total_prompt_tokens"] = v
    if (v := total_usage.get("output_tokens")) is not None:
        metrics["total_completion_tokens"] = v
    if (v := total_usage.get("cached_input_tokens")) is not None:
        metrics["total_cached_tokens"] = v
    extra: dict[str, Any] = {}
    for key in ("reasoning_output_tokens", "total_tokens"):
        if total_usage.get(key) is not None:
            extra[key] = total_usage[key]
    last_usage = target_info.get("last_token_usage")
    if isinstance(last_usage, dict):
        extra["last_token_usage"] = last_usage
    if extra:
        metrics["extra"] = extra
    return metrics or None


class CodexAdapter(BaseAdapter):
    """Convert a Codex session JSONL into a single ATIF :class:`Trajectory`.

    One file = one session = one Trajectory row. Step ids are renumbered
    sequentially from 1 at the end. Tool-call and tool-output events are
    paired by ``call_id`` and emitted as ONE step (with ``tool_calls`` and
    ``observation`` both populated) -- Harbor's pattern, more faithful to
    "one model action = one step" than splitting them.
    """

    source_format = "codex"

    def to_atif(
        self,
        raw_text: str,
        source_path: Path | None = None,
    ) -> Trajectory | None:
        events = _parse_jsonl(raw_text)
        if not events:
            log.warning(
                "CodexAdapter: no events parsed from %s",
                source_path or "<no path>",
            )
            return None

        session_id: str | None = None
        agent_version = "unknown"
        agent_extra: dict[str, Any] = {}
        agent_model_name: str | None = None

        for event in events:
            etype = event.get("type")
            payload = (
                event.get("payload") if isinstance(event.get("payload"), dict) else None
            )

            if etype == "session_meta" and payload is not None and session_id is None:
                sid = payload.get("id")
                if isinstance(sid, str) and sid:
                    session_id = sid
                cli_version = payload.get("cli_version")
                if isinstance(cli_version, str) and cli_version:
                    agent_version = cli_version
                # Verified against codex-rs/protocol/src/protocol.rs SessionMeta:
                # the field is `base_instructions` on current Codex. Older
                # rollouts (and Harbor's adapter) wrote `instructions`; we
                # accept either for forward+backward compat. Source comment
                # in the OpenAI repo: "There used to be an `instructions`
                # field here, which stored user_instructions, but we now
                # save that on TurnContext. base_instructions stores the
                # base instructions for the session."
                for k in ("originator", "cwd", "git"):
                    v = payload.get(k)
                    if v is not None:
                        agent_extra[k] = v
                base_inst = payload.get("base_instructions")
                if base_inst is None:
                    base_inst = payload.get("instructions")
                if base_inst is not None:
                    agent_extra["base_instructions"] = base_inst
            elif etype == "turn_context" and payload is not None:
                if agent_model_name is None:
                    model = payload.get("model")
                    if isinstance(model, str) and model:
                        agent_model_name = model
                # user_instructions moved to TurnContext per the same
                # protocol comment. First occurrence wins.
                user_inst = payload.get("user_instructions")
                if user_inst is not None and "user_instructions" not in agent_extra:
                    agent_extra["user_instructions"] = user_inst

        if session_id is None and source_path is not None:
            session_id = source_path.stem
        if not isinstance(session_id, str) or not session_id:
            log.warning(
                "CodexAdapter: no session_id determinable for %s",
                source_path or "<no path>",
            )
            return None

        steps_data: list[dict[str, Any]] = []
        pending_calls: dict[str, dict[str, Any]] = {}
        pending_reasoning: str | None = None
        unknown_payload_types: dict[str, int] = {}
        unmatched_outputs: int = 0
        for event in events:
            etype = event.get("type")
            if etype not in OUTER_EVENT_TYPES:
                unknown_payload_types[str(etype or "<no-type>")] = (
                    unknown_payload_types.get(str(etype or "<no-type>"), 0) + 1
                )
                continue
            if etype != "response_item":
                continue

            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            ptype = payload.get("type")
            timestamp = event.get("timestamp")

            if ptype == "reasoning":
                pending_reasoning = _extract_reasoning_summary(payload.get("summary"))
                continue

            if ptype == "message":
                role = payload.get("role", "user")
                source = ROLE_TO_SOURCE.get(
                    role if isinstance(role, str) else "", "system"
                )
                text = _extract_message_text(payload.get("content"))
                if not text:
                    # Skip empty assistant turns (Codex emits these as
                    # filler when only a tool_call followed); they would
                    # fail ATIF's non-empty message requirement.
                    continue
                step: dict[str, Any] = {
                    "step_id": 0,
                    "timestamp": timestamp,
                    "source": source,
                    "message": text,
                }
                if source == "agent":
                    if agent_model_name:
                        step["model_name"] = agent_model_name
                    if pending_reasoning:
                        step["reasoning_content"] = pending_reasoning
                        pending_reasoning = None
                steps_data.append(step)
                continue

            if ptype in {"function_call", "custom_tool_call"}:
                call_id = payload.get("call_id")
                if not isinstance(call_id, str) or not call_id:
                    log.debug(
                        "CodexAdapter: %s missing call_id (skipping)",
                        ptype,
                    )
                    continue
                args_key = "arguments" if ptype == "function_call" else "input"
                parsed_args, raw_args = _parse_function_arguments(payload.get(args_key))
                tool_name = payload.get("name")
                if not isinstance(tool_name, str):
                    tool_name = ""
                status = payload.get("status")
                pending_calls[call_id] = {
                    "timestamp": timestamp,
                    "tool_name": tool_name,
                    "parsed_args": parsed_args,
                    "raw_args": raw_args,
                    "status": status,
                    "reasoning": pending_reasoning,
                }
                pending_reasoning = None
                continue

            if ptype in {"function_call_output", "custom_tool_call_output"}:
                call_id = payload.get("call_id")
                output_text, output_meta = _parse_tool_output(payload.get("output"))
                call_info = (
                    pending_calls.pop(call_id, None)
                    if isinstance(call_id, str)
                    else None
                )
                if call_info is None:
                    unmatched_outputs += 1
                    log.debug(
                        "CodexAdapter: tool_call_output with no matching call_id=%r",
                        call_id,
                    )
                    call_info = {
                        "timestamp": timestamp,
                        "tool_name": payload.get("name") or "",
                        "parsed_args": {},
                        "raw_args": None,
                        "status": None,
                        "reasoning": pending_reasoning,
                    }
                    pending_reasoning = None

                tool_call_dict: dict[str, Any] = {
                    "tool_call_id": call_id if isinstance(call_id, str) else "",
                    "function_name": call_info["tool_name"],
                    "arguments": call_info["parsed_args"],
                }
                tc_extra: dict[str, Any] = {}
                if call_info["raw_args"] is not None:
                    tc_extra["raw_arguments"] = call_info["raw_args"]
                if call_info["status"] is not None:
                    tc_extra["status"] = call_info["status"]
                if tc_extra:
                    tool_call_dict["extra"] = tc_extra

                obs_result: dict[str, Any] = {
                    "source_call_id": call_id if isinstance(call_id, str) else None,
                    "content": output_text,
                }
                if output_meta:
                    obs_result["extra"] = {"tool_metadata": output_meta}

                step = {
                    "step_id": 0,
                    "timestamp": call_info["timestamp"] or timestamp,
                    "source": "agent",
                    "message": f"Tool call: {call_info['tool_name'] or '<unnamed>'}",
                    "tool_calls": [tool_call_dict],
                    "observation": {"results": [obs_result]},
                }
                if agent_model_name:
                    step["model_name"] = agent_model_name
                if call_info["reasoning"]:
                    step["reasoning_content"] = call_info["reasoning"]
                steps_data.append(step)
                continue

            if ptype == "web_search_call":
                action = (
                    payload.get("action")
                    if isinstance(payload.get("action"), dict)
                    else {}
                )
                args: dict[str, Any] = {}
                if isinstance(action, dict):
                    for k in ("type", "query", "queries", "url"):
                        if k in action and action[k] is not None:
                            args[k] = action[k]
                status = payload.get("status")
                tc_extra: dict[str, Any] = {}
                if status is not None:
                    tc_extra["status"] = status
                tool_call_dict = {
                    "tool_call_id": payload.get("id") or "",
                    "function_name": "web_search_call",
                    "arguments": args,
                }
                if tc_extra:
                    tool_call_dict["extra"] = tc_extra
                step = {
                    "step_id": 0,
                    "timestamp": timestamp,
                    "source": "agent",
                    "message": f"Web search: {args.get('query') or args.get('url') or '<no query>'}",
                    "tool_calls": [tool_call_dict],
                }
                if agent_model_name:
                    step["model_name"] = agent_model_name
                if pending_reasoning:
                    step["reasoning_content"] = pending_reasoning
                    pending_reasoning = None
                steps_data.append(step)
                continue

            unknown_payload_types[str(ptype or "<no-payload-type>")] = (
                unknown_payload_types.get(str(ptype or "<no-payload-type>"), 0) + 1
            )

        # Flush any pending tool_calls that never got an output. Better to
        # surface them as tool_calls-without-observation than drop them.
        for call_id, call_info in pending_calls.items():
            tool_call_dict = {
                "tool_call_id": call_id,
                "function_name": call_info["tool_name"],
                "arguments": call_info["parsed_args"],
            }
            tc_extra: dict[str, Any] = {}
            if call_info["raw_args"] is not None:
                tc_extra["raw_arguments"] = call_info["raw_args"]
            if call_info["status"] is not None:
                tc_extra["status"] = call_info["status"]
            tc_extra["pending"] = True
            tool_call_dict["extra"] = tc_extra
            step = {
                "step_id": 0,
                "timestamp": call_info["timestamp"],
                "source": "agent",
                "message": f"Tool call (no output): {call_info['tool_name'] or '<unnamed>'}",
                "tool_calls": [tool_call_dict],
            }
            if agent_model_name:
                step["model_name"] = agent_model_name
            if call_info["reasoning"]:
                step["reasoning_content"] = call_info["reasoning"]
            steps_data.append(step)

        if not steps_data:
            log.warning(
                "CodexAdapter: produced 0 steps for %s (no usable events)",
                source_path or "<no path>",
            )
            return None

        for new_id, step in enumerate(steps_data, start=1):
            step["step_id"] = new_id

        traj_extra: dict[str, Any] = {}
        # Top-level synthetic markers (set by our synthetic generator;
        # absent on real sessions) propagate so verify/breakdown can
        # distinguish real vs fixture rows later. ``synthetic_failure_mode``
        # is a free-form short string label (e.g. ``"hallucinated_api"``)
        # that gives downstream failure-mode search a ground-truth tag
        # to evaluate against.
        first_event = events[0]
        if first_event.get("synthetic") is True:
            traj_extra["synthetic"] = True
        failure_mode = first_event.get("synthetic_failure_mode")
        if isinstance(failure_mode, str) and failure_mode:
            traj_extra["synthetic_failure_mode"] = failure_mode
        if unknown_payload_types:
            traj_extra["unknown_event_counts"] = unknown_payload_types
        if unmatched_outputs:
            traj_extra["unmatched_tool_outputs"] = unmatched_outputs

        agent_dict: dict[str, Any] = {
            "name": AGENT_NAME,
            "version": agent_version,
        }
        if agent_model_name:
            agent_dict["model_name"] = agent_model_name
        if agent_extra:
            agent_dict["extra"] = agent_extra

        traj_dict: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "agent": agent_dict,
            "steps": steps_data,
        }
        final_metrics = _build_final_metrics(events)
        if final_metrics:
            traj_dict["final_metrics"] = final_metrics
        if traj_extra:
            traj_dict["extra"] = traj_extra

        try:
            return Trajectory.model_validate(traj_dict)
        except ValidationError as exc:
            log.warning(
                "CodexAdapter: validation failed for %s (%d error(s))",
                source_path or "<no path>",
                len(exc.errors()),
            )
            log.debug("Full validation error: %s", exc)
            return None


__all__ = ["CodexAdapter", "OUTER_EVENT_TYPES", "RESPONSE_ITEM_TYPES"]
