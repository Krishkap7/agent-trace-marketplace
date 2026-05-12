"""Claude Code session JSONL -> ATIF adapter.

A Claude Code session is a newline-delimited JSON file. Each line is one
event with a top-level ``"type"`` field. The events of interest:

* ``user`` -- either a real user message (string content) or the carrier
  for one-or-more ``tool_result`` blocks responding to a previous assistant
  turn (array content).
* ``assistant`` -- a model turn. ``message.content`` is an array of mixed
  ``text``/``thinking``/``tool_use`` blocks (Anthropic Messages API shape).
* ``system`` -- a system-injected message; the empty-content variant
  (``subtype=compact_boundary``, etc.) is dropped.

Everything else (``permission-mode``, ``file-history-snapshot``,
``last-prompt``, ``queue-operation``, every ``attachment`` sub-type) is
metadata and is **not** emitted as a Step. We do, however, count
occurrences so the trajectory's ``extra`` field records what was filtered.

Cross-reference: Harbor's installed Claude Code adapter
[harbor/agents/installed/claude_code.py](https://github.com/harbor-framework/harbor/blob/main/src/harbor/agents/installed/claude_code.py)
inspired the content-block walking strategy. We re-implement on our own
small interface rather than importing Harbor's agent runner.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from harbor.models.trajectories import Trajectory
from trace_marketplace.adapters.base import BaseAdapter

log = logging.getLogger(__name__)

SCHEMA_VERSION = "ATIF-v1.7"
AGENT_NAME = "claude-code"

# Event types we explicitly handle. Anything else falls through to the
# "unknown" branch with a warning.
HANDLED_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "user",
        "assistant",
        "system",
        "attachment",
        "permission-mode",
        "file-history-snapshot",
        "last-prompt",
        "queue-operation",
    }
)


def _parse_jsonl(raw_text: str) -> list[dict[str, Any]]:
    """Return one dict per non-empty line of ``raw_text``.

    Skips blank lines and lines that fail to parse (logged at WARNING with
    the line number, never with the offending content).
    """
    events: list[dict[str, Any]] = []
    for lineno, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError as exc:
            log.warning(
                "ClaudeCodeAdapter: line %d not valid JSON (skipping): %s",
                lineno,
                exc.msg,
            )
            continue
        if not isinstance(event, dict):
            log.warning(
                "ClaudeCodeAdapter: line %d JSON is not an object (skipping)",
                lineno,
            )
            continue
        events.append(event)
    return events


def _extract_first(
    events: list[dict[str, Any]],
    key_path: tuple[str, ...],
) -> Any | None:
    """Return the first non-None value at ``key_path`` across ``events``."""
    for event in events:
        current: Any = event
        for key in key_path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
            if current is None:
                break
        if current is not None:
            return current
    return None


def _build_metrics(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    """Translate Anthropic ``message.usage`` to an ATIF Metrics dict.

    Returns ``None`` if ``usage`` is missing so callers can omit the field
    rather than serialise empty Metrics. Tokens not represented in ATIF
    (cache_creation_input_tokens, server_tool_use counts, etc.) are stashed
    in ``metrics.extra``.
    """
    if not isinstance(usage, dict):
        return None
    metrics: dict[str, Any] = {}
    if (v := usage.get("input_tokens")) is not None:
        metrics["prompt_tokens"] = v
    if (v := usage.get("output_tokens")) is not None:
        metrics["completion_tokens"] = v
    if (v := usage.get("cache_read_input_tokens")) is not None:
        metrics["cached_tokens"] = v
    # Stash the Anthropic-specific fields ATIF doesn't model.
    extra_keys = ("cache_creation_input_tokens", "service_tier", "iterations")
    extra: dict[str, Any] = {
        k: usage[k] for k in extra_keys if usage.get(k) is not None
    }
    if extra:
        metrics["extra"] = extra
    return metrics or None


def _walk_assistant_content(
    blocks: list[Any],
) -> tuple[str, str | None, list[dict[str, Any]]]:
    """Walk an assistant ``message.content`` array.

    Returns ``(text, reasoning, tool_call_dicts)`` where:
    * ``text`` is all ``text`` blocks concatenated with single newlines.
    * ``reasoning`` is all ``thinking`` blocks similarly concatenated, or
      ``None`` if absent. Lands in ``Step.reasoning_content``.
    * ``tool_call_dicts`` is one entry per ``tool_use`` block, ready to be
      passed to ``ToolCall.model_validate``.
    """
    texts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
        elif btype == "thinking":
            thought = block.get("thinking") or block.get("text")
            if isinstance(thought, str) and thought:
                thinking_parts.append(thought)
        elif btype == "tool_use":
            tool_id = block.get("id")
            name = block.get("name")
            if not isinstance(tool_id, str) or not isinstance(name, str):
                log.debug(
                    "ClaudeCodeAdapter: skipping malformed tool_use block (id=%r, name=%r)",
                    tool_id,
                    name,
                )
                continue
            args = block.get("input")
            if not isinstance(args, dict):
                args = {} if args is None else {"_raw": args}
            tool_calls.append(
                {
                    "tool_call_id": tool_id,
                    "function_name": name,
                    "arguments": args,
                }
            )
        else:
            log.debug(
                "ClaudeCodeAdapter: unknown assistant content block type %r",
                btype,
            )

    return (
        "\n".join(texts),
        "\n".join(thinking_parts) if thinking_parts else None,
        tool_calls,
    )


def _extract_tool_result_content(block: dict[str, Any]) -> str:
    """Best-effort string from a ``tool_result`` content block.

    ``block["content"]`` can be a string OR a list of typed sub-blocks (per
    the Anthropic Messages API). We concatenate the textual parts and
    silently drop image references (their base64 payload is huge and not
    useful for failure-mode search).
    """
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for sub in content:
            if isinstance(sub, dict) and sub.get("type") == "text":
                text = sub.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _tool_result_extras(event: dict[str, Any]) -> dict[str, Any] | None:
    """Pull structured tool-result metadata (exitCode, interrupted, etc.).

    Lives at the event's top-level ``toolUseResult``. Returns ``None`` if
    nothing useful is present so we don't bloat ``ObservationResult.extra``
    with empty dicts.
    """
    raw = event.get("toolUseResult")
    if not isinstance(raw, dict):
        return None
    interesting = {
        k: raw[k]
        for k in ("exitCode", "interrupted", "isImage", "stderr_lines", "type")
        if raw.get(k) is not None
    }
    return interesting or None


def _build_trajectory_extra(
    events: list[dict[str, Any]],
    drop_counts: Counter[str],
    attachment_counts: Counter[str],
    unknown_types: Counter[str],
) -> dict[str, Any]:
    """Assemble ``Trajectory.extra`` from first-occurrence metadata + drop counters.

    Real Fleet sessions usually open with a ``permission-mode`` event which
    does NOT carry ``cwd`` / ``gitBranch`` / etc. We therefore scan events
    until the first non-null value for each metadata key, rather than
    grabbing everything off ``events[0]``.
    """
    extra: dict[str, Any] = {}
    for key in ("cwd", "gitBranch", "userType", "entrypoint"):
        value = _extract_first(events, (key,))
        if value is not None:
            extra[key] = value

    if drop_counts.get("permission-mode"):
        extra["permission_mode_changes"] = drop_counts["permission-mode"]
    if drop_counts.get("file-history-snapshot"):
        extra["file_history_count"] = drop_counts["file-history-snapshot"]
    if drop_counts.get("last-prompt"):
        extra["last_prompt_count"] = drop_counts["last-prompt"]
    if drop_counts.get("queue-operation"):
        extra["queue_operation_count"] = drop_counts["queue-operation"]

    if attachment_counts:
        extra["attachment_counts"] = dict(attachment_counts)
    if unknown_types:
        extra["unknown_event_counts"] = dict(unknown_types)

    return extra


class ClaudeCodeAdapter(BaseAdapter):
    """Convert a Claude Code session JSONL into a single ATIF :class:`Trajectory`.

    One file = one session = one Trajectory row. Step ids are renumbered
    sequentially from 1. The adapter is a pure function: it never writes
    anywhere, never reads anything besides its ``raw_text`` argument.
    """

    source_format = "claude_code"

    def to_atif(
        self,
        raw_text: str,
        source_path: Path | None = None,
    ) -> Trajectory | None:
        events = _parse_jsonl(raw_text)
        if not events:
            log.warning(
                "ClaudeCodeAdapter: no events parsed from %s",
                source_path or "<no path>",
            )
            return None

        session_id = _extract_first(events, ("sessionId",))
        if not isinstance(session_id, str) and source_path is not None:
            session_id = source_path.stem

        version = _extract_first(events, ("version",))
        version_str = version if isinstance(version, str) and version else "unknown"

        model_name = _extract_first(events, ("message", "model"))
        model_name_str = model_name if isinstance(model_name, str) else None

        agent_dict: dict[str, Any] = {
            "name": AGENT_NAME,
            "version": version_str,
        }
        if model_name_str:
            agent_dict["model_name"] = model_name_str

        steps_data: list[dict[str, Any]] = []
        tool_use_id_to_step_idx: dict[str, int] = {}
        drop_counts: Counter[str] = Counter()
        attachment_counts: Counter[str] = Counter()
        unknown_types: Counter[str] = Counter()
        unmatched_box: list[int] = [0]

        for event in events:
            etype = event.get("type")

            if etype == "user":
                self._handle_user(
                    event,
                    steps_data,
                    tool_use_id_to_step_idx,
                    unmatched_box,
                )
            elif etype == "assistant":
                self._handle_assistant(
                    event,
                    steps_data,
                    tool_use_id_to_step_idx,
                )
            elif etype == "system":
                self._handle_system(event, steps_data)
            elif etype == "attachment":
                sub = event.get("attachment", {})
                sub_type = sub.get("type") if isinstance(sub, dict) else None
                attachment_counts[str(sub_type or "<unknown>")] += 1
            elif etype in HANDLED_EVENT_TYPES:
                drop_counts[etype] += 1
            else:
                key = etype if isinstance(etype, str) else "<no-type>"
                unknown_types[key] += 1
                log.warning(
                    "ClaudeCodeAdapter: unknown event type %r at %s",
                    key,
                    source_path.name if source_path else "<no path>",
                )

        # Renumber sequentially. ATIF requires 1..N.
        for new_id, step in enumerate(steps_data, start=1):
            step["step_id"] = new_id

        # Count any tool_result ids we couldn't match -- helps catch
        # adapter bugs without raising.
        for step in steps_data:
            obs = step.get("observation")
            if isinstance(obs, dict):
                for r in obs.get("results", []):
                    if r.get("source_call_id") and not r.get("content"):
                        log.debug(
                            "ClaudeCodeAdapter: empty content for tool_use_id=%s",
                            r["source_call_id"],
                        )

        traj_extra = _build_trajectory_extra(
            events, drop_counts, attachment_counts, unknown_types
        )
        if unmatched_box[0]:
            traj_extra["unmatched_tool_results"] = unmatched_box[0]

        traj_dict: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id if isinstance(session_id, str) else None,
            "agent": agent_dict,
            "steps": steps_data,
        }
        if traj_extra:
            traj_dict["extra"] = traj_extra

        try:
            return Trajectory.model_validate(traj_dict)
        except ValidationError as exc:
            log.warning(
                "ClaudeCodeAdapter: validation failed for %s (%d error(s))",
                source_path or "<no path>",
                len(exc.errors()),
            )
            log.debug("Full validation error: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Per-event-type handlers (each mutates steps_data in place)
    # ------------------------------------------------------------------ #

    def _handle_user(
        self,
        event: dict[str, Any],
        steps_data: list[dict[str, Any]],
        tool_use_id_to_step_idx: dict[str, int],
        unmatched_box: list[int],
    ) -> None:
        message = event.get("message")
        if not isinstance(message, dict):
            return
        content = message.get("content")

        if isinstance(content, str):
            text = content.strip()
            if not text:
                return
            steps_data.append(
                {
                    "step_id": 0,
                    "source": "user",
                    "timestamp": event.get("timestamp"),
                    "message": text,
                    "extra": self._step_extra(event),
                }
            )
            return

        if not isinstance(content, list):
            return

        tool_result_blocks = [
            b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"
        ]
        text_blocks = [
            b for b in content if isinstance(b, dict) and b.get("type") == "text"
        ]

        if tool_result_blocks:
            # This event is a vehicle for tool results -- not a Step.
            tool_result_extras = _tool_result_extras(event)
            for block in tool_result_blocks:
                tool_use_id = block.get("tool_use_id")
                if not isinstance(tool_use_id, str):
                    continue
                idx = tool_use_id_to_step_idx.get(tool_use_id)
                if idx is None:
                    unmatched_box[0] += 1
                    log.warning(
                        "ClaudeCodeAdapter: tool_result with no matching tool_use_id=%s "
                        "(likely from a pre-compaction turn)",
                        tool_use_id,
                    )
                    continue
                target_step = steps_data[idx]
                obs = target_step.setdefault("observation", {"results": []})
                result_dict: dict[str, Any] = {
                    "source_call_id": tool_use_id,
                    "content": _extract_tool_result_content(block),
                }
                if block.get("is_error"):
                    result_dict.setdefault("extra", {})["is_error"] = True
                if tool_result_extras:
                    result_dict.setdefault("extra", {}).update(tool_result_extras)
                obs["results"].append(result_dict)
            return

        # No tool_results -- just plain text blocks the user sent.
        if text_blocks:
            joined = "\n".join(
                b.get("text", "") for b in text_blocks if isinstance(b.get("text"), str)
            ).strip()
            if joined:
                steps_data.append(
                    {
                        "step_id": 0,
                        "source": "user",
                        "timestamp": event.get("timestamp"),
                        "message": joined,
                        "extra": self._step_extra(event),
                    }
                )

    def _handle_assistant(
        self,
        event: dict[str, Any],
        steps_data: list[dict[str, Any]],
        tool_use_id_to_step_idx: dict[str, int],
    ) -> None:
        message = event.get("message")
        if not isinstance(message, dict):
            return
        content_blocks = message.get("content")
        if not isinstance(content_blocks, list):
            content_blocks = []

        text, reasoning, tool_calls = _walk_assistant_content(content_blocks)

        step: dict[str, Any] = {
            "step_id": 0,
            "source": "agent",
            "timestamp": event.get("timestamp"),
            "model_name": message.get("model"),
            "message": text,
        }
        if reasoning:
            step["reasoning_content"] = reasoning
        if tool_calls:
            step["tool_calls"] = tool_calls
        metrics = _build_metrics(message.get("usage"))
        if metrics:
            step["metrics"] = metrics
        step_extra = self._step_extra(event)
        if step_extra:
            step["extra"] = step_extra

        steps_data.append(step)
        step_idx = len(steps_data) - 1
        for tc in tool_calls:
            tool_use_id_to_step_idx[tc["tool_call_id"]] = step_idx

    def _handle_system(
        self,
        event: dict[str, Any],
        steps_data: list[dict[str, Any]],
    ) -> None:
        content = event.get("content")
        if not isinstance(content, str) or not content.strip():
            # E.g. compact_boundary system events have null content.
            return
        step_extra = self._step_extra(event)
        if event.get("subtype") is not None:
            step_extra = dict(step_extra or {})
            step_extra["subtype"] = event["subtype"]
        steps_data.append(
            {
                "step_id": 0,
                "source": "system",
                "timestamp": event.get("timestamp"),
                "message": content,
                "extra": step_extra,
            }
        )

    @staticmethod
    def _step_extra(event: dict[str, Any]) -> dict[str, Any] | None:
        """Preserve Claude Code's graph metadata on each Step.extra."""
        extra: dict[str, Any] = {}
        for key in ("parentUuid", "promptId", "uuid", "isSidechain", "requestId"):
            value = event.get(key)
            if value is not None:
                extra[key] = value
        return extra or None


__all__ = ["ClaudeCodeAdapter"]
