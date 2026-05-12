"""Cursor session (v1 export format) -> ATIF adapter.

Cursor exposes two ways to get a chat out of the IDE:

1. **The per-workspace SQLite DB** the IDE itself writes
   (``~/Library/Application Support/Cursor/User/workspaceStorage/<hash>/state.vscdb``
   on macOS, similar paths on Linux/Windows). Schema is undocumented and
   shifts between IDE versions; community tools like
   [cursor-chat-export](https://github.com/somogyijanos/cursor-chat-export)
   reverse-engineer it.
2. **Cursor's official CLI** ``cursor-agent --print --output-format
   stream-json``, which emits a documented NDJSON event stream
   (``system`` / ``user`` / ``assistant`` / ``tool_call`` / ``result``)
   per [cursor.com/docs/cli/reference/output-format](https://cursor.com/docs/cli/reference/output-format).

Neither matches this adapter. This adapter reads a **v1 project-defined
JSON envelope** that callers (or a downstream conversion script) produce
from either source above. We deliberately picked this path for v1
because the envelope is the smallest stable shape that lets us ingest
*any* Cursor chat -- IDE SQLite extractions and CLI stream-json runs can
both be flattened into it. Native ingest of either canonical shape is a
v2 deferral so we can ship the search loop first.

Expected envelope shape (the format we accept as v1):

```jsonc
{
  "session_id": "uuid-string",
  "cursor_version": "0.45.0",          // optional
  "workspace_path": "/Users/foo/proj", // optional
  "model": "claude-3.5-sonnet",        // optional
  "synthetic": true,                   // optional, only on our generator output
  "messages": [
    {
      "role": "user" | "assistant",
      "content": "string OR array of content blocks",
      "timestamp": "2026-01-01T12:00:00Z",  // optional ISO 8601
      "tool_calls": [                       // optional, only on assistant
        {"id": "tc-1", "name": "edit_file", "arguments": {"path": "...", "edit": "..."}}
      ],
      "tool_results": [                     // optional, response to prior tool_calls
        {"tool_call_id": "tc-1", "content": "..."}
      ]
    }
  ]
}
```

Mapping rules (lossy is acceptable per slice spec):

- One ``messages[i]`` -> one ATIF Step (``user`` -> source=user,
  ``assistant`` -> source=agent).
- Assistant ``tool_calls`` attach to that step's ``tool_calls`` field.
- ``tool_results`` -- whether in the SAME message or a later one -- are
  looked up by ``tool_call_id`` and appended to the prior agent step's
  ``Observation.results``. Results with no matching tool_call are
  counted in ``trajectory.extra.unmatched_tool_results``.
- ``content`` can be a plain string OR an Anthropic-style array of typed
  blocks (``[{"type": "text", "text": "..."}]``); we accept both.

Like the other adapters, this is a pure function. Malformed input
returns ``None``; the raw blob is preserved in the ``traces`` table.
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
AGENT_NAME = "cursor"

# Roles we honour; everything else is logged at DEBUG and skipped.
CURSOR_ROLE_TO_SOURCE: dict[str, str] = {
    "user": "user",
    "assistant": "agent",
    "system": "system",
}


def _extract_message_text(content: Any) -> str:
    """Coerce a Cursor message ``content`` field into plain text.

    Cursor exports occasionally adopt the Anthropic content-blocks shape
    (``[{"type": "text", "text": "..."}]``). We accept both string and
    list-of-blocks. Non-text blocks (image refs etc.) are silently
    dropped -- they aren't useful for failure-mode search.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
            elif block.get("type") in (None, "text"):
                # Some exporters put text directly under different keys.
                pass
    return "\n".join(parts)


def _build_tool_call_dict(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Normalise one entry from ``messages[i].tool_calls``.

    Returns ``None`` if the entry doesn't have at minimum an ``id`` and
    a ``name``; we'd rather drop one malformed call than fail the whole
    trajectory.
    """
    tc_id = raw.get("id") or raw.get("tool_call_id")
    name = raw.get("name") or raw.get("function_name")
    if not isinstance(tc_id, str) or not isinstance(name, str):
        log.debug(
            "CursorAdapter: skipping malformed tool_call (id=%r, name=%r)",
            tc_id,
            name,
        )
        return None
    args = raw.get("arguments")
    if args is None:
        args = raw.get("input")
    if not isinstance(args, dict):
        # Lossy: keep whatever the model emitted under a stable key.
        args = {} if args is None else {"_raw": args}
    return {
        "tool_call_id": tc_id,
        "function_name": name,
        "arguments": args,
    }


def _extract_tool_result_content(raw: dict[str, Any]) -> str:
    """Best-effort string from a tool_result entry's ``content`` field."""
    content = raw.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for sub in content:
            if isinstance(sub, dict):
                text = sub.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    if content is None:
        return ""
    # Fall back to JSON dump for dicts/numbers; lossy but never empty.
    try:
        return json.dumps(content, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(content)


def _normalise_timestamp(value: Any) -> str | None:
    """ISO 8601 strings only; everything else is dropped."""
    if isinstance(value, str) and value:
        return value
    return None


class CursorAdapter(BaseAdapter):
    """Convert a Cursor v1 export envelope into an ATIF :class:`Trajectory`.

    One file = one session = one Trajectory row. Step ids are renumbered
    sequentially from 1. ``tool_results`` linking is "first matching prior
    agent step wins"; orphans (no matching ``tool_call_id``) are tallied
    in ``trajectory.extra.unmatched_tool_results``.
    """

    source_format = "cursor"

    def to_atif(
        self,
        raw_text: str,
        source_path: Path | None = None,
    ) -> Trajectory | None:
        try:
            envelope = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            log.warning(
                "CursorAdapter: invalid JSON in %s: %s",
                source_path or "<no path>",
                exc,
            )
            return None

        if not isinstance(envelope, dict):
            log.warning(
                "CursorAdapter: top-level is %s, expected object (%s)",
                type(envelope).__name__,
                source_path or "<no path>",
            )
            return None

        session_id = envelope.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            if source_path is not None:
                # cursor-chat-export sometimes elides session_id; fall
                # back to filename stem so we still get a unique row.
                session_id = source_path.stem
            else:
                log.warning(
                    "CursorAdapter: missing session_id in %s",
                    source_path or "<no path>",
                )
                return None

        messages = envelope.get("messages")
        if not isinstance(messages, list) or not messages:
            log.warning(
                "CursorAdapter: messages missing/empty for %s",
                session_id,
            )
            return None

        steps_data: list[dict[str, Any]] = []
        tc_id_to_step_idx: dict[str, int] = {}
        unmatched_results: int = 0
        unknown_roles: dict[str, int] = {}

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            source = CURSOR_ROLE_TO_SOURCE.get(role) if isinstance(role, str) else None
            if source is None:
                key = role if isinstance(role, str) else "<no-role>"
                unknown_roles[key] = unknown_roles.get(key, 0) + 1
                continue

            text = _extract_message_text(msg.get("content"))
            tool_calls_raw = msg.get("tool_calls")
            tool_calls: list[dict[str, Any]] = []
            if isinstance(tool_calls_raw, list):
                for tc_raw in tool_calls_raw:
                    if isinstance(tc_raw, dict):
                        tc_dict = _build_tool_call_dict(tc_raw)
                        if tc_dict is not None:
                            tool_calls.append(tc_dict)

            tool_results_raw = msg.get("tool_results")

            # Attach tool_results to the prior agent step they reference,
            # NOT to this message. We do this even when the current
            # message has its own text + tool_calls.
            if isinstance(tool_results_raw, list):
                for tr_raw in tool_results_raw:
                    if not isinstance(tr_raw, dict):
                        continue
                    tc_id = tr_raw.get("tool_call_id") or tr_raw.get("id")
                    if not isinstance(tc_id, str):
                        unmatched_results += 1
                        continue
                    target_idx = tc_id_to_step_idx.get(tc_id)
                    if target_idx is None:
                        unmatched_results += 1
                        log.debug(
                            "CursorAdapter: tool_result with no matching tool_call_id=%s",
                            tc_id,
                        )
                        continue
                    target_step = steps_data[target_idx]
                    obs = target_step.setdefault("observation", {"results": []})
                    obs["results"].append(
                        {
                            "source_call_id": tc_id,
                            "content": _extract_tool_result_content(tr_raw),
                        }
                    )

            # If this is a pure tool_results carrier (no text, no own
            # tool_calls), it's not a Step on its own -- skip after
            # attaching results above.
            if not text.strip() and not tool_calls:
                continue

            step: dict[str, Any] = {
                "step_id": 0,
                "source": source,
                "message": text,
            }
            timestamp = _normalise_timestamp(msg.get("timestamp"))
            if timestamp is not None:
                step["timestamp"] = timestamp
            if tool_calls:
                step["tool_calls"] = tool_calls
            if source == "agent":
                model = envelope.get("model")
                if isinstance(model, str) and model:
                    step["model_name"] = model
            steps_data.append(step)
            step_idx = len(steps_data) - 1
            for tc in tool_calls:
                tc_id_to_step_idx[tc["tool_call_id"]] = step_idx

        if not steps_data:
            log.warning(
                "CursorAdapter: produced 0 steps for %s",
                session_id,
            )
            return None

        for new_id, step in enumerate(steps_data, start=1):
            step["step_id"] = new_id

        agent_dict: dict[str, Any] = {
            "name": AGENT_NAME,
        }
        cursor_version = envelope.get("cursor_version")
        agent_dict["version"] = (
            cursor_version
            if isinstance(cursor_version, str) and cursor_version
            else "unknown"
        )
        model = envelope.get("model")
        if isinstance(model, str) and model:
            agent_dict["model_name"] = model

        traj_extra: dict[str, Any] = {}
        workspace_path = envelope.get("workspace_path")
        if isinstance(workspace_path, str) and workspace_path:
            traj_extra["workspace_path"] = workspace_path
        if envelope.get("synthetic") is True:
            traj_extra["synthetic"] = True
        # ``synthetic_failure_mode`` is a free-form short string label
        # (e.g. ``"broke_working_code"``) that our generator attaches so
        # downstream failure-mode search has ground-truth tags to
        # evaluate against. Absent on real user uploads.
        failure_mode = envelope.get("synthetic_failure_mode")
        if isinstance(failure_mode, str) and failure_mode:
            traj_extra["synthetic_failure_mode"] = failure_mode
        if unmatched_results:
            traj_extra["unmatched_tool_results"] = unmatched_results
        if unknown_roles:
            traj_extra["unknown_roles"] = unknown_roles

        traj_dict: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "agent": agent_dict,
            "steps": steps_data,
        }
        if traj_extra:
            traj_dict["extra"] = traj_extra

        try:
            return Trajectory.model_validate(traj_dict)
        except ValidationError as exc:
            log.warning(
                "CursorAdapter: validation failed for %s (%d error(s))",
                session_id,
                len(exc.errors()),
            )
            log.debug("Full validation error: %s", exc)
            return None


__all__ = ["CursorAdapter"]
