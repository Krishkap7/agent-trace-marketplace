"""SWE-agent (Nebius HF dataset row) -> ATIF adapter.

Input shape (one HF row, after pyarrow auto-deserialisation):

```
{
  "instance_id":     "AnalogJ__lexicon-336",
  "model_name":      "swe-agent-llama-70b",
  "target":          False,
  "exit_status":     "submitted (exit_context)",
  "generated_patch": "diff --git ...",
  "eval_logs":       "...",
  "trajectory": [
    {"role": "system", "system_prompt": "SETTING: ...", "text": "",  "mask": False, "cutoff_date": "01.01.2023"},
    {"role": "user",   "system_prompt": "",             "text": "...", "mask": False, "cutoff_date": None},
    {"role": "ai",     "system_prompt": "",             "text": "...", "mask": True,  "cutoff_date": None},
    ...
  ]
}
```

Two important deviations from what the dataset card and the original prompt
described, both confirmed during Step-0 inspection:

1. Per-entry content lives in ``text`` (and ``system_prompt`` for the system
   entry). The card and the prompt both said ``content``; that field does
   not exist.
2. ``trajectory`` is a real list of structs at the parquet level, not a
   JSON-encoded string. We don't need to ``json.loads`` it.

Mapping decisions (locked):

* ``role: "system"`` (always trajectory[0]) -> ``Step(source="system")`` from
  ``system_prompt``.
* ``role: "user"`` -> ``Step(source="system")``. These entries are SWE-agent
  harness output (file editor state, command stdout, etc.), NOT real user
  messages. Mapping them to "system" is more semantically honest than
  faking them as "user."
* ``role: "ai"`` -> ``Step(source="agent")``. Full content goes in
  ``message``. We deliberately do NOT regex-parse "Thought:"/"Action:" out
  of the raw string -- lossy on purpose; raw_blob keeps full fidelity.
* ``target`` -> ``Trajectory.extra["resolved"] = bool``. Pipeline derives
  ``has_error`` from this.
* ``exit_status``, ``generated_patch`` -> ``Trajectory.extra``.
* ``eval_logs`` -> ``Trajectory.extra["eval_logs"]`` truncated to 10 KB so
  the silver column doesn't bloat. Full text remains in raw_blob.
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
AGENT_NAME = "swe-agent"
EVAL_LOGS_TRUNCATE_BYTES = 10_000

# Top-level keys every SWE-agent HF row carries (verified during slice 2.5
# Step-0 inspection of nebius/SWE-agent-trajectories shard 0).
#
# SINGLE SOURCE OF TRUTH for this set: ``trace_marketplace.ingest.detect``
# imports this constant (aliased to ``SWE_AGENT_REQUIRED_KEYS``) so the
# format detector and the adapter cannot disagree on what counts as a
# valid SWE-agent row. Edit here, never in the detector.
REQUIRED_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {"instance_id", "model_name", "target", "trajectory"}
)


def _truncate(s: str, limit: int) -> str:
    """Return ``s`` capped at ``limit`` bytes with a marker suffix."""
    if len(s) <= limit:
        return s
    dropped = len(s) - limit
    return s[:limit] + f"\n...[truncated, {dropped} more bytes]"


def _entry_text(entry: dict[str, Any]) -> str:
    """Return the textual content of one trajectory entry.

    The system entry uses ``system_prompt``; user/ai entries use ``text``.
    Falls back to whichever is non-empty.
    """
    for key in ("text", "system_prompt"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _entry_extra(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Stash per-step SWE-agent metadata (mask, cutoff_date) on Step.extra."""
    extra: dict[str, Any] = {}
    if (mask := entry.get("mask")) is not None:
        extra["mask"] = bool(mask)
    cutoff = entry.get("cutoff_date")
    if isinstance(cutoff, str) and cutoff:
        extra["cutoff_date"] = cutoff
    return extra or None


def _build_trajectory_extra(row: dict[str, Any]) -> dict[str, Any]:
    """Translate row-level SWE-agent metadata into Trajectory.extra.

    ``target`` is normalised to a real Python bool under the ``resolved``
    key so the pipeline's ``_derive_has_error`` can recognise it.
    ``eval_logs`` is truncated; the full thing stays in raw_blob.
    """
    extra: dict[str, Any] = {}
    target = row.get("target")
    if isinstance(target, bool):
        extra["resolved"] = target

    exit_status = row.get("exit_status")
    if isinstance(exit_status, str) and exit_status:
        extra["exit_status"] = exit_status

    patch = row.get("generated_patch")
    if isinstance(patch, str) and patch.strip():
        extra["generated_patch"] = patch

    eval_logs = row.get("eval_logs")
    if isinstance(eval_logs, str) and eval_logs:
        extra["eval_logs"] = _truncate(eval_logs, EVAL_LOGS_TRUNCATE_BYTES)
        extra["eval_logs_full_bytes"] = len(eval_logs)

    return extra


class SWEAgentAdapter(BaseAdapter):
    """Convert one Nebius SWE-agent HF row into an ATIF :class:`Trajectory`.

    The adapter takes the row already serialised to a JSON file on disk
    (that's what the bronze layer holds). One file = one HF row = one
    Trajectory. Step ids are renumbered sequentially from 1.
    """

    source_format = "swe_agent"

    def to_atif(
        self,
        raw_text: str,
        source_path: Path | None = None,
    ) -> Trajectory | None:
        try:
            row = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            log.warning(
                "SWEAgentAdapter: invalid JSON in %s: %s",
                source_path or "<no path>",
                exc,
            )
            return None

        if not isinstance(row, dict):
            log.warning(
                "SWEAgentAdapter: top-level value is %s, expected object (%s)",
                type(row).__name__,
                source_path or "<no path>",
            )
            return None

        if not REQUIRED_TOP_LEVEL_KEYS.issubset(row.keys()):
            missing = REQUIRED_TOP_LEVEL_KEYS - row.keys()
            log.warning(
                "SWEAgentAdapter: missing required keys %s in %s",
                sorted(missing),
                source_path or "<no path>",
            )
            return None

        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            log.warning(
                "SWEAgentAdapter: instance_id missing or non-string in %s",
                source_path or "<no path>",
            )
            return None

        model_name = row.get("model_name")
        model_name_str = (
            model_name if isinstance(model_name, str) and model_name else None
        )

        traj_entries = row.get("trajectory")
        if not isinstance(traj_entries, list) or not traj_entries:
            log.warning(
                "SWEAgentAdapter: trajectory missing/empty for %s",
                instance_id,
            )
            return None

        steps_data: list[dict[str, Any]] = []
        unknown_role_counts: dict[str, int] = {}

        for entry in traj_entries:
            if not isinstance(entry, dict):
                continue
            role = entry.get("role")
            text = _entry_text(entry)
            if not text.strip():
                # Skip empty entries -- harmless and avoids ATIF requiring
                # non-empty messages.
                continue

            if role == "system" or role == "user":
                # NOTE: role="user" entries are env observations from the
                # SWE-agent harness, not real user input. Mapping them to
                # ATIF source="system" is the honest call.
                source = "system"
            elif role == "ai":
                source = "agent"
            else:
                key = role if isinstance(role, str) else "<no-role>"
                unknown_role_counts[key] = unknown_role_counts.get(key, 0) + 1
                continue

            step: dict[str, Any] = {
                "step_id": 0,
                "source": source,
                "message": text,
            }
            if source == "agent" and model_name_str:
                step["model_name"] = model_name_str
            entry_extra = _entry_extra(entry)
            if entry_extra:
                step["extra"] = entry_extra
            steps_data.append(step)

        if not steps_data:
            log.warning(
                "SWEAgentAdapter: produced 0 steps for %s (all entries skipped)",
                instance_id,
            )
            return None

        for new_id, step in enumerate(steps_data, start=1):
            step["step_id"] = new_id

        traj_extra = _build_trajectory_extra(row)
        if unknown_role_counts:
            traj_extra["unknown_role_counts"] = unknown_role_counts

        agent_dict: dict[str, Any] = {
            "name": AGENT_NAME,
            "version": "unknown",
            "extra": {"original_format": "swe-agent-hf-row"},
        }
        if model_name_str:
            agent_dict["model_name"] = model_name_str

        traj_dict: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "session_id": instance_id,
            "agent": agent_dict,
            "steps": steps_data,
        }
        if traj_extra:
            traj_dict["extra"] = traj_extra

        try:
            return Trajectory.model_validate(traj_dict)
        except ValidationError as exc:
            log.warning(
                "SWEAgentAdapter: validation failed for %s (%d error(s))",
                instance_id,
                len(exc.errors()),
            )
            log.debug("Full validation error: %s", exc)
            return None


__all__ = ["SWEAgentAdapter", "REQUIRED_TOP_LEVEL_KEYS"]
