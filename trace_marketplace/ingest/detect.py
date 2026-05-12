"""Format detection for raw trace files.

Cheap heuristics only -- reads at most the first non-empty line of a JSONL
file and the top-level keys of a single JSON object. The detector never
parses the full payload; that's the adapter's job.

Returned strings match the keys in
:data:`trace_marketplace.adapters.ADAPTERS`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

FormatName = Literal["atif", "claude_code", "unknown"]

# Event types we accept on the first non-empty line of a JSONL file when
# classifying as Claude Code. Real Fleet data routinely starts with
# `permission-mode`, so we accept structural/metadata events too -- the
# detector must not require a conversational first event.
CLAUDE_CODE_FIRST_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "user",
        "assistant",
        "system",
        "permission-mode",
        "last-prompt",
        "file-history-snapshot",
        "attachment",
        "queue-operation",
        # Types from the Claude Code public spec we may yet encounter.
        "tool_use",
        "tool_result",
        "summary",
    }
)


def _first_non_empty_line(path: Path, max_bytes: int = 200_000) -> str | None:
    """Return the first non-empty line of ``path``, or ``None``."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        read = 0
        for line in fh:
            read += len(line)
            stripped = line.strip()
            if stripped:
                return stripped
            if read >= max_bytes:
                break
    return None


def detect_format(path: Path) -> FormatName:
    """Sniff ``path`` and return one of ``"atif"``, ``"claude_code"``, ``"unknown"``.

    The function never raises; on any I/O or parsing problem it logs at
    DEBUG and returns ``"unknown"`` so the pipeline can still store the raw
    bytes.
    """
    suffix = path.suffix.lower()

    if suffix == ".json":
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            obj = json.loads(content)
        except (OSError, json.JSONDecodeError) as exc:
            log.debug("detect_format: %s not parseable as JSON: %s", path, exc)
            return "unknown"
        if isinstance(obj, dict):
            schema_version = obj.get("schema_version")
            if isinstance(schema_version, str) and schema_version.startswith("ATIF"):
                return "atif"
        return "unknown"

    if suffix == ".jsonl":
        first = _first_non_empty_line(path)
        if first is None:
            log.debug("detect_format: %s has no non-empty lines", path)
            return "unknown"
        try:
            event = json.loads(first)
        except json.JSONDecodeError as exc:
            log.debug("detect_format: %s first line not JSON: %s", path, exc)
            return "unknown"
        if not isinstance(event, dict):
            return "unknown"
        event_type = event.get("type")
        if (
            isinstance(event_type, str)
            and event_type in CLAUDE_CODE_FIRST_EVENT_TYPES
            and "sessionId" in event
        ):
            return "claude_code"
        return "unknown"

    return "unknown"


__all__ = ["FormatName", "detect_format", "CLAUDE_CODE_FIRST_EVENT_TYPES"]
