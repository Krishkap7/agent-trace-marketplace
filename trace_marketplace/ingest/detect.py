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

# Re-export the format-shape constants the adapters already own, so the
# detector and the adapter can never disagree on what counts as a valid
# row of a given format. Caught by Cursor bugbot on PR #2 commit 3dd7a3e:
# slice 2.5 / 2.6 had two independently-edited copies of the SWE-agent
# and Codex frozensets in detect.py vs the adapter modules, with no
# obvious link between them.
from trace_marketplace.adapters.codex import (
    OUTER_EVENT_TYPES as CODEX_OUTER_EVENT_TYPES,
)
from trace_marketplace.adapters.swe_agent import (
    REQUIRED_TOP_LEVEL_KEYS as SWE_AGENT_REQUIRED_KEYS,
)

log = logging.getLogger(__name__)

FormatName = Literal["atif", "claude_code", "codex", "cursor", "swe_agent", "unknown"]

# Top-level keys a Cursor v1 export envelope must carry. ``messages`` is
# the discriminator from ATIF (uses ``steps``) and from SWE-agent (uses
# ``trajectory``). This constant has no adapter-side twin -- the Cursor
# adapter probes for these fields inline -- so it's owned here.
CURSOR_REQUIRED_KEYS: frozenset[str] = frozenset({"session_id", "messages"})

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


def _looks_like_cursor_envelope(obj: dict) -> bool:
    """Structural fingerprint check for a Cursor v1 export envelope.

    Requires top-level ``session_id`` + ``messages`` (a list), and that
    the first message dict carries a ``role`` field. The ``role``
    requirement is what distinguishes us from a generic ``{messages: [...]}``
    document and from ATIF (which uses ``steps``).
    """
    if not CURSOR_REQUIRED_KEYS.issubset(obj.keys()):
        return False
    messages = obj.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    first = messages[0]
    return isinstance(first, dict) and "role" in first


def _first_non_empty_line(path: Path, max_bytes: int = 200_000) -> str | None:
    """Return the first non-empty line of ``path``, or ``None``.

    Returns ``None`` (not raises) on any I/O failure -- missing file,
    permission denied, is-a-directory -- so callers preserve the
    "never raises" contract documented on :func:`detect_format`.
    Logs at DEBUG so the failure is still observable in verbose runs.
    """
    try:
        fh = path.open("r", encoding="utf-8", errors="replace")
    except OSError as exc:
        log.debug("_first_non_empty_line: %s not openable: %s", path, exc)
        return None
    try:
        read = 0
        for line in fh:
            read += len(line)
            stripped = line.strip()
            if stripped:
                return stripped
            if read >= max_bytes:
                break
    except OSError as exc:
        log.debug("_first_non_empty_line: %s read error: %s", path, exc)
        return None
    finally:
        fh.close()
    return None


def detect_format(path: Path) -> FormatName:
    """Sniff ``path`` and return one of the supported :data:`FormatName` values.

    The function never raises; on any I/O or parsing problem it logs at
    DEBUG and returns ``"unknown"`` so the pipeline can still store the raw
    bytes.
    """
    suffix = path.suffix.lower()
    # ``.cursor.json`` is treated as a stronger Cursor signal than the
    # structural check below; we still verify the file parses as JSON.
    cursor_extension_hint = path.name.lower().endswith(".cursor.json")

    if suffix == ".json":
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            obj = json.loads(content)
        except (OSError, json.JSONDecodeError) as exc:
            log.debug("detect_format: %s not parseable as JSON: %s", path, exc)
            return "unknown"
        if isinstance(obj, dict):
            # Cursor check runs before SWE-agent/ATIF because the
            # ``.cursor.json`` filename hint is unambiguous and the
            # structural fingerprint (session_id + messages[i].role) is
            # disjoint from the other JSON formats.
            if cursor_extension_hint or _looks_like_cursor_envelope(obj):
                return "cursor"
            # SWE-agent rows are pure JSON without a schema_version, so
            # this branch must check next to avoid wrongly classifying
            # them as "unknown".
            if SWE_AGENT_REQUIRED_KEYS.issubset(obj.keys()):
                return "swe_agent"
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
        # Codex check runs before Claude Code: its vocabulary
        # (session_meta / response_item / event_msg / turn_context) is
        # disjoint from Claude Code's, and Codex events typically lack
        # the camelCase `sessionId` field Claude Code requires.
        if isinstance(event_type, str) and event_type in CODEX_OUTER_EVENT_TYPES:
            return "codex"
        if (
            isinstance(event_type, str)
            and event_type in CLAUDE_CODE_FIRST_EVENT_TYPES
            and "sessionId" in event
        ):
            return "claude_code"
        return "unknown"

    return "unknown"


__all__ = [
    "FormatName",
    "detect_format",
    "CLAUDE_CODE_FIRST_EVENT_TYPES",
    "CODEX_OUTER_EVENT_TYPES",
    "CURSOR_REQUIRED_KEYS",
    "SWE_AGENT_REQUIRED_KEYS",
]
