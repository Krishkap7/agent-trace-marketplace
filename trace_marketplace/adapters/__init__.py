"""Format-specific trace adapters.

Each adapter converts raw text from one source format into a validated ATIF
``Trajectory`` object (or ``None`` on failure). Adapters are pure functions:
no I/O side effects, no DB writes, no logging side-channels that mutate state.

``ADAPTERS`` is the registry the ingest pipeline dispatches on, keyed by the
strings returned by :func:`trace_marketplace.ingest.detect.detect_format`.
"""

from trace_marketplace.adapters.atif import ATIFAdapter
from trace_marketplace.adapters.base import BaseAdapter
from trace_marketplace.adapters.claude_code import ClaudeCodeAdapter
from trace_marketplace.adapters.codex import CodexAdapter
from trace_marketplace.adapters.cursor import CursorAdapter
from trace_marketplace.adapters.swe_agent import SWEAgentAdapter

ADAPTERS: dict[str, BaseAdapter] = {
    "atif": ATIFAdapter(),
    "claude_code": ClaudeCodeAdapter(),
    "codex": CodexAdapter(),
    "cursor": CursorAdapter(),
    "swe_agent": SWEAgentAdapter(),
}

__all__ = [
    "ADAPTERS",
    "ATIFAdapter",
    "BaseAdapter",
    "ClaudeCodeAdapter",
    "CodexAdapter",
    "CursorAdapter",
    "SWEAgentAdapter",
]
