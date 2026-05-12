"""Cursor adapter -- STUB. Not implemented in v1.

Cursor stores chat sessions in a SQLite database:
  Mac:     ~/Library/Application Support/Cursor/User/workspaceStorage/<hash>/state.vscdb
  Linux:   ~/.config/Cursor/User/workspaceStorage/<hash>/state.vscdb
  Windows: %APPDATA%/Cursor/User/workspaceStorage/<hash>/state.vscdb

Data is JSON blobs inside SQLite rows in ItemTable (keys starting with 'composer').
Schema is undocumented and changes between Cursor versions -- would require reverse
engineering, similar to cursor-chat-export and ccdiag.

Adapter would:
  1. Accept .vscdb file OR pre-extracted JSON dump
  2. Query composer/chat tables
  3. Parse JSON blobs for messages, tool invocations, edit operations
  4. Map onto ATIF Steps (Cursor 'apply_edit' -> ToolCall function_name='edit_file', etc.)
  5. Stash Cursor-specific edit-acceptance state in step.extra

Deferred to v2: 3-5 hours of reverse engineering displaces time better spent on
failure detection / search / UI. See DESIGN.md.
"""

from __future__ import annotations

from pathlib import Path

from harbor.models.trajectories import Trajectory
from trace_marketplace.adapters.base import BaseAdapter


class CursorAdapter(BaseAdapter):
    """STUB. See module docstring for the reverse-engineering plan."""

    source_format = "cursor"

    def to_atif(
        self,
        raw_text: str,
        source_path: Path | None = None,
    ) -> Trajectory | None:
        raise NotImplementedError(
            "Cursor adapter is intentionally a stub for v1. "
            "See trace_marketplace/adapters/cursor.py module docstring."
        )


__all__ = ["CursorAdapter"]
