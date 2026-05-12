"""End-to-end ingest pipeline: detect -> adapt -> redact -> store.

Single entry point: :func:`ingest_file`. The function always stores
``raw_blob`` even when the adapter fails or the format is unknown -- that's
the medallion / dual-storage guarantee from the project brief. The ``atif``
column stays NULL only when no canonical representation could be produced.

PII redaction is a deliberate stub here: :func:`redact` is the identity.
Costs five lines, signals we've thought about the problem, leaves the
interface ready for the real implementation when the marketplace starts
accepting user uploads.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

from harbor.models.trajectories import Trajectory
from trace_marketplace.adapters import ADAPTERS
from trace_marketplace.ingest.detect import FormatName, detect_format
from trace_marketplace.storage.db import insert_trace

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    """What :func:`ingest_file` reports back to the caller."""

    trace_id: str
    source_format: FormatName
    validated: bool
    """``True`` if the ``atif`` column was populated, ``False`` if NULL."""


def redact(trajectory: Trajectory) -> Trajectory:
    """STUB: identity. Eventually scrubs PII from messages, paths, tool args.

    Lives here (not in the adapters) because redaction is a cross-cutting
    concern that should be applied uniformly regardless of source format.
    """
    return trajectory


def _derive_has_error(trajectory: Trajectory) -> int | None:
    """Translate ``trajectory.extra["resolved"]`` (bool) into the ``has_error`` column.

    Generic plumbing so any adapter can opt in by writing a single key:

        trajectory.extra["resolved"] = True   -> has_error = 0
        trajectory.extra["resolved"] = False  -> has_error = 1
        (key missing or non-bool)             -> has_error = None  (NULL)

    The double negation (``resolved`` -> ``not resolved``) is awkward but
    matches the existing column semantics: ``has_error=1`` means "this trace
    is a failure," and benchmark datasets ground that in their own
    success/failure label.
    """
    extra = trajectory.extra or {}
    resolved = extra.get("resolved")
    if isinstance(resolved, bool):
        return int(not resolved)
    return None


def _trace_id_for(trajectory: Trajectory | None, path: Path) -> str:
    """Pick a stable primary key for the ``traces`` row.

    Preference order:
    1. ATIF ``trajectory_id`` (v1.7+, document-scoped)
    2. ATIF ``session_id`` (v1.2+, run-scoped, fine for our corpora)
    3. The file's stem (e.g. ``claude-cmux-docs-2026-05-05``)
    4. A random uuid4 (only if even the path is empty)
    """
    if trajectory is not None:
        for candidate in (
            getattr(trajectory, "trajectory_id", None),
            trajectory.session_id,
        ):
            if isinstance(candidate, str) and candidate:
                return candidate
    if path.stem:
        return path.stem
    return str(uuid.uuid4())


def ingest_file(path: Path, conn: sqlite3.Connection) -> IngestResult:
    """Ingest a single file. Idempotent (INSERT OR REPLACE on trace id)."""
    raw_text = path.read_text(encoding="utf-8", errors="replace")

    fmt: FormatName = detect_format(path)
    adapter = ADAPTERS.get(fmt)

    trajectory: Trajectory | None = None
    if adapter is not None:
        try:
            trajectory = adapter.to_atif(raw_text, source_path=path)
        except Exception as exc:
            log.warning(
                "Adapter %r raised on %s: %s (storing raw_blob only)",
                fmt,
                path.name,
                exc,
            )
            trajectory = None
    else:
        log.info(
            "No adapter registered for format %r (path=%s); storing raw_blob only",
            fmt,
            path.name,
        )

    if trajectory is not None:
        trajectory = redact(trajectory)

    if trajectory is not None:
        atif_text = trajectory.model_dump_json(exclude_none=True)
        agent_name = trajectory.agent.name
        model = trajectory.agent.model_name
        num_steps = len(trajectory.steps)
        num_tool_calls = sum(len(s.tool_calls or []) for s in trajectory.steps)
        has_error = _derive_has_error(trajectory)
    else:
        atif_text = None
        agent_name = None
        model = None
        num_steps = None
        num_tool_calls = None
        has_error = None

    trace_id = _trace_id_for(trajectory, path)

    insert_trace(
        conn,
        trace_id=trace_id,
        source_format=fmt,
        raw_blob=raw_text,
        atif=atif_text,
        agent_name=agent_name,
        model=model,
        num_steps=num_steps,
        num_tool_calls=num_tool_calls,
        has_error=has_error,
    )

    return IngestResult(
        trace_id=trace_id,
        source_format=fmt,
        validated=trajectory is not None,
    )


__all__ = ["IngestResult", "ingest_file", "redact"]
