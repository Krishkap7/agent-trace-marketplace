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

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _identity_from_dict(d: dict[str, Any]) -> str | None:
    """Pull a session/trajectory identity out of one parsed JSON dict.

    Probes every known identity field across the 5 supported formats:

    * snake_case top level: ``trajectory_id`` / ``session_id`` /
      ``instance_id``  (ATIF, Cursor, SWE-agent body)
    * camelCase top level: ``sessionId``       (Claude Code event line)
    * nested              : ``payload.id`` when ``type == "session_meta"``
                            (Codex event line)
    """
    for key in ("trajectory_id", "session_id", "instance_id", "sessionId"):
        value = d.get(key)
        if isinstance(value, str) and value:
            return value
    if d.get("type") == "session_meta":
        payload = d.get("payload")
        if isinstance(payload, dict):
            sid = payload.get("id")
            if isinstance(sid, str) and sid:
                return sid
    return None


def _extract_raw_identity(raw_text: str) -> str | None:
    """Best-effort identity peek for files where Pydantic validation
    failed but the raw bytes still carry a session/trajectory id.

    Without this, every failed validation falls through to
    ``path.stem`` -- and because ATIF benchmark files are uniformly
    named ``trajectory.json``, every failure would collide on the
    single id ``"trajectory"`` and silently clobber the previous row
    via ``INSERT OR REPLACE``. (Caught by Cursor bugbot on PR #2.)

    Tries two interpretations of the raw text:
    1. The whole document as a JSON object  (handles ATIF / Cursor /
       SWE-agent / Codex-or-Claude-Code-with-only-one-event).
    2. Its first non-empty line as a JSON object  (handles real JSONL
       streams where the whole document isn't a single JSON value).

    Returns the first identity match found across both interpretations,
    or ``None``. Caller then falls back to ``path.stem`` then content
    hash.
    """
    stripped = raw_text.strip()
    if not stripped:
        return None
    try:
        whole = json.loads(stripped)
    except json.JSONDecodeError:
        whole = None
    if isinstance(whole, dict):
        identity = _identity_from_dict(whole)
        if identity is not None:
            return identity
    first_line = next(
        (line.strip() for line in stripped.splitlines() if line.strip()),
        None,
    )
    if first_line is None or first_line == stripped:
        # Either no lines, or the whole document was already a single
        # line we just probed above -- no second interpretation possible.
        return None
    try:
        event = json.loads(first_line)
    except json.JSONDecodeError:
        return None
    if isinstance(event, dict):
        return _identity_from_dict(event)
    return None


def _trace_id_for(
    trajectory: Trajectory | None, path: Path, raw_text: str
) -> str:
    """Pick a stable primary key for the ``traces`` row.

    Preference order, all deterministic:
    1. ATIF ``trajectory_id``   (v1.7+, validated)
    2. ATIF ``session_id``      (v1.2+, validated, fine for our corpora)
    3. Raw-text identity peek   (validation-failure path; covers all 5
       formats' on-the-wire identity fields)
    4. ``path.stem``            (e.g. ``claude-cmux-docs-2026-05-05``)
    5. ``sha256:<hex>``         over the raw bytes -- content-addressable
       leaf that guarantees the documented idempotency contract.

    Two earlier bugs the layering closes:

    * uuid4 leaf (caught by bugbot on PR #1) broke idempotency for
      empty-stem paths; now we hash the bytes.
    * path.stem fallback (caught by bugbot on PR #2) collided every
      failed-validation ATIF row onto the single id ``"trajectory"``
      because the benchmark corpus uniformly uses that filename. The
      raw-text identity peek above closes that gap.
    """
    if trajectory is not None:
        for candidate in (
            getattr(trajectory, "trajectory_id", None),
            trajectory.session_id,
        ):
            if isinstance(candidate, str) and candidate:
                return candidate
    raw_identity = _extract_raw_identity(raw_text)
    if raw_identity is not None:
        return raw_identity
    if path.stem:
        return path.stem
    digest = hashlib.sha256(raw_text.encode("utf-8", errors="replace")).hexdigest()
    return f"sha256:{digest[:32]}"


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

    trace_id = _trace_id_for(trajectory, path, raw_text)

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
