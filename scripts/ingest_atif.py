"""Ingest ATIF trajectories from ``data/raw/`` into the SQLite ``traces`` table.

Pipeline per trajectory:

1. Read the file as UTF-8 text -> ``raw_blob`` (preserves exact upload bytes).
2. ``json.loads`` -> ``raw_dict``.
3. ``Trajectory.model_validate(raw_dict)`` -> canonical ATIF.
   On failure, log the full Pydantic error and store ``atif = NULL``;
   ``raw_blob`` stays populated so we never silently drop data.
4. Cheap derived columns (``agent_name``, ``model``, ``num_steps``,
   ``num_tool_calls``) come from the validated ATIF (or, when validation
   failed, from the raw dict as a best-effort fallback).
5. ``INSERT OR REPLACE`` keyed on session_id / trajectory_id, so re-running
   the ingest pipeline against the same corpus is idempotent.

Notes:

- The corpus is ATIF-v1.2; harbor's Pydantic models are v1.7. v1.3 -> v1.7
  changes are all additive optional fields, so v1.2 documents *should*
  validate. The first failure is printed in full to help diagnose any real
  incompatibility.
- ``agent_name`` is ATIF ``agent.name`` (the agent SYSTEM, e.g. "mcp"),
  not the per-domain agent name (e.g. "accountant"). The directory name
  carries the domain hint and is not indexed in this slice.

Run:
    uv run python scripts/ingest_atif.py
    uv run python scripts/ingest_atif.py --all           # ingest all 38, not just the stratified 5
    uv run python scripts/ingest_atif.py --agents foo bar  # custom set
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from harbor.models.trajectories import Trajectory
from trace_marketplace.storage.db import (
    DEFAULT_DB_PATH,
    connect,
    count_traces,
    init_schema,
    insert_trace,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"

# One agent per business vertical: Finance, Health, Operations, Marketing, HR.
# Hard-coded so verify.py shows variety instead of e.g. five finance traces.
STRATIFIED_AGENTS: tuple[str, ...] = (
    "accountant",
    "calorie-counter",
    "inventory-supply-chain",
    "crm-manager",
    "recruiter",
)


def _count_tool_calls(steps: list[dict[str, Any]]) -> int:
    return sum(len(step.get("tool_calls") or []) for step in steps)


# ---------------------------------------------------------------------------
# v1.2 -> v1.7 normalization
#
# The obaydata corpus declares ``schema_version: "ATIF-v1.2"`` but uses two
# field names that harbor's v1.7 Pydantic model rejects with extra_forbidden:
#
#   step.usage           -> step.metrics
#   step.usage.cache_tokens -> step.metrics.cached_tokens
#
# Whether this is an obaydata-specific quirk or a real v1.2 spec detail we
# can't fully tell without an archived v1.2 schema, but the mapping is
# unambiguous (same numeric semantics, just renamed). We apply it here so
# slice 1's "raw_blob + parsed atif" bar is met. The raw_blob column still
# stores the original bytes, so the un-normalized form is never lost.
# ---------------------------------------------------------------------------
def _normalize_step_metrics(raw_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``raw_dict`` with ``usage`` renamed to ``metrics``."""
    out = dict(raw_dict)
    new_steps: list[dict[str, Any]] = []
    for step in raw_dict.get("steps") or []:
        if not isinstance(step, dict):
            new_steps.append(step)
            continue
        new_step = dict(step)
        if "usage" in new_step and "metrics" not in new_step:
            metrics = dict(new_step.pop("usage") or {})
            if "cache_tokens" in metrics and "cached_tokens" not in metrics:
                metrics["cached_tokens"] = metrics.pop("cache_tokens")
            new_step["metrics"] = metrics
        new_steps.append(new_step)
    out["steps"] = new_steps
    return out


def _derived_from_dict(raw_dict: dict[str, Any]) -> dict[str, Any]:
    """Pull derived columns from a raw ATIF dict (used as a fallback)."""
    agent = raw_dict.get("agent") or {}
    steps = raw_dict.get("steps") or []
    return {
        "agent_name": agent.get("name"),
        "model": agent.get("model_name"),
        "num_steps": len(steps),
        "num_tool_calls": _count_tool_calls(steps),
    }


def _derived_from_trajectory(traj: Trajectory) -> dict[str, Any]:
    """Pull derived columns from a validated Trajectory."""
    return {
        "agent_name": traj.agent.name,
        "model": getattr(traj.agent, "model_name", None),
        "num_steps": len(traj.steps),
        "num_tool_calls": sum(len(s.tool_calls or []) for s in traj.steps),
    }


def _trace_id(raw_dict: dict[str, Any]) -> str:
    """Stable per-trace primary key.

    Prefers ATIF ``trajectory_id`` (v1.7+, doc-scoped), then ``session_id``
    (v1.2+, run-scoped but unique enough for this corpus), and finally a
    random uuid4 if both are absent.
    """
    return (
        raw_dict.get("trajectory_id")
        or raw_dict.get("session_id")
        or str(uuid.uuid4())
    )


def ingest_one(
    conn,
    path: Path,
    *,
    first_failure_printed: list[bool],
) -> tuple[bool, str]:
    """Ingest a single trajectory.json. Returns ``(validated, trace_id)``."""
    raw_blob = path.read_text(encoding="utf-8")
    raw_dict = json.loads(raw_blob)
    normalized = _normalize_step_metrics(raw_dict)

    atif_text: str | None
    try:
        trajectory = Trajectory.model_validate(normalized)
    except ValidationError as exc:
        atif_text = None
        derived = _derived_from_dict(raw_dict)
        validated = False
        if not first_failure_printed[0]:
            first_failure_printed[0] = True
            print(
                f"\nFIRST VALIDATION FAILURE for {path.name}:\n{exc}\n",
                file=sys.stderr,
            )
        else:
            print(
                f"  validation failed for {path.name}: "
                f"{len(exc.errors())} error(s) (atif=NULL, raw_blob retained)",
                file=sys.stderr,
            )
    else:
        # We round-trip through Pydantic so the stored ATIF is canonical
        # (default field ordering, None-stripping per Pydantic defaults).
        atif_text = trajectory.model_dump_json(exclude_none=True)
        derived = _derived_from_trajectory(trajectory)
        validated = True

    trace_id = _trace_id(raw_dict)
    insert_trace(
        conn,
        trace_id=trace_id,
        source_format="atif",
        raw_blob=raw_blob,
        atif=atif_text,
        ingested_at=datetime.now(timezone.utc).isoformat(),
        **derived,
    )
    return validated, trace_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Directory containing <agent>/trajectory.json subdirs (default: {DEFAULT_RAW_DIR}).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite database path (default: {DEFAULT_DB_PATH}).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest every trajectory.json under --raw-dir, not just the stratified 5.",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        metavar="AGENT",
        help="Explicit list of agent directory names to ingest. Overrides --all and the stratified default.",
    )
    args = parser.parse_args()

    if args.agents:
        agent_names = list(args.agents)
    elif args.all:
        agent_names = sorted(
            p.parent.name for p in args.raw_dir.glob("*/trajectory.json")
        )
    else:
        agent_names = list(STRATIFIED_AGENTS)

    paths: list[Path] = []
    missing: list[str] = []
    for agent in agent_names:
        candidate = args.raw_dir / agent / "trajectory.json"
        if candidate.is_file():
            paths.append(candidate)
        else:
            missing.append(agent)

    if missing:
        print(
            f"WARNING: {len(missing)} requested agent(s) not found under "
            f"{args.raw_dir}: {missing}",
            file=sys.stderr,
        )
    if not paths:
        print("ERROR: no trajectories to ingest. Did you run download_corpus.py?", file=sys.stderr)
        return 1

    print(f"Ingesting {len(paths)} trajectory(ies) into {args.db}")

    first_failure_printed = [False]
    validated_count = 0
    with connect(args.db) as conn:
        init_schema(conn)
        for path in paths:
            validated, trace_id = ingest_one(
                conn, path, first_failure_printed=first_failure_printed
            )
            status = "OK   " if validated else "RAW  "
            print(f"  {status} {path.parent.name:30s} -> {trace_id}")
            if validated:
                validated_count += 1

        total_rows = count_traces(conn)

    print(
        f"\nDone. {validated_count}/{len(paths)} validated against ATIF v1.7. "
        f"traces table now holds {total_rows} row(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
