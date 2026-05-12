"""Ingest ATIF trajectories from ``data/raw/`` into the SQLite ``traces`` table.

Thin wrapper over :func:`trace_marketplace.ingest.pipeline.ingest_file`. The
heavy lifting (JSON parsing, v1.2 metrics normalization, harbor Pydantic
validation, row insertion) now lives in
:mod:`trace_marketplace.adapters.atif` and
:mod:`trace_marketplace.ingest.pipeline` so the same pipeline serves every
source format.

Run:
    uv run python scripts/ingest_atif.py
    uv run python scripts/ingest_atif.py --all
    uv run python scripts/ingest_atif.py --agents accountant recruiter
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from trace_marketplace.ingest.pipeline import ingest_file
from trace_marketplace.storage.db import (
    DEFAULT_DB_PATH,
    connect,
    count_traces,
    init_schema,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"

STRATIFIED_AGENTS: tuple[str, ...] = (
    "accountant",
    "calorie-counter",
    "inventory-supply-chain",
    "crm-manager",
    "recruiter",
)


def _resolve_paths(
    raw_dir: Path, agents: list[str] | None, ingest_all: bool
) -> list[Path]:
    """Translate CLI selection into a list of trajectory.json paths."""
    if agents:
        names = list(agents)
    elif ingest_all:
        names = sorted(p.parent.name for p in raw_dir.glob("*/trajectory.json"))
    else:
        names = list(STRATIFIED_AGENTS)

    paths: list[Path] = []
    missing: list[str] = []
    for name in names:
        candidate = raw_dir / name / "trajectory.json"
        if candidate.is_file():
            paths.append(candidate)
        else:
            missing.append(name)
    if missing:
        logging.warning(
            "%d requested agent(s) not found under %s: %s",
            len(missing),
            raw_dir,
            missing,
        )
    return paths


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
        dest="ingest_all",
        help="Ingest every trajectory.json under --raw-dir, not just the stratified 5.",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        metavar="AGENT",
        help="Explicit list of agent directory names to ingest. Overrides --all and the stratified default.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging (shows full Pydantic validation errors).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    paths = _resolve_paths(args.raw_dir, args.agents, args.ingest_all)
    if not paths:
        print(
            "ERROR: no trajectories to ingest. Did you run download_corpus.py?",
            file=sys.stderr,
        )
        return 1

    print(f"Ingesting {len(paths)} ATIF trajectory(ies) into {args.db}")

    validated_count = 0
    with connect(args.db) as conn:
        init_schema(conn)
        for path in paths:
            result = ingest_file(path, conn)
            status = "OK   " if result.validated else "RAW  "
            print(f"  {status} {path.parent.name:30s} -> {result.trace_id}")
            if result.validated:
                validated_count += 1
        total_rows = count_traces(conn)

    print(
        f"\nDone. {validated_count}/{len(paths)} validated against ATIF. "
        f"traces table now holds {total_rows} row(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
