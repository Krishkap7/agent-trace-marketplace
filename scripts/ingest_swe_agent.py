"""Ingest SWE-agent HF rows (one JSON file per row) into the ``traces`` table.

Walks ``data/raw/swe_agent/*.json`` (override with ``--raw-dir``) and runs
each file through :func:`trace_marketplace.ingest.pipeline.ingest_file`.
Same ``IngestResult`` reporting and idempotent upsert semantics as
``scripts/ingest_atif.py`` and ``scripts/ingest_claude_code.py``.

Run:
    uv run python scripts/ingest_swe_agent.py
    uv run python scripts/ingest_swe_agent.py --files path/to/one.json
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
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "swe_agent"


def _resolve_paths(raw_dir: Path, files: list[Path] | None) -> list[Path]:
    if files:
        return [f for f in files if f.is_file()]
    if not raw_dir.is_dir():
        return []
    return sorted(raw_dir.glob("*.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Directory of *.json files to ingest (default: {DEFAULT_RAW_DIR}).",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        metavar="FILE",
        help="Explicit list of JSON files. Overrides --raw-dir.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite database path (default: {DEFAULT_DB_PATH}).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    paths = _resolve_paths(args.raw_dir, args.files)
    if not paths:
        print(
            f"ERROR: no .json files found under {args.raw_dir} (and --files not given).",
            file=sys.stderr,
        )
        return 1

    print(f"Ingesting {len(paths)} SWE-agent file(s) into {args.db}")

    validated_count = 0
    with connect(args.db) as conn:
        init_schema(conn)
        for path in paths:
            result = ingest_file(path, conn)
            status = "OK   " if result.validated else "RAW  "
            print(
                f"  {status} {path.name:60s} -> {result.trace_id}  [{result.source_format}]"
            )
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
