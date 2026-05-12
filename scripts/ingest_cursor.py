"""Ingest Cursor v1 export envelopes (``*.cursor.json``) into the ``traces`` table.

Walks ``data/raw/cursor/*.cursor.json`` (override with ``--raw-dir``)
and runs each file through
:func:`trace_marketplace.ingest.pipeline.ingest_file`. Same
``IngestResult`` reporting and idempotent upsert semantics as the
other ingest scripts.

The Cursor v1 envelope is documented in
``trace_marketplace.adapters.cursor``. Native ingest of Cursor's two
canonical shapes (workspace SQLite, ``cursor-agent --output-format
stream-json`` NDJSON) is a deliberate v2 deferral.

Run:
    uv run python scripts/ingest_cursor.py
    uv run python scripts/ingest_cursor.py --files path/to/session.cursor.json
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
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cursor"


def _resolve_paths(raw_dir: Path, files: list[Path] | None) -> list[Path]:
    if files:
        return [f for f in files if f.is_file()]
    if not raw_dir.is_dir():
        return []
    # The double-suffix ``*.cursor.json`` glob: matches what our
    # generator emits and what envelope authors are expected to use.
    # Plain ``.json`` files are also tried so detection (which falls
    # back to structural sniffing) can still work for ad-hoc uploads.
    cursor_files = list(raw_dir.glob("*.cursor.json"))
    plain_json = [
        p for p in raw_dir.glob("*.json") if not p.name.endswith(".cursor.json")
    ]
    return sorted(cursor_files + plain_json)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Directory of *.cursor.json (and *.json) files to ingest (default: {DEFAULT_RAW_DIR}).",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        metavar="FILE",
        help="Explicit list of files. Overrides --raw-dir.",
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
            f"ERROR: no *.cursor.json files found under {args.raw_dir} (and --files not given).",
            file=sys.stderr,
        )
        return 1

    print(f"Ingesting {len(paths)} Cursor file(s) into {args.db}")

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
