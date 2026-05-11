"""SELECT and pretty-print the ``traces`` table.

The whole point of slice 1 is to demonstrate end-to-end that ``raw_blob`` and
``atif`` are populated for real trajectories. This script prints a one-row
summary table plus a separator showing that both blob columns have non-zero
length for every row.

Run:
    uv run python scripts/verify.py
    uv run python scripts/verify.py --show-json <trace_id>   # dump the ATIF for one row
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from trace_marketplace.storage.db import (
    DEFAULT_DB_PATH,
    connect,
    fetch_all_summaries,
)

COLUMNS: tuple[tuple[str, int, str], ...] = (
    ("id", 38, "<"),
    ("source_format", 6, "<"),
    ("agent_name", 10, "<"),
    ("model", 18, "<"),
    ("num_steps", 9, ">"),
    ("num_tool_calls", 14, ">"),
    ("raw_blob_len", 12, ">"),
    ("atif_len", 9, ">"),
)


def _fmt_cell(value: object, width: int, align: str) -> str:
    text = "NULL" if value is None else str(value)
    return f"{text:{align}{width}}"


def _print_table(rows: list[dict]) -> None:
    header = " | ".join(_fmt_cell(name, w, a) for name, w, a in COLUMNS)
    sep = "-+-".join("-" * w for _, w, _ in COLUMNS)
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(_fmt_cell(row.get(name), w, a) for name, w, a in COLUMNS))


def _print_health(rows: list[dict]) -> int:
    total = len(rows)
    if total == 0:
        print("\nWARNING: traces table is empty. Did you run ingest_atif.py?", file=sys.stderr)
        return 1

    raw_ok = sum(1 for r in rows if (r.get("raw_blob_len") or 0) > 0)
    atif_ok = sum(1 for r in rows if (r.get("atif_len") or 0) > 0)
    print(f"\n{total} row(s) total")
    print(f"  raw_blob populated: {raw_ok}/{total}")
    print(f"  atif populated:     {atif_ok}/{total}")

    if raw_ok < total:
        print("ERROR: some rows have empty raw_blob.", file=sys.stderr)
        return 1
    if atif_ok < total:
        print(
            "NOTE: some rows have NULL atif (validation failed). "
            "raw_blob is still present so we can re-process later.",
            file=sys.stderr,
        )
    return 0


def _show_json(db_path: Path, trace_id: str) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            "SELECT id, source_format, raw_blob, atif FROM traces WHERE id = ?",
            (trace_id,),
        )
        row = cur.fetchone()
    if row is None:
        print(f"No row with id={trace_id!r} found in {db_path}.", file=sys.stderr)
        return 1
    print(f"id           : {row['id']}")
    print(f"source_format: {row['source_format']}")
    print(f"raw_blob.len : {len(row['raw_blob'])}")
    print(f"atif.len     : {len(row['atif']) if row['atif'] else 'NULL'}")
    if row["atif"]:
        atif = json.loads(row["atif"])
        print(f"atif.schema_version : {atif.get('schema_version')}")
        print(f"atif.session_id     : {atif.get('session_id')}")
        print(f"atif.agent          : {atif.get('agent')}")
        print(f"atif.steps[0].source: {atif['steps'][0].get('source') if atif.get('steps') else 'NO STEPS'}")
        print(f"atif.steps[0].keys  : {sorted((atif['steps'][0] if atif.get('steps') else {}).keys())}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite database path (default: {DEFAULT_DB_PATH}).",
    )
    parser.add_argument(
        "--show-json",
        metavar="TRACE_ID",
        help="Instead of the summary table, show the parsed ATIF metadata for one trace.",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(
            f"ERROR: database {args.db} does not exist. "
            "Run scripts/download_corpus.py and scripts/ingest_atif.py first.",
            file=sys.stderr,
        )
        return 1

    if args.show_json:
        return _show_json(args.db, args.show_json)

    try:
        with connect(args.db) as conn:
            rows = fetch_all_summaries(conn)
    except sqlite3.OperationalError as exc:
        print(f"ERROR: {exc}. The traces table may not exist yet.", file=sys.stderr)
        return 1

    _print_table(rows)
    return _print_health(rows)


if __name__ == "__main__":
    raise SystemExit(main())
