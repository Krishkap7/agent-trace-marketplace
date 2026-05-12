"""SELECT and pretty-print the ``traces`` table.

Originally a slice-1 sanity check (raw_blob/atif populated). Now also
surfaces the ``has_error`` column added in slice 2.5 and prints a
source-format breakdown.

Run:
    uv run python scripts/verify.py
    uv run python scripts/verify.py --limit 50              # cap table rows
    uv run python scripts/verify.py --show-json <trace_id>  # dump ATIF for one row
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
    ("source_format", 11, "<"),
    ("agent_name", 11, "<"),
    ("model", 22, "<"),
    ("num_steps", 9, ">"),
    ("num_tool_calls", 14, ">"),
    ("has_error", 9, ">"),
    ("raw_blob_len", 12, ">"),
    ("atif_len", 9, ">"),
)


def _fmt_cell(value: object, width: int, align: str) -> str:
    text = "NULL" if value is None else str(value)
    return f"{text:{align}{width}}"


def _print_table(rows: list[dict], *, limit: int | None) -> None:
    header = " | ".join(_fmt_cell(name, w, a) for name, w, a in COLUMNS)
    sep = "-+-".join("-" * w for _, w, _ in COLUMNS)
    print(header)
    print(sep)
    visible = rows if limit is None else rows[:limit]
    for row in visible:
        print(" | ".join(_fmt_cell(row.get(name), w, a) for name, w, a in COLUMNS))
    if limit is not None and len(rows) > limit:
        print(f"... {len(rows) - limit} more row(s) hidden (use --limit to see more)")


def _print_health(rows: list[dict]) -> int:
    total = len(rows)
    if total == 0:
        print(
            "\nWARNING: traces table is empty. Did you run ingest_atif.py?",
            file=sys.stderr,
        )
        return 1

    raw_ok = sum(1 for r in rows if (r.get("raw_blob_len") or 0) > 0)
    atif_ok = sum(1 for r in rows if (r.get("atif_len") or 0) > 0)
    print(f"\n{total} row(s) total")
    print(f"  raw_blob populated: {raw_ok}/{total}")
    print(f"  atif populated:     {atif_ok}/{total}")

    fmt_counts: dict[str, int] = {}
    for r in rows:
        f = r.get("source_format") or "<null>"
        fmt_counts[f] = fmt_counts.get(f, 0) + 1
    print("\nsource_format breakdown:")
    for fmt, cnt in sorted(fmt_counts.items()):
        print(f"  {cnt:5d}  {fmt}")

    he_counts: dict[str, int] = {}
    for r in rows:
        key = "NULL" if r.get("has_error") is None else str(r["has_error"])
        he_counts[key] = he_counts.get(key, 0) + 1
    print("\nhas_error breakdown:")
    for key in sorted(he_counts, key=lambda k: (k == "NULL", k)):
        print(f"  {he_counts[key]:5d}  {key}")

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
        print(
            f"atif.steps[0].source: {atif['steps'][0].get('source') if atif.get('steps') else 'NO STEPS'}"
        )
        print(
            f"atif.steps[0].keys  : {sorted((atif['steps'][0] if atif.get('steps') else {}).keys())}"
        )
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
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max table rows to print (default: 20). Use 0 for all.",
    )
    args = parser.parse_args()
    table_limit = None if args.limit == 0 else args.limit

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

    _print_table(rows, limit=table_limit)
    return _print_health(rows)


if __name__ == "__main__":
    raise SystemExit(main())
