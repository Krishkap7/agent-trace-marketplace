"""Show failure / success distribution across the traces table.

This is the ground-truth view that slice 3 (failure detection) will
calibrate against. We split by:

  * source_format     -- which ingest path the row came in through
  * has_error         -- 0 = succeeded, 1 = failed, NULL = unlabelled
  * model             -- per-model breakdown (within source_format)

Run:
    uv run python scripts/breakdown.py
    uv run python scripts/breakdown.py --db data/marketplace.db
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from trace_marketplace.storage.db import DEFAULT_DB_PATH, connect


def _print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite database path (default: {DEFAULT_DB_PATH}).",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"ERROR: database {args.db} does not exist.", file=sys.stderr)
        return 1

    with connect(args.db) as conn:
        rows = conn.execute(
            """
            SELECT source_format, model, has_error
            FROM traces
            """
        ).fetchall()

    if not rows:
        print("traces table is empty.")
        return 1

    total = len(rows)
    print(f"Total rows: {total}")

    fmt_counter: Counter[str] = Counter()
    he_by_fmt: dict[str, Counter[str]] = {}
    model_by_fmt: dict[str, Counter[tuple[str, str]]] = {}

    for row in rows:
        fmt = row["source_format"] or "<null>"
        model = row["model"] or "<null>"
        he_key = "NULL" if row["has_error"] is None else str(row["has_error"])

        fmt_counter[fmt] += 1
        he_by_fmt.setdefault(fmt, Counter())[he_key] += 1
        model_by_fmt.setdefault(fmt, Counter())[(model, he_key)] += 1

    _print_section("Rows per source_format")
    for fmt, count in sorted(fmt_counter.items()):
        print(f"  {count:5d}  {fmt}")

    _print_section("has_error per source_format")
    print(
        f"  {'source_format':12s}  {'has_error=0':>11s}  {'has_error=1':>11s}  {'NULL':>6s}"
    )
    for fmt in sorted(he_by_fmt.keys()):
        c = he_by_fmt[fmt]
        zero = c.get("0", 0)
        one = c.get("1", 0)
        null = c.get("NULL", 0)
        print(f"  {fmt:12s}  {zero:>11d}  {one:>11d}  {null:>6d}")

    _print_section("Model x outcome (per source_format)")
    for fmt in sorted(model_by_fmt.keys()):
        sub = model_by_fmt[fmt]
        if not sub:
            continue
        print(f"\n  [{fmt}]")
        # Group by model first
        per_model: dict[str, dict[str, int]] = {}
        for (model, he_key), cnt in sub.items():
            per_model.setdefault(model, {"0": 0, "1": 0, "NULL": 0})[he_key] += cnt
        for model in sorted(per_model.keys()):
            c = per_model[model]
            total_for_model = sum(c.values())
            success_rate = (
                f"{c['0'] / (c['0'] + c['1']) * 100:.1f}%"
                if (c["0"] + c["1"]) > 0
                else "n/a"
            )
            print(
                f"    {model:30s}  total={total_for_model:4d}  "
                f"success={c['0']:3d}  failure={c['1']:3d}  null={c['NULL']:3d}  "
                f"success_rate={success_rate}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
