"""Pretty-print one trace from the ``traces`` table by id.

Useful for eyeballing what the adapters produced. Defaults to printing a
structural summary; pass ``--full`` to dump the entire ATIF JSON, ``--raw``
to dump the un-adapted raw blob.

Run:
    uv run python scripts/inspect_trace.py mt-fin2-accountant__4TAccTyw
    uv run python scripts/inspect_trace.py <claude_code_session_uuid> --full
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from trace_marketplace.storage.db import DEFAULT_DB_PATH, connect


def _print_summary(row: dict[str, Any], atif: dict[str, Any] | None) -> None:
    print(f"id            : {row['id']}")
    print(f"source_format : {row['source_format']}")
    print(f"ingested_at   : {row['ingested_at']}")
    print(f"agent_name    : {row['agent_name']}")
    print(f"model         : {row['model']}")
    print(f"num_steps     : {row['num_steps']}")
    print(f"num_tool_calls: {row['num_tool_calls']}")
    print(f"raw_blob len  : {len(row['raw_blob'])} bytes")
    print(f"atif len      : {len(row['atif']) if row['atif'] else 0} bytes")

    if atif is None:
        print("\n(atif column is NULL -- adapter could not validate)")
        return

    print("\n--- Agent ---")
    agent = atif.get("agent", {})
    print(f"  name        : {agent.get('name')}")
    print(f"  version     : {agent.get('version')}")
    print(f"  model_name  : {agent.get('model_name')}")

    steps = atif.get("steps", [])
    print(f"\n--- Steps ({len(steps)}) ---")
    src_counter = Counter(s.get("source") for s in steps)
    print(f"  by source        : {dict(src_counter)}")
    total_tool_calls = sum(len(s.get("tool_calls") or []) for s in steps)
    steps_with_obs = sum(1 for s in steps if s.get("observation"))
    print(f"  total tool_calls : {total_tool_calls}")
    print(f"  steps w/ obs     : {steps_with_obs}")
    if steps:
        first = steps[0]
        msg = first.get("message", "")
        preview = msg if isinstance(msg, str) else str(msg)
        if len(preview) > 120:
            preview = preview[:120] + "..."
        print(f"  step[1] preview  : source={first.get('source')!r} msg={preview!r}")

    if atif.get("extra"):
        print("\n--- Trajectory.extra ---")
        for k, v in atif["extra"].items():
            if isinstance(v, dict):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_id", help="Primary key of the row to inspect.")
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite database path (default: {DEFAULT_DB_PATH}).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Dump the full parsed ATIF JSON.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Dump the raw_blob bytes instead of the ATIF view.",
    )
    args = parser.parse_args()

    with connect(args.db) as conn:
        conn.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))
        cur = conn.execute("SELECT * FROM traces WHERE id = ?", (args.trace_id,))
        row = cur.fetchone()

    if row is None:
        print(f"ERROR: no trace with id={args.trace_id!r}", file=sys.stderr)
        return 1

    if args.raw:
        sys.stdout.write(row["raw_blob"])
        return 0

    atif = json.loads(row["atif"]) if row["atif"] else None

    if args.full:
        if atif is None:
            print("(atif is NULL)")
        else:
            print(json.dumps(atif, indent=2, ensure_ascii=False))
        return 0

    _print_summary(row, atif)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
