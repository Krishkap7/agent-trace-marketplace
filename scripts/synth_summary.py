"""Pretty-print aggregated stats from a synth run manifest.

Reads ``data/raw/synth_manifest.json`` (or any path passed via ``--path``)
and prints a quick-look report: counts by format, counts by primary
failure mode, mean / median step + tool-call counts, retry rate, drop
rate, top-N most expensive and longest specs.

Why a dedicated script rather than inline into ``generate_synthetic_traces.py``?

* The generator prints a single-line summary on completion. That's
  enough to decide "did the run succeed?" -- but not enough to spot
  shape drift (e.g. "the average loop trace got 40% longer this run
  vs. the smoke run").
* The same manifest is the artefact we keep in the PR; reviewers
  can run this on the committed JSON without re-running the corpus.
* Read-only, pure-Python -- no LLM, no DB, free.

Usage:
    uv run python scripts/synth_summary.py
    uv run python scripts/synth_summary.py --path /path/to/manifest.json
    uv run python scripts/synth_summary.py --top 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path(__file__).parent.parent / "data" / "raw" / "synth_manifest.json"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Manifest JSON path (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="How many rows in the top-cost / top-length tables (default: 5).",
    )
    return parser.parse_args(argv)


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Manifest not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate_by(field_path: list[str], rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count rows by nested field path.

    ``field_path=["spec", "primary_failure"]`` walks each row's
    ``spec.primary_failure`` and tallies into a label -> count dict.
    Missing fields are tallied as ``"<missing>"`` so a malformed row
    is visible rather than silently dropped.
    """
    counts: dict[str, int] = {}
    for row in rows:
        cur: Any = row
        for key in field_path:
            if not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(key)
        key_label = str(cur) if cur is not None else "<missing>"
        counts[key_label] = counts.get(key_label, 0) + 1
    return counts


def _summary_for_label(
    rows: list[dict[str, Any]], label: str
) -> tuple[int, float, float, float, float]:
    """Return ``(n, mean_steps, median_steps, mean_tcs, median_tcs)`` for one label.

    Only successful (``ok=True``) rows contribute -- the atif_summary
    is None on failed rows.
    """
    steps: list[int] = []
    tcs: list[int] = []
    for row in rows:
        if not row.get("ok"):
            continue
        spec = row.get("spec") or {}
        if spec.get("primary_failure") != label:
            continue
        atif = row.get("atif_summary") or {}
        n_steps = atif.get("n_steps")
        n_tcs = atif.get("n_tool_calls")
        if isinstance(n_steps, int):
            steps.append(n_steps)
        if isinstance(n_tcs, int):
            tcs.append(n_tcs)
    if not steps:
        return (0, 0.0, 0.0, 0.0, 0.0)
    return (
        len(steps),
        statistics.mean(steps),
        statistics.median(steps),
        statistics.mean(tcs) if tcs else 0.0,
        statistics.median(tcs) if tcs else 0.0,
    )


def _print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _print_counts(title: str, counts: dict[str, int]) -> None:
    _print_section(title)
    for key, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {key:<24} {count}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    manifest = _load_manifest(args.path)

    summary = manifest.get("summary") or {}
    rows: list[dict[str, Any]] = manifest.get("results") or []
    if not rows:
        print(f"No result rows in {args.path}.")
        return 1

    print("=" * 60)
    print(f"SYNTH RUN SUMMARY -- {args.path}")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key:<22} {value}")

    _print_counts("By format", _aggregate_by(["spec", "output_format"], rows))
    _print_counts(
        "By primary failure mode",
        _aggregate_by(["spec", "primary_failure"], rows),
    )

    # Retry / drop rate
    n_total = len(rows)
    n_ok = sum(1 for r in rows if r.get("ok"))
    n_failed = sum(1 for r in rows if not r.get("ok") and not r.get("budget_skipped"))
    n_budget_skipped = sum(1 for r in rows if r.get("budget_skipped"))
    n_retries = sum(1 for r in rows if r.get("attempts", 1) > 1)
    _print_section("Outcomes")
    print(f"  total                  {n_total}")
    print(f"  ok                     {n_ok} ({n_ok / n_total:.1%})")
    print(f"  failed (validation)    {n_failed} ({n_failed / n_total:.1%})")
    print(f"  budget_skipped         {n_budget_skipped}")
    print(f"  needed 2nd attempt     {n_retries} ({n_retries / n_total:.1%})")

    # Per-label step + tool-call distribution
    labels = sorted({r["spec"]["primary_failure"] for r in rows if r.get("spec")})
    _print_section("Per-label trace shape (successful rows only)")
    print(
        f"  {'label':<22} {'n':>4}  {'mean steps':>10}  "
        f"{'median':>6}  {'mean tcs':>9}  {'median':>6}"
    )
    for label in labels:
        n, mean_s, med_s, mean_t, med_t = _summary_for_label(rows, label)
        if n == 0:
            continue
        print(
            f"  {label:<22} {n:>4}  {mean_s:>10.1f}  {med_s:>6.0f}  "
            f"{mean_t:>9.1f}  {med_t:>6.0f}"
        )

    # Top-N most expensive
    by_cost = sorted(rows, key=lambda r: r.get("cost_usd") or 0.0, reverse=True)
    _print_section(f"Top {args.top} most expensive specs")
    for row in by_cost[: args.top]:
        spec = row.get("spec") or {}
        print(
            f"  ${row.get('cost_usd', 0):.4f}  "
            f"{spec.get('output_format'):<12}  "
            f"{spec.get('primary_failure'):<22}  "
            f"attempts={row.get('attempts', '?')}"
        )

    # Top-N longest
    def _step_count(row: dict[str, Any]) -> int:
        atif = row.get("atif_summary") or {}
        return int(atif.get("n_steps") or 0)

    by_length = sorted(rows, key=_step_count, reverse=True)
    _print_section(f"Top {args.top} longest specs (by step count)")
    for row in by_length[: args.top]:
        spec = row.get("spec") or {}
        atif = row.get("atif_summary") or {}
        print(
            f"  {atif.get('n_steps', '?'):>3} steps  "
            f"{spec.get('output_format'):<12}  "
            f"{spec.get('primary_failure'):<22}  "
            f"${row.get('cost_usd', 0):.4f}"
        )

    # Failure errors -- when validation rejected a spec twice.
    failures = [r for r in rows if not r.get("ok") and not r.get("budget_skipped")]
    if failures:
        _print_section(f"All {len(failures)} validation failures")
        for row in failures:
            spec = row.get("spec") or {}
            print(
                f"  {spec.get('output_format'):<12}  "
                f"{spec.get('primary_failure'):<22}  "
                f"err={row.get('error', '')[:80]}"
            )

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
