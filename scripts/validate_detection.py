"""Validate the rule-based detector against SWE-agent ground truth.

SWE-agent is the only corpus in the marketplace that ships with binary
success/failure labels (the dataset's ``target: bool`` field, propagated
to the ``has_error`` column at ingest time). This script re-runs every
signal against those traces and computes precision / recall / F1, both
per-signal and for the aggregate "any signal fired" predicate.

The output is printed for the agent loop AND written to
[data/validation_report.md](data/validation_report.md) so the design
doc can pull a table without re-running the script.

Usage:
    uv run python scripts/validate_detection.py [--db PATH] [--report PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

from trace_marketplace.enrich.failure_rules import any_signal_fired, compute_signals
from trace_marketplace.storage.db import DEFAULT_DB_PATH, connect

log = logging.getLogger("validate_detection")

DEFAULT_REPORT_PATH: Path = Path("data/validation_report.md")
SIGNAL_ORDER: tuple[str, ...] = (
    "hit_max_turns",
    "repeated_tool_call_loop",
    "ended_on_error",
    "abandonment",
    "incomplete_session",
)


@dataclass(frozen=True)
class Counts:
    """Confusion-matrix counts for one signal (or the aggregate).

    Treating the signal firing as the positive prediction and
    ``has_error = 1`` as the positive label.
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, fired: bool, is_failure: bool) -> "Counts":
        return Counts(
            tp=self.tp + int(fired and is_failure),
            fp=self.fp + int(fired and not is_failure),
            fn=self.fn + int(not fired and is_failure),
            tn=self.tn + int(not fired and not is_failure),
        )

    def precision(self) -> float | None:
        denom = self.tp + self.fp
        return self.tp / denom if denom else None

    def recall(self) -> float | None:
        denom = self.tp + self.fn
        return self.tp / denom if denom else None

    def f1(self) -> float | None:
        p, r = self.precision(), self.recall()
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def _walk_swe_agent(
    conn: sqlite3.Connection,
) -> tuple[dict[str, Counts], Counts, int, int]:
    """Re-run signals on every SWE-agent trace; tally per-signal +
    aggregate confusion matrices.

    Re-running (rather than reading the persisted ``failure_signals``
    column) is intentional: the column was written by an older code
    path on a previous run, and we want validation to reflect the
    code as it stands right now. Otherwise threshold tweaks wouldn't
    show up until ``detect_failures.py --force`` was re-run.
    """
    per_signal: dict[str, Counts] = {name: Counts() for name in SIGNAL_ORDER}
    aggregate = Counts()
    n_fail = 0
    n_pass = 0
    cur = conn.execute(
        "SELECT id, atif, has_error FROM traces "
        "WHERE source_format = 'swe_agent' AND atif IS NOT NULL"
    )
    for row in cur:
        truth = row["has_error"]
        if truth not in (0, 1):
            # Skip rows without ground truth (shouldn't exist for
            # SWE-agent, but defensive).
            continue
        is_failure = bool(truth)
        if is_failure:
            n_fail += 1
        else:
            n_pass += 1
        try:
            traj = json.loads(row["atif"])
        except json.JSONDecodeError:
            log.warning("Trace %s has malformed ATIF; skipping", row["id"])
            continue
        signals = compute_signals(traj)
        for name in SIGNAL_ORDER:
            per_signal[name] = per_signal[name].add(signals[name], is_failure)
        aggregate = aggregate.add(any_signal_fired(signals), is_failure)
    return per_signal, aggregate, n_fail, n_pass


def _format_report(
    per_signal: dict[str, Counts],
    aggregate: Counts,
    n_fail: int,
    n_pass: int,
) -> str:
    lines: list[str] = []
    total = n_fail + n_pass
    lines.append("# Failure-detection validation report")
    lines.append("")
    lines.append(
        f"Computed against {total} SWE-agent traces "
        f"({n_fail} failures / {n_pass} successes by `target`)."
    )
    lines.append("")
    lines.append(
        "Each row is the confusion matrix produced when the named signal "
        "(or `any_signal` for the aggregate) is treated as the positive "
        "prediction and `has_error = 1` as the positive label."
    )
    lines.append("")
    lines.append("| Signal | TP | FP | FN | TN | Precision | Recall | F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    def _row(name: str, c: Counts) -> str:
        return (
            f"| `{name}` | {c.tp} | {c.fp} | {c.fn} | {c.tn} | "
            f"{_fmt(c.precision())} | {_fmt(c.recall())} | {_fmt(c.f1())} |"
        )

    for name in SIGNAL_ORDER:
        lines.append(_row(name, per_signal[name]))
    lines.append(_row("any_signal", aggregate))
    lines.append("")
    lines.append(
        "**Reading the table.** High precision + low recall means the rule "
        "is correct when it fires but misses many failures (the LLM judge "
        "picks those up in the second pass). High recall + low precision "
        "means the rule over-triggers and we'd need to tighten it. The "
        "aggregate `any_signal` row is what gates whether a trace is sent "
        "to the LLM judge: anything firing here gets a label."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if not args.db.exists():
        log.error("Database not found at %s", args.db)
        return 1

    with connect(args.db) as conn:
        per_signal, aggregate, n_fail, n_pass = _walk_swe_agent(conn)

    if (n_fail + n_pass) == 0:
        print("No SWE-agent traces with ground-truth has_error found.")
        return 1

    print(
        f"Validation on {n_fail + n_pass} SWE-agent traces "
        f"({n_fail} failures, {n_pass} successes)"
    )
    header = f"{'Signal':<28} {'Precision':>10} {'Recall':>10} {'F1':>8}"
    print(header)
    print("-" * len(header))
    for name in SIGNAL_ORDER:
        c = per_signal[name]
        print(
            f"{name:<28} {_fmt(c.precision()):>10} {_fmt(c.recall()):>10} "
            f"{_fmt(c.f1()):>8}"
        )
    print(
        f"{'any_signal (aggregate)':<28} {_fmt(aggregate.precision()):>10} "
        f"{_fmt(aggregate.recall()):>10} {_fmt(aggregate.f1()):>8}"
    )

    report = _format_report(per_signal, aggregate, n_fail, n_pass)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print()
    print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
