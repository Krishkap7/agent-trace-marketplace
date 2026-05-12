"""Measure NL search precision@K against the synthetic-corpus ground truth.

Each synthetic trace has a ``synthetic_failure_mode`` tag in
``atif.extra`` (currently propagated for Cursor envelopes; Codex /
Claude Code propagation is a follow-up adapter tweak). This script
runs a handful of natural-language queries through the production
search path and reports what fraction of the top-K results actually
carry the expected label.

Cost: one Anthropic NL-parse call + one OpenAI intent-embed call per
query. With the default 8-query set that's ~$0.08 total. Use
``--queries-file`` to load a custom set.

Usage:
    uv run python scripts/eval_search.py                       # default 8 queries
    uv run python scripts/eval_search.py --top-k 20            # wider window
    uv run python scripts/eval_search.py --queries-file qs.json
    uv run python scripts/eval_search.py --include-judge       # also score against judge label
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from trace_marketplace.enrich.embeddings import require_api_key as _require_openai_key
from trace_marketplace.search.nl_query import require_api_key as _require_anthropic_key
from trace_marketplace.search.runner import run_nl_search
from trace_marketplace.storage.db import DEFAULT_DB_PATH, connect, init_schema

log = logging.getLogger("eval_search")


# ---------------------------------------------------------------------------
# Default eval set.
# ---------------------------------------------------------------------------
#
# Each entry is a tuple of ``(natural-language query, expected label
# set)``. We use a set on the right so a query like "find loops or
# tool spamming" can match either. The query strings deliberately
# avoid mentioning the enum value verbatim -- if Sonnet's NL parser
# is doing real work it should map "agent kept retrying" to
# ``loop``, etc.
DEFAULT_QUERIES: list[tuple[str, set[str]]] = [
    ("agents that kept retrying the same tool call", {"loop"}),
    ("traces where the agent gave up on the task", {"gave_up"}),
    ("agents that hallucinated a function or API", {"hallucinated_api"}),
    ("agents that misread an error message", {"misread_error"}),
    ("traces where the environment or tooling failed", {"environment_issue"}),
    ("agents fixated on the wrong solution approach", {"wrong_approach"}),
    ("traces with partial success", {"partial_success"}),
    ("successful agent runs", {"success"}),
]


# ---------------------------------------------------------------------------
# CLI parsing.
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many ranked results per query to score (default: 10).",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=None,
        help=(
            "JSON file: list of {\"query\": str, \"expected\": [labels]}. "
            "Overrides the default 8-query eval set."
        ),
    )
    parser.add_argument(
        "--include-judge",
        action="store_true",
        help=(
            "Also score against the LLM-judge ``failure_label`` column "
            "as a secondary ground-truth source. Useful for cross-"
            "validating the synthetic tag against the judge."
        ),
    )
    return parser.parse_args(argv)


def _load_queries(path: Path) -> list[tuple[str, set[str]]]:
    """Load a custom eval set from JSON, validating shape."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit(f"{path}: top-level must be a JSON array")
    out: list[tuple[str, set[str]]] = []
    for entry in raw:
        if not isinstance(entry, dict) or "query" not in entry or "expected" not in entry:
            raise SystemExit(
                f"{path}: each entry needs ``query`` and ``expected`` keys"
            )
        query = entry["query"]
        expected = entry["expected"]
        if not isinstance(query, str) or not isinstance(expected, list):
            raise SystemExit(
                f"{path}: ``query`` must be str, ``expected`` must be list"
            )
        out.append((query, set(map(str, expected))))
    return out


# ---------------------------------------------------------------------------
# DB helpers.
# ---------------------------------------------------------------------------


def _ground_truth_for(
    conn, trace_ids: list[str]
) -> dict[str, tuple[str | None, str | None]]:
    """Return ``trace_id -> (synthetic_failure_mode, failure_label)``.

    Uses ``json_extract`` so we don't have to load + parse the full
    ATIF blob per row. ``synthetic_failure_mode`` is the
    SOURCE-baked ground truth (currently only present on Cursor
    synth rows; Codex / Claude Code propagation TODO).
    ``failure_label`` is the LLM-judge's classification (when
    ``scripts/judge_failures.py`` has been run).
    """
    if not trace_ids:
        return {}
    placeholders = ",".join("?" * len(trace_ids))
    sql = (
        "SELECT id, "
        "       json_extract(atif, '$.extra.synthetic_failure_mode') AS gt, "
        "       failure_label "
        "FROM traces WHERE id IN (" + placeholders + ")"
    )
    out: dict[str, tuple[str | None, str | None]] = {}
    for row in conn.execute(sql, trace_ids):
        out[row["id"]] = (row["gt"], row["failure_label"])
    return out


# ---------------------------------------------------------------------------
# Per-query scoring.
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    query: str
    expected: set[str]
    hits: int
    relevant_synth: int
    relevant_judge: int
    no_ground_truth: int
    parsed_explanation: str
    parsed_filters: dict


def _eval_one(
    conn,
    query: str,
    expected: set[str],
    *,
    top_k: int,
) -> QueryResult:
    """Run one NL query and tally precision@K.

    ``relevant_synth`` counts hits whose synthetic ground-truth tag is
    in ``expected``. ``no_ground_truth`` counts hits we couldn't
    score (no ``synthetic_failure_mode`` -- typically real / SWE-
    agent / non-Cursor-synth rows). Both are reported so the user
    can see how much of the top-K window was un-scoreable.
    """
    result = run_nl_search(conn, query=query, top_k=top_k)
    if result.error:
        return QueryResult(
            query=query,
            expected=expected,
            hits=0,
            relevant_synth=0,
            relevant_judge=0,
            no_ground_truth=0,
            parsed_explanation=f"ERROR: {result.error}",
            parsed_filters={},
        )
    trace_ids = [h.trace_id for h in result.hits]
    gt = _ground_truth_for(conn, trace_ids)
    rel_synth = 0
    rel_judge = 0
    no_gt = 0
    for tid in trace_ids:
        synth, judge = gt.get(tid, (None, None))
        if synth is None:
            no_gt += 1
        elif synth in expected:
            rel_synth += 1
        if judge is not None and judge in expected:
            rel_judge += 1
    parsed_filters = {}
    parsed_explanation = ""
    if result.parsed is not None:
        parsed_explanation = result.parsed.explanation
        parsed_filters = {
            "failure_labels": list(result.parsed.failure_labels),
            "source_formats": list(result.parsed.source_formats),
            "has_error": result.parsed.has_error,
        }
    return QueryResult(
        query=query,
        expected=expected,
        hits=len(trace_ids),
        relevant_synth=rel_synth,
        relevant_judge=rel_judge,
        no_ground_truth=no_gt,
        parsed_explanation=parsed_explanation,
        parsed_filters=parsed_filters,
    )


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def _print_report(results: list[QueryResult], *, top_k: int, include_judge: bool) -> None:
    print("=" * 72)
    print(f"NL search precision@{top_k} eval")
    print("=" * 72)

    sum_rel_synth = 0
    sum_rel_judge = 0
    sum_hits = 0
    sum_scoreable = 0
    for r in results:
        scoreable = r.hits - r.no_ground_truth
        precision_synth = r.relevant_synth / scoreable if scoreable else 0.0
        line = (
            f"\n[{r.query}]\n"
            f"  expected:        {sorted(r.expected)}\n"
            f"  parser said:     {r.parsed_explanation}\n"
            f"  parser filters:  {r.parsed_filters}\n"
            f"  hits:            {r.hits} "
            f"({scoreable} scoreable + {r.no_ground_truth} no-ground-truth)\n"
            f"  precision@K (synth ground truth):  "
            f"{r.relevant_synth}/{scoreable} = {precision_synth:.1%}"
        )
        if include_judge:
            precision_judge = r.relevant_judge / r.hits if r.hits else 0.0
            line += (
                f"\n  precision@K (judge label):         "
                f"{r.relevant_judge}/{r.hits} = {precision_judge:.1%}"
            )
        print(line)

        sum_rel_synth += r.relevant_synth
        sum_rel_judge += r.relevant_judge
        sum_hits += r.hits
        sum_scoreable += scoreable

    print()
    print("-" * 72)
    overall_synth = sum_rel_synth / sum_scoreable if sum_scoreable else 0.0
    print(f"AGGREGATE precision@{top_k} (synth ground truth):  "
          f"{sum_rel_synth}/{sum_scoreable} = {overall_synth:.1%}")
    if include_judge:
        overall_judge = sum_rel_judge / sum_hits if sum_hits else 0.0
        print(f"AGGREGATE precision@{top_k} (judge label):         "
              f"{sum_rel_judge}/{sum_hits} = {overall_judge:.1%}")
    print("=" * 72)
    print()
    print(
        "Note: 'no-ground-truth' rows are traces without a baked-in "
        "synthetic_failure_mode (real-user uploads, SWE-agent rows, or "
        "Codex/Claude-Code synth rows whose adapters don't yet "
        "propagate the field). They are excluded from the synth-ground-"
        "truth denominator."
    )


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    _require_anthropic_key()
    _require_openai_key()

    queries = _load_queries(args.queries_file) if args.queries_file else DEFAULT_QUERIES

    if not args.db.exists():
        print(f"Database not found at {args.db}", file=sys.stderr)
        return 1

    with connect(args.db) as conn:
        init_schema(conn)
        # Verify the corpus has any ground-truth rows at all; bail
        # early with a clear message if not.
        count_with_gt = conn.execute(
            "SELECT COUNT(*) FROM traces WHERE "
            "json_extract(atif, '$.extra.synthetic_failure_mode') IS NOT NULL"
        ).fetchone()[0]
        if count_with_gt == 0:
            print(
                "No traces in this DB carry a synthetic_failure_mode "
                "ground-truth tag. Did you ingest the synth corpus?",
                file=sys.stderr,
            )
            return 1
        print(f"Corpus: {count_with_gt} traces with synthetic_failure_mode.")
        print(f"Running {len(queries)} queries at top-K={args.top_k}...")
        print()

        results: list[QueryResult] = []
        for query, expected in queries:
            r = _eval_one(conn, query, expected, top_k=args.top_k)
            results.append(r)

        _print_report(results, top_k=args.top_k, include_judge=args.include_judge)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
