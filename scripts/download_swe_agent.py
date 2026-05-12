"""Download a stratified sample of nebius/SWE-agent-trajectories rows.

Plan-deviation note (relative to slice 2.5 plan): the plan called for
`datasets.load_dataset(..., streaming=True)`. Step-0 inspection confirmed
that path gets badly rate-limited / appears to hang on unauthenticated HF.
This script instead pulls the dataset's parquet shards directly via
``huggingface_hub.hf_hub_download`` and reads them locally with pyarrow --
the same approach we used for Step-0 and slice 1's ATIF download. Faster,
deterministic, and avoids a heavy dep (no `datasets` install needed).

Stratification: We bucket on ``target`` only (success vs failure). The
prompt and dataset card both claimed three model_names; Step-0 saw only two
in shard 0. To stay robust to that uncertainty, model balance is incidental
-- the script logs the mix it ends up with so it's visible in the PR.

Each row is written as ``data/raw/swe_agent/{instance_id}.json`` -- one row
per file, one file per trace. The JSON keeps the row exactly as pyarrow
yielded it (trajectory as a real list, no JSON-string wrapping), which is
what the SWEAgentAdapter expects.

Usage:
    HF_HOME="$PWD/.cache/huggingface" uv run python scripts/download_swe_agent.py
    HF_HOME="$PWD/.cache/huggingface" uv run python scripts/download_swe_agent.py --limit 10
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

DATASET_ID = "nebius/SWE-agent-trajectories"
DEFAULT_OUT_DIR = Path("data/raw/swe_agent")

# 250 success + 250 failure = 500 total. Single 6.7k-row shard easily
# yields enough of each.
TARGET_PER_BUCKET = 250

log = logging.getLogger(__name__)


def _iter_dataset_rows(dataset_id: str) -> Iterator[dict[str, Any]]:
    """Yield rows from every parquet shard, one shard at a time.

    Stops as soon as the consumer breaks the loop, so we never download more
    shards than we need. The first shard alone (~85 MB) holds 6,670 rows --
    enough for our 500-row sample with plenty of headroom.
    """
    api = HfApi()
    files = api.list_repo_files(dataset_id, repo_type="dataset")
    parquet_files = sorted(f for f in files if f.endswith(".parquet"))
    log.info("Dataset %s has %d parquet shard(s)", dataset_id, len(parquet_files))

    for shard_idx, shard_name in enumerate(parquet_files):
        log.info(
            "Downloading shard %d/%d: %s", shard_idx + 1, len(parquet_files), shard_name
        )
        local_path = hf_hub_download(
            dataset_id,
            filename=shard_name,
            repo_type="dataset",
        )
        pf = pq.ParquetFile(local_path)
        for rg_idx in range(pf.num_row_groups):
            for row in pf.read_row_group(rg_idx).to_pylist():
                yield row


def _safe_filename(instance_id: str) -> str:
    """Make instance_id filesystem-safe (e.g. swap '/' for '__')."""
    return instance_id.replace("/", "__").replace(" ", "_")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory to write JSON files into (default: %(default)s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Override per-bucket target. e.g. --limit 5 yields ~5 success "
            "+ ~5 failure rows. Used during Phase C smoke-test."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_bucket = args.limit if args.limit is not None else TARGET_PER_BUCKET
    log.info("Targeting %d rows per bucket (success / failure).", per_bucket)

    success_written = 0
    failure_written = 0
    seen_ids: set[str] = set()
    model_breakdown: Counter[str] = Counter()
    skipped_reasons: Counter[str] = Counter()

    for row in _iter_dataset_rows(DATASET_ID):
        if success_written >= per_bucket and failure_written >= per_bucket:
            break

        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            skipped_reasons["missing_instance_id"] += 1
            continue
        if instance_id in seen_ids:
            skipped_reasons["duplicate_instance_id"] += 1
            continue

        target = row.get("target")
        if not isinstance(target, bool):
            skipped_reasons["non_bool_target"] += 1
            continue

        if target and success_written >= per_bucket:
            continue
        if not target and failure_written >= per_bucket:
            continue

        out_path = args.out_dir / f"{_safe_filename(instance_id)}.json"
        out_path.write_text(
            json.dumps(row, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        seen_ids.add(instance_id)
        model = row.get("model_name") or "<unknown>"
        model_breakdown[(model, "success" if target else "failure")] += 1
        if target:
            success_written += 1
        else:
            failure_written += 1

        if (success_written + failure_written) % 50 == 0:
            log.info(
                "Progress: success=%d/%d  failure=%d/%d",
                success_written,
                per_bucket,
                failure_written,
                per_bucket,
            )

    print(f"\nDone. Wrote {success_written + failure_written} files to {args.out_dir}")
    print(f"  success (target=True) : {success_written}")
    print(f"  failure (target=False): {failure_written}")
    if skipped_reasons:
        print("\nSkipped during scan:")
        for reason, count in skipped_reasons.most_common():
            print(f"  {count:5d}  {reason}")
    print("\nModel x outcome breakdown:")
    for (model, outcome), count in sorted(model_breakdown.items()):
        print(f"  {count:5d}  {model:30s}  {outcome}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
