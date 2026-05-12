"""Step-0 exploration of nebius/SWE-agent-trajectories.

The streaming `datasets.load_dataset(streaming=True)` path gets badly
rate-limited on unauthenticated HF endpoints (see prior incident notes).
Instead we list the dataset's parquet shards and pull just one, then read
locally with pyarrow. Faster, more reliable, and gives us thousands of rows
to characterise instead of the dozens streaming would have yielded.

Goals:
  1. Confirm `trajectory` field type at the parquet level (string vs list).
  2. Enumerate distinct `model_name` values.
  3. Enumerate distinct `exit_status` values.
  4. Per-row size statistics.
  5. Skim one trajectory to confirm role/content shape.

Run:
    HF_HOME="$PWD/.cache/huggingface" uv run python scripts/explore_swe_agent.py
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

DATASET_ID = "nebius/SWE-agent-trajectories"
ROWS_TO_INSPECT = 200

log = logging.getLogger(__name__)


def _truncate(s: str, n: int = 600) -> str:
    return s if len(s) <= n else s[:n] + f"\n... [truncated, {len(s) - n} more chars]"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    api = HfApi()
    files = api.list_repo_files(DATASET_ID, repo_type="dataset")
    parquet_files = sorted(f for f in files if f.endswith(".parquet"))
    if not parquet_files:
        raise RuntimeError(f"No .parquet files in {DATASET_ID}: {files}")

    print(f"{DATASET_ID} has {len(parquet_files)} parquet shard(s)")
    print(f"  first  : {parquet_files[0]}")
    print(f"  last   : {parquet_files[-1]}")

    target = parquet_files[0]
    print(f"\nDownloading first shard ({target}) ...")
    local_path = hf_hub_download(
        DATASET_ID,
        filename=target,
        repo_type="dataset",
    )
    size_mb = Path(local_path).stat().st_size / (1024 * 1024)
    print(f"  fetched {local_path} ({size_mb:.1f} MB)")

    # ----- Schema / Arrow types -----
    pf = pq.ParquetFile(local_path)
    print("\nParquet metadata:")
    print(f"  total rows in this shard: {pf.metadata.num_rows:,}")
    print(f"  num row groups          : {pf.metadata.num_row_groups}")

    print("\nArrow schema (this is the SOURCE OF TRUTH for trajectory's type):")
    for field in pf.schema_arrow:
        print(f"  {field.name:20s} type={field.type}")

    # ----- Sample rows -----
    print(f"\nReading first {ROWS_TO_INSPECT} rows for content survey ...")
    table = pf.read_row_group(0).slice(0, ROWS_TO_INSPECT)
    rows = table.to_pylist()
    print(f"  got {len(rows)} rows")

    # ----- Cross-row stats -----
    model_counter: Counter[str] = Counter()
    exit_counter: Counter[str] = Counter()
    target_counter: Counter[bool] = Counter()
    trajectory_python_type: Counter[str] = Counter()

    sizes_total: list[int] = []
    sizes_traj: list[int] = []
    sizes_eval_logs: list[int] = []
    sizes_patch: list[int] = []
    traj_step_counts: list[int] = []
    traj_role_counts: Counter[str] = Counter()

    for r in rows:
        model_counter[r.get("model_name", "<missing>")] += 1
        exit_counter[r.get("exit_status", "<missing>")] += 1
        target_counter[bool(r.get("target"))] += 1

        traj = r.get("trajectory")
        trajectory_python_type[type(traj).__name__] += 1

        if isinstance(traj, str):
            sizes_traj.append(len(traj))
            try:
                parsed = json.loads(traj)
                if isinstance(parsed, list):
                    traj_step_counts.append(len(parsed))
                    for entry in parsed:
                        if isinstance(entry, dict):
                            traj_role_counts[entry.get("role", "<no role>")] += 1
            except json.JSONDecodeError:
                pass
        elif isinstance(traj, list):
            sizes_traj.append(len(json.dumps(traj, default=str)))
            traj_step_counts.append(len(traj))
            for entry in traj:
                if isinstance(entry, dict):
                    traj_role_counts[entry.get("role", "<no role>")] += 1

        eval_logs = r.get("eval_logs")
        if isinstance(eval_logs, str):
            sizes_eval_logs.append(len(eval_logs))
        patch = r.get("generated_patch")
        if isinstance(patch, str):
            sizes_patch.append(len(patch))

        sizes_total.append(len(json.dumps(r, default=str)))

    print(f"\n=== Distinct values across {len(rows)} rows ===")
    print(f"\nmodel_name (n={len(model_counter)}):")
    for k, c in model_counter.most_common():
        print(f"  {c:4d}  {k}")
    print(f"\nexit_status (n={len(exit_counter)}):")
    for k, c in exit_counter.most_common():
        print(f"  {c:4d}  {k}")
    print(f"\ntarget distribution: {dict(target_counter)}")

    print("\n=== Trajectory shape ===")
    print(
        f"Python type of row['trajectory'] across rows: {dict(trajectory_python_type)}"
    )
    if traj_step_counts:
        print(
            f"Trajectory step count stats: min={min(traj_step_counts)}, "
            f"max={max(traj_step_counts)}, "
            f"mean={sum(traj_step_counts) // len(traj_step_counts)}"
        )
    print(f"Roles seen across all trajectory entries: {dict(traj_role_counts)}")

    def _stats(label: str, xs: list[int]) -> None:
        if not xs:
            print(f"  {label:20s} (no samples)")
            return
        print(
            f"  {label:20s} min={min(xs):>10,}  "
            f"max={max(xs):>12,}  "
            f"mean={sum(xs) // len(xs):>10,}  "
            f"p95={sorted(xs)[int(len(xs) * 0.95)]:>10,}"
        )

    print("\n=== Field sizes (bytes) ===")
    _stats("row JSON total", sizes_total)
    _stats("trajectory string", sizes_traj)
    _stats("generated_patch", sizes_patch)
    _stats("eval_logs", sizes_eval_logs)

    # ----- Spot-check one trajectory -----
    print("\n=== Spot-check first row (truncated) ===")
    first = rows[0]
    print(f"instance_id : {first.get('instance_id')}")
    print(f"model_name  : {first.get('model_name')}")
    print(f"target      : {first.get('target')}")
    print(f"exit_status : {first.get('exit_status')}")

    traj = first.get("trajectory")
    if isinstance(traj, str):
        try:
            parsed = json.loads(traj)
        except json.JSONDecodeError:
            parsed = None
    else:
        parsed = traj

    if isinstance(parsed, list) and parsed:
        print(f"\ntrajectory has {len(parsed)} entries")
        print(
            f"trajectory[0] keys: {sorted(parsed[0].keys()) if isinstance(parsed[0], dict) else type(parsed[0]).__name__}"
        )
        if isinstance(parsed[0], dict):
            print(f"trajectory[0].role    : {parsed[0].get('role')!r}")
            content_0 = parsed[0].get("content", "")
            print(
                f"trajectory[0].content (truncated):\n{_truncate(str(content_0), 400)}"
            )
        for idx in (1, 2, 3):
            if idx < len(parsed) and isinstance(parsed[idx], dict):
                role = parsed[idx].get("role")
                content_preview = str(parsed[idx].get("content", ""))[:120]
                print(
                    f"trajectory[{idx}].role={role!r}  content[:120]={content_preview!r}"
                )


if __name__ == "__main__":
    main()
