"""Download trajectory.json files from obaydata/mcp-agent-trajectory-benchmark.

Uses ``huggingface_hub.snapshot_download`` (not ``datasets.load_dataset``) so we
get the actual JSON files on disk, byte-for-byte. The eventual product takes
file uploads, and ``raw_blob`` is defined as the raw bytes the user uploaded,
so this mirrors the production ingest path exactly: file -> read bytes ->
store -> validate.

We only fetch ``*/trajectory.json`` (the ATIF single-pass trajectories); the
per-agent ``workspace/*.md`` files and the multi-conv directory are skipped.

Run:
    uv run python scripts/download_corpus.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = PROJECT_ROOT / "data" / "raw"

REPO_ID = "obaydata/mcp-agent-trajectory-benchmark"


def download(target: Path = DEFAULT_TARGET) -> Path:
    """Pull every ``*/trajectory.json`` under the dataset into ``target``."""
    target.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=["*/trajectory.json"],
        local_dir=str(target),
    )
    return Path(local)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"Local directory to download into (default: {DEFAULT_TARGET}).",
    )
    args = parser.parse_args()

    print(f"Downloading {REPO_ID} trajectory.json files into {args.target} ...")
    local = download(args.target)

    trajectories = sorted(local.glob("*/trajectory.json"))
    print(f"\nDone. Wrote {len(trajectories)} trajectory.json files to {local}")
    if not trajectories:
        print(
            "WARNING: no trajectory.json files found. Either the repo layout "
            "changed or the download silently failed.",
            file=sys.stderr,
        )
        return 1

    print("\nFirst few:")
    for path in trajectories[:5]:
        rel = path.relative_to(local)
        size = path.stat().st_size
        print(f"  {rel}  ({size:,} bytes)")
    if len(trajectories) > 5:
        print(f"  ... and {len(trajectories) - 5} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
