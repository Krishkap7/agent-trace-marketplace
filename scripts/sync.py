"""One-shot catch-up for the auto-upload daemon.

Walks ``~/.claude/projects/`` and ``~/.codex/sessions/`` once, copies
each ``.jsonl`` into ``data/raw/<format>/`` if it's not already there
(filename + size + mtime match), and pipes new files through the same
:func:`scripts.auto_upload.process` chain (``ingest_file`` -> rule pass
-> embedding).

Useful when the daemon wasn't running for a while (closed laptop,
crashed shell, fresh clone). The dedupe is best-effort: matching
filename + size + mtime is enough to skip a file we've already pulled
in, but a file that was edited in place since the last sync will be
re-ingested (which is what we want, since ``ingest_file`` is
idempotent on ``trace_id`` anyway).

Usage::

    uv run python scripts/sync.py
    uv run python scripts/sync.py --db data/dev.db
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from scripts.auto_upload import CLAUDE_DIR, CODEX_DIR, process, target_dir
from trace_marketplace.storage.db import connect

log = logging.getLogger("sync")


def _already_staged(source: Path) -> bool:
    """True when ``data/raw/<fmt>/<name>`` exists with matching size + mtime.

    Cheap pre-filter so we don't burn an OpenAI embedding call on
    every file every time someone runs ``sync.py`` on a steady-state
    box. Content-level dedup happens downstream in ``ingest_file``
    via the trace-id INSERT OR REPLACE, but that doesn't help the
    embedding step, which has no pre-call hash check.
    """
    dest_dir = target_dir(source)
    if dest_dir is None:
        return False
    dest = dest_dir / source.name
    if not dest.exists():
        return False
    src_stat = source.stat()
    dst_stat = dest.stat()
    return src_stat.st_size == dst_stat.st_size and int(src_stat.st_mtime) == int(
        dst_stat.st_mtime
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/marketplace.db")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    conn = connect(args.db)
    try:
        scanned = ingested = skipped = 0
        for root, label in [(CLAUDE_DIR, "Claude Code"), (CODEX_DIR, "Codex")]:
            if not root.exists():
                log.info("skipping %s: %s not found", label, root)
                continue
            log.info("scanning %s: %s", label, root)
            for source in sorted(root.rglob("*.jsonl")):
                scanned += 1
                if _already_staged(source):
                    skipped += 1
                    continue
                process(source, conn)
                ingested += 1
        log.info(
            "done. scanned=%d ingested=%d skipped=%d",
            scanned,
            ingested,
            skipped,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
