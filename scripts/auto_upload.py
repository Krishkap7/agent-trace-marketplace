"""Auto-upload daemon for Claude Code + Codex session files.

Watches ``~/.claude/projects/`` and ``~/.codex/sessions/`` recursively.
When a ``.jsonl`` file goes quiet for ``--debounce-seconds`` (default 10),
the file is copied into ``data/raw/{claude_code,codex}/`` and run through
``ingest_file`` -> ``persist_signals_for_trace`` -> ``embed_and_recommend_similar``
so the row lands in ``data/marketplace.db`` fully enriched (raw_blob +
ATIF + failure_signals + embedding) the way the upload page would have
ingested it.

Local-only. The hosted Streamlit deployment picks up new traces when the
user pushes the updated ``data/marketplace.db`` to GitHub.

Usage::

    uv run python scripts/auto_upload.py
    uv run python scripts/auto_upload.py --debounce-seconds 5 --db data/dev.db

Ctrl+C exits cleanly.

Scope notes
-----------
* Claude Code + Codex only. Cursor session storage is a SQLite database
  held open by the editor process; tailing it without a polling export
  is out of scope (tracked as v2).
* Foreground process. No systemd / launchd integration; if the user
  closes the shell, the daemon dies and they re-launch it on their next
  session.
* Debounce is a per-file timer reset on every write, not a global
  cooldown -- a long Claude Code session that keeps appending stays out
  of the queue until the agent actually stops writing.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sqlite3
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from trace_marketplace.enrich.failure_rules import persist_signals_for_trace
from trace_marketplace.ingest.pipeline import ingest_file
from trace_marketplace.search.runner import embed_and_recommend_similar
from trace_marketplace.storage.db import connect

log = logging.getLogger("auto_upload")

CLAUDE_DIR = Path.home() / ".claude" / "projects"
CODEX_DIR = Path.home() / ".codex" / "sessions"


class SessionHandler(FileSystemEventHandler):
    """Records the last write time of every ``.jsonl`` under watched dirs.

    Watchdog fires a flurry of ``on_modified`` events for a single
    flush (open, write, close = 2-3 events on macOS / Linux), so the
    actual "is it ready to ingest" decision is moved into
    :func:`get_settled`, which the main loop polls. The handler itself
    is dirt cheap: dict insert per event, no I/O.
    """

    def __init__(self) -> None:
        self.last_modified: dict[Path, float] = {}

    def _record(self, src_path: str, is_directory: bool) -> None:
        if is_directory:
            return
        path = Path(src_path)
        if path.suffix == ".jsonl":
            self.last_modified[path] = time.time()

    def on_modified(self, event: FileSystemEvent) -> None:
        self._record(event.src_path, event.is_directory)

    def on_created(self, event: FileSystemEvent) -> None:
        self._record(event.src_path, event.is_directory)


def get_settled(handler: SessionHandler, debounce: float) -> list[Path]:
    """Return + clear files whose last write was at least ``debounce`` seconds ago.

    The handler dict is mutated in place: returned paths are removed
    so subsequent passes don't re-ingest the same file. If the agent
    writes more to a removed file later, the next event re-adds it
    with a fresh timestamp.
    """
    now = time.time()
    settled = [p for p, t in handler.last_modified.items() if now - t >= debounce]
    for p in settled:
        handler.last_modified.pop(p, None)
    return settled


def target_dir(path: Path) -> Path | None:
    """Map a watched-source path to its ``data/raw/`` staging dir.

    Returns ``None`` if the path isn't under either of the two known
    source roots; the caller treats that as a skip rather than an
    error so a stray symlink under ``~/.claude/projects/`` can't
    crash the loop.
    """
    s = path.as_posix()
    if "/.claude/projects/" in s or s.endswith("/.claude/projects"):
        return Path("data/raw/claude_code")
    if "/.codex/sessions/" in s or s.endswith("/.codex/sessions"):
        return Path("data/raw/codex")
    return None


def process(source: Path, conn: sqlite3.Connection) -> None:
    """Copy + ingest + enrich one session file. Must never raise.

    Mirrors the upload page's per-file flow
    (``trace_marketplace.ui.upload_view._process_upload``):

    1. ``shutil.copy2`` into ``data/raw/<format>/`` so the trace has a
       stable on-disk home outside ``~/.claude`` / ``~/.codex`` (those
       dirs get rotated/deleted by the agents themselves).
    2. ``ingest_file`` -> writes raw_blob + ATIF.
    3. ``persist_signals_for_trace`` -> rule-based ``failure_signals``
       and, if ``has_error`` was previously NULL, the boolean too.
    4. ``embed_and_recommend_similar`` -> OpenAI embedding (gated on
       ``OPENAI_API_KEY``; never raises) so the new trace is
       immediately searchable from the NL query box.

    Every failure path logs and returns -- a malformed adapter, a
    missing API key, even a vanished source file are all "log it and
    keep watching" cases.
    """
    try:
        if not source.exists():
            log.warning("- %s: file vanished before processing", source.name)
            return

        dest_dir = target_dir(source)
        if dest_dir is None:
            log.warning("- %s: not under a known source root, skipping", source)
            return

        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / source.name
        shutil.copy2(source, dest)

        result = ingest_file(dest, conn)
        conn.commit()
        trace_id = result.trace_id

        row = conn.execute(
            "SELECT atif FROM traces WHERE id = ?", (trace_id,)
        ).fetchone()
        atif_text = row["atif"] if row is not None else None
        signals = persist_signals_for_trace(conn, trace_id, atif_text)
        conn.commit()
        fired = sorted(k for k, v in (signals or {}).items() if v)

        embed = embed_and_recommend_similar(conn, trace_id, top_k=3)

        validated = "validated" if result.validated else "raw_blob only"
        log.info("ingested %s -> trace_id=%s (%s)", source.name, trace_id, validated)
        if fired:
            log.info("  signals: %s", fired)
        if embed.skipped_reason:
            log.info("  embedding skipped: %s", embed.skipped_reason)
        elif embed.similar:
            preview = ", ".join(hit.trace_id for hit in embed.similar[:3])
            log.info("  similar: %s", preview)
    except Exception as exc:
        log.warning("- %s: %s", source.name, exc)


def _build_observer(handler: SessionHandler) -> Observer:
    observer = Observer()
    for source_dir, label in [(CLAUDE_DIR, "Claude Code"), (CODEX_DIR, "Codex")]:
        if source_dir.exists():
            observer.schedule(handler, str(source_dir), recursive=True)
            log.info("watching %s: %s", label, source_dir)
        else:
            log.info("skipping %s: %s not found", label, source_dir)
    return observer


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Auto-upload daemon for Claude Code + Codex sessions."
    )
    ap.add_argument(
        "--debounce-seconds",
        type=float,
        default=10.0,
        help="Quiet period (sec) on a .jsonl before ingest fires (default: 10).",
    )
    ap.add_argument(
        "--db",
        default="data/marketplace.db",
        help="Path to the local marketplace SQLite database.",
    )
    ap.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between settle-check passes in the main loop (default: 2).",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    conn = connect(args.db)
    handler = SessionHandler()
    observer = _build_observer(handler)

    observer.start()
    log.info(
        "daemon started; debounce=%.0fs; Ctrl+C to exit",
        args.debounce_seconds,
    )

    try:
        while True:
            time.sleep(args.poll_interval)
            for path in get_settled(handler, args.debounce_seconds):
                process(path, conn)
    except KeyboardInterrupt:
        log.info("stopping...")
    finally:
        observer.stop()
        observer.join()
        conn.close()


if __name__ == "__main__":
    main()
