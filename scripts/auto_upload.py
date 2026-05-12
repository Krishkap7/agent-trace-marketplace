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
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from trace_marketplace.enrich.failure_rules import persist_signals_for_trace
from trace_marketplace.ingest.pipeline import ingest_file
from trace_marketplace.search.runner import embed_and_recommend_similar
from trace_marketplace.storage.db import connect, init_schema

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

    Thread safety: watchdog calls ``on_modified`` / ``on_created`` from
    its observer thread while the main loop reads ``last_modified``
    from :func:`get_settled`. Without the lock, a new event firing
    mid-iteration in ``get_settled`` would raise
    ``RuntimeError: dictionary changed size during iteration`` and
    crash the daemon (the outer ``KeyboardInterrupt`` handler doesn't
    catch it). The lock is held for the smallest possible window in
    each method -- one dict assignment or a snapshot copy.
    """

    def __init__(self) -> None:
        self.last_modified: dict[Path, float] = {}
        self._lock = threading.Lock()

    def _record(self, src_path: str, is_directory: bool) -> None:
        if is_directory:
            return
        path = Path(src_path)
        if path.suffix == ".jsonl":
            now = time.time()
            with self._lock:
                self.last_modified[path] = now

    def on_modified(self, event: FileSystemEvent) -> None:
        self._record(event.src_path, event.is_directory)

    def on_created(self, event: FileSystemEvent) -> None:
        self._record(event.src_path, event.is_directory)


def get_settled(handler: SessionHandler, debounce: float) -> list[Path]:
    """Return + clear files whose last write was at least ``debounce`` seconds ago.

    Takes a snapshot of the handler dict under its lock, picks the
    settled paths, and removes them under the same lock. The
    snapshot-then-mutate pattern is what keeps the watchdog observer
    thread safe from a ``dictionary changed size during iteration``
    crash if a fresh event lands mid-pass.
    """
    now = time.time()
    with handler._lock:
        snapshot = list(handler.last_modified.items())
        settled = [p for p, t in snapshot if now - t >= debounce]
        for p in settled:
            handler.last_modified.pop(p, None)
    return settled


_SOURCE_ROOTS: tuple[tuple[Path, Path], ...] = (
    (CLAUDE_DIR, Path("data/raw/claude_code")),
    (CODEX_DIR, Path("data/raw/codex")),
)


def staged_path(source: Path) -> Path | None:
    """Map a watched-source path to its full ``data/raw/<fmt>/<...>`` dest.

    Preserves the relative path *under* the source root so two
    sessions that share a bare filename in different project /
    date directories don't collide on disk. Examples::

        ~/.claude/projects/projA/session.jsonl
            -> data/raw/claude_code/projA/session.jsonl

        ~/.codex/sessions/2026/05/12/abc.jsonl
            -> data/raw/codex/2026/05/12/abc.jsonl

    Returns ``None`` if the source isn't under either of the two
    known roots; the caller treats that as a skip rather than an
    error so a stray symlink under ``~/.claude/projects/`` can't
    crash the loop. (Caught by Cursor bugbot on commit c579f7e:
    the prior flat ``dest_dir / source.name`` layout broke
    ``sync.py``'s mtime+size dedup whenever two project subdirs
    contained a same-named ``.jsonl``.)
    """
    for root, staging in _SOURCE_ROOTS:
        try:
            rel = source.relative_to(root)
        except ValueError:
            continue
        return staging / rel
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

        dest = staged_path(source)
        if dest is None:
            log.warning("- %s: not under a known source root, skipping", source)
            return

        dest.parent.mkdir(parents=True, exist_ok=True)
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
    init_schema(conn)
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
