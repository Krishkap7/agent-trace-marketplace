"""Tests for ``scripts/auto_upload.py``.

Scope: the debounce / queue-clearing logic in :func:`get_settled` and
:class:`SessionHandler`. The watchdog ``Observer`` and the
``process`` -> ``ingest_file`` chain are covered transitively by the
existing pipeline tests; this file only owns the bits that are unique
to the daemon (deciding *when* a file is ready to ingest).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import scripts.auto_upload as _au
from scripts.auto_upload import (
    CLAUDE_DIR,
    CODEX_DIR,
    SessionHandler,
    get_settled,
    process,
    staged_path,
)
from trace_marketplace.storage.db import connect, count_traces, init_schema


def test_settled_returns_quiet_files() -> None:
    """A file whose last write was older than the debounce is returned + cleared."""
    handler = SessionHandler()
    p = Path("/tmp/old.jsonl")
    handler.last_modified[p] = time.time() - 11

    settled = get_settled(handler, debounce=10)

    assert p in settled
    assert p not in handler.last_modified


def test_recent_files_not_settled() -> None:
    """A file written within the debounce window stays in the queue."""
    handler = SessionHandler()
    p = Path("/tmp/new.jsonl")
    handler.last_modified[p] = time.time() - 5

    assert get_settled(handler, debounce=10) == []
    assert p in handler.last_modified


def test_settled_only_returns_quiet_subset() -> None:
    """Mixed queue: quiet files come out, fresh files stay."""
    handler = SessionHandler()
    old = Path("/tmp/old.jsonl")
    fresh = Path("/tmp/fresh.jsonl")
    handler.last_modified[old] = time.time() - 15
    handler.last_modified[fresh] = time.time() - 2

    settled = get_settled(handler, debounce=10)

    assert settled == [old]
    assert old not in handler.last_modified
    assert fresh in handler.last_modified


def _fake_event(src_path: str, is_directory: bool = False):
    """Minimal stand-in for a watchdog ``FileSystemEvent``.

    We only touch the two attributes ``SessionHandler._record`` reads,
    so a tiny dataclass-shaped object is enough and we don't have to
    instantiate watchdog's real event classes (which want a watch
    reference on some platforms).
    """

    class _E:
        pass

    e = _E()
    e.src_path = src_path
    e.is_directory = is_directory
    return e


def test_handler_records_only_jsonl_files() -> None:
    """Non-.jsonl events are dropped before they enter the queue."""
    handler = SessionHandler()

    handler.on_modified(_fake_event("/tmp/session.jsonl"))
    handler.on_modified(_fake_event("/tmp/session.log"))
    handler.on_modified(_fake_event("/tmp/session.tmp"))

    assert list(handler.last_modified) == [Path("/tmp/session.jsonl")]


def test_handler_ignores_directory_events() -> None:
    """Directory create/modify events never enter the queue."""
    handler = SessionHandler()

    handler.on_created(_fake_event("/tmp/some/dir", is_directory=True))

    assert handler.last_modified == {}


def _fake_moved_event(src_path: str, dest_path: str, is_directory: bool = False):
    """Stand-in for watchdog's ``FileMovedEvent``.

    The real event class has both ``src_path`` and ``dest_path``; we
    forward both because the handler reads ``dest_path`` (the
    rename target) and falls back to ``src_path`` if ``dest_path``
    is missing.
    """

    class _E:
        pass

    e = _E()
    e.src_path = src_path
    e.dest_path = dest_path
    e.is_directory = is_directory
    return e


def test_handler_records_moved_files_by_destination() -> None:
    """Atomic-write rename pattern is captured via on_moved.

    Regression for the Cursor bugbot finding on PR #11 commit 09c1e2d:
    Linux's inotify reports ``open(tempfile) -> write -> rename(tempfile,
    target)`` as a single ``on_moved`` event, not the ``on_modified``
    /``on_created`` pair that file editors usually trigger. Without an
    ``on_moved`` handler the daemon would silently miss any agent that
    uses the standard atomic-write pattern.
    """
    handler = SessionHandler()

    handler.on_moved(_fake_moved_event("/tmp/.session.jsonl.swp", "/tmp/session.jsonl"))

    assert Path("/tmp/session.jsonl") in handler.last_modified
    assert Path("/tmp/.session.jsonl.swp") not in handler.last_modified


def test_handler_ignores_non_jsonl_moves() -> None:
    """A move whose destination isn't a .jsonl is dropped."""
    handler = SessionHandler()

    handler.on_moved(_fake_moved_event("/tmp/old.log", "/tmp/new.log"))

    assert handler.last_modified == {}


def test_handler_skips_moves_without_dest_path() -> None:
    """An ``on_moved`` event without a usable dest_path is a no-op.

    Some platform-specific event classes (or future watchdog
    versions) might not populate dest_path; we treat that defensively
    rather than crashing the observer thread.
    """
    handler = SessionHandler()

    handler.on_moved(_fake_moved_event("/tmp/old.jsonl", "", is_directory=False))

    assert handler.last_modified == {}


def test_staged_path_routes_known_sources() -> None:
    """Paths under the two supported roots route to the matching staging tree."""
    claude = CLAUDE_DIR / "projA" / "abc.jsonl"
    codex = CODEX_DIR / "2026" / "01" / "01" / "x.jsonl"

    assert staged_path(claude) == Path("data/raw/claude_code/projA/abc.jsonl")
    assert staged_path(codex) == Path("data/raw/codex/2026/01/01/x.jsonl")


def test_staged_path_returns_none_for_unknown_paths() -> None:
    """Anything outside the known roots returns None (caller will skip)."""
    assert staged_path(Path("/tmp/random/sess.jsonl")) is None


def test_get_settled_is_thread_safe_against_concurrent_writes() -> None:
    """get_settled must not raise even when a watchdog-style writer
    thread is mutating last_modified mid-iteration.

    Regression for the Cursor bugbot finding on PR #11: the original
    implementation iterated ``handler.last_modified.items()`` directly,
    which would raise ``RuntimeError: dictionary changed size during
    iteration`` whenever a fresh ``on_modified`` event fired during
    the pass and crash the daemon.

    The test hammers the handler from a background thread for a short
    window while the main thread runs ``get_settled`` in a tight loop.
    With the lock in place this completes without exceptions; without
    the lock it raises a RuntimeError within a handful of iterations.
    """
    handler = SessionHandler()
    stop = threading.Event()
    errors: list[BaseException] = []

    def writer() -> None:
        i = 0
        while not stop.is_set():
            handler._record(f"/tmp/sess-{i}.jsonl", is_directory=False)
            i = (i + 1) % 50  # cycle so the dict size oscillates

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    try:
        for _ in range(200):
            try:
                get_settled(handler, debounce=0.001)
            except BaseException as exc:  # noqa: BLE001 -- we want to see anything
                errors.append(exc)
                break
    finally:
        stop.set()
        t.join(timeout=1.0)

    assert not errors, f"get_settled raised under concurrent writes: {errors!r}"


_CLAUDE_CODE_FIXTURE = Path("tests/fixtures/claude_code_minimal.jsonl")


def test_process_inserts_row_into_db(tmp_path: Path) -> None:
    """process() must copy the session file AND write a row to the SQLite DB.

    Regression test for the gap where the daemon copied files to
    ``data/raw/`` but the enrichment chain (ingest_file ->
    persist_signals_for_trace -> embed_and_recommend_similar ->
    persist_judge_for_trace -> persist_events_for_trace ->
    persist_recommendations_for_trace) was not wired up to the DB connection.

    We use no-API-key conditions so the LLM passes degrade gracefully;
    the test pins that the basic DB write (raw_blob + atif +
    failure_signals) happens unconditionally.
    """
    raw = _CLAUDE_CODE_FIXTURE.read_text(encoding="utf-8")

    fake_claude_root = tmp_path / "fake_claude"
    fake_claude_root.mkdir()
    source = fake_claude_root / "session.jsonl"
    source.write_text(raw, encoding="utf-8")

    staging_root = tmp_path / "staging"
    db_path = tmp_path / "test.db"

    orig_roots = _au._SOURCE_ROOTS
    _au._SOURCE_ROOTS = ((fake_claude_root, staging_root / "claude_code"),)
    try:
        conn = connect(db_path)
        init_schema(conn)
        assert count_traces(conn) == 0

        process(source, conn)

        assert count_traces(conn) == 1, "process() must insert a row into the DB"
        row = conn.execute(
            "SELECT source_format, raw_blob, atif, failure_signals FROM traces"
        ).fetchone()
        assert row["source_format"] == "claude_code"
        assert row["raw_blob"], "raw_blob must be populated"
        assert row["atif"] is not None, "ATIF must be populated for a valid Claude Code file"
        assert row["failure_signals"] is not None, "rule pass must populate failure_signals"

        staged = (staging_root / "claude_code" / "session.jsonl")
        assert staged.exists(), "process() must copy the file to the staging dir"
        conn.close()
    finally:
        _au._SOURCE_ROOTS = orig_roots


def test_process_does_not_raise_on_vanished_file(tmp_path: Path) -> None:
    """process() must return silently when the source file is gone."""
    db_path = tmp_path / "test.db"
    conn = connect(db_path)
    init_schema(conn)
    process(Path("/nonexistent/session.jsonl"), conn)
    assert count_traces(conn) == 0
    conn.close()


def test_staged_path_preserves_distinguishing_subdir() -> None:
    """Two same-named .jsonl files in different source subdirs stage to
    distinct destinations so sync.py's mtime+size dedup keeps working.

    Regression for the flat-staging-dir bug Cursor bugbot flagged on
    PR #11: ``dest_dir / source.name`` collapsed both
    ``~/.claude/projects/projA/session.jsonl`` and
    ``~/.claude/projects/projB/session.jsonl`` into the same
    ``data/raw/claude_code/session.jsonl`` so one always overwrote
    the other.
    """
    proj_a = CLAUDE_DIR / "projA" / "session.jsonl"
    proj_b = CLAUDE_DIR / "projB" / "session.jsonl"

    dest_a = staged_path(proj_a)
    dest_b = staged_path(proj_b)

    assert dest_a is not None and dest_b is not None
    assert dest_a != dest_b
