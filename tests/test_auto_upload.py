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

from scripts.auto_upload import (
    CLAUDE_DIR,
    CODEX_DIR,
    SessionHandler,
    get_settled,
    staged_path,
)


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
