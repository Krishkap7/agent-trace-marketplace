"""Tests for ``scripts/auto_upload.py``.

Scope: the debounce / queue-clearing logic in :func:`get_settled` and
:class:`SessionHandler`. The watchdog ``Observer`` and the
``process`` -> ``ingest_file`` chain are covered transitively by the
existing pipeline tests; this file only owns the bits that are unique
to the daemon (deciding *when* a file is ready to ingest).
"""

from __future__ import annotations

import time
from pathlib import Path

from scripts.auto_upload import SessionHandler, get_settled, target_dir


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


def test_target_dir_routes_known_sources() -> None:
    """Paths under the two supported roots route to the matching staging dir."""
    claude = Path.home() / ".claude/projects/foo/abc.jsonl"
    codex = Path.home() / ".codex/sessions/2026/01/01/x.jsonl"

    assert target_dir(claude) == Path("data/raw/claude_code")
    assert target_dir(codex) == Path("data/raw/codex")


def test_target_dir_returns_none_for_unknown_paths() -> None:
    """Anything outside the known roots returns None (caller will skip)."""
    assert target_dir(Path("/tmp/random/sess.jsonl")) is None
