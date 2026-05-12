"""Regression tests for :func:`trace_marketplace.ui.app._resolve_writable_db_path`.

Streamlit Cloud checks the repo into a read-only filesystem, which
caused ``OperationalError: attempt to write a readonly database`` on
the first upload attempt against the hosted demo (May 2026). The fix
copies the seed DB into the OS tempdir on first connection. These
tests pin the three behaviours that matter for the demo to keep
working:

* writable source -> returned in place (local dev is unaffected),
* read-only source + no existing tmp copy -> copy created and tmp
  path returned,
* read-only source + EXISTING tmp copy -> tmp path returned WITHOUT
  overwriting (so an in-session upload isn't clobbered by a rerun).

We monkeypatch ``os.access`` and ``tempfile.gettempdir`` to exercise
both branches deterministically without depending on the actual file
permissions of the test environment.
"""

from __future__ import annotations

import os
from pathlib import Path

from trace_marketplace.ui.app import _resolve_writable_db_path


def _seed_db_file(path: Path, *, content: bytes = b"original") -> None:
    """Create a fake "DB" with recognisable contents.

    We don't need a real SQLite file for these tests because
    ``_resolve_writable_db_path`` only checks writability and copies
    bytes -- it never opens the file as SQLite. Using raw bytes lets
    us assert on the copy by reading them back.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_writable_source_returned_in_place(tmp_path) -> None:
    """Local dev path: ``data/marketplace.db`` is writable, no copy."""
    source = tmp_path / "data" / "marketplace.db"
    _seed_db_file(source)
    # tmp_path is always writable, no monkeypatch needed.
    result = _resolve_writable_db_path(source)
    assert result == source


def test_readonly_source_triggers_copy_to_tempdir(tmp_path, monkeypatch) -> None:
    """Streamlit Cloud path: source is read-only -> copy into tempdir."""
    source = tmp_path / "data" / "marketplace.db"
    _seed_db_file(source, content=b"seed-corpus-bytes")

    # Pretend the source file is read-only without actually chmod-ing
    # (chmod behaviour varies across CI environments and macOS APFS).
    real_access = os.access

    def fake_access(path: object, mode: int) -> bool:
        if Path(str(path)) == source and mode == os.W_OK:
            return False
        return real_access(path, mode)

    monkeypatch.setattr(os, "access", fake_access)

    # Redirect tempfile.gettempdir() into the per-test tmp_path so the
    # writable copy lands somewhere we control + clean up cleanly.
    fake_tmp = tmp_path / "fake-tmp"
    fake_tmp.mkdir()
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(fake_tmp))

    result = _resolve_writable_db_path(source)

    assert result == fake_tmp / "marketplace.db"
    assert result.exists()
    assert result.read_bytes() == b"seed-corpus-bytes"


def test_readonly_source_with_existing_tmp_copy_is_not_overwritten(
    tmp_path, monkeypatch
) -> None:
    """In-session safety: if the tmp copy already exists (e.g. previous
    rerun within the same Streamlit container), DO NOT overwrite it.
    Otherwise a Streamlit rerun would wipe any uploads the user did
    during this session.
    """
    source = tmp_path / "data" / "marketplace.db"
    _seed_db_file(source, content=b"original-seed")

    fake_tmp = tmp_path / "fake-tmp"
    fake_tmp.mkdir()
    # Pre-existing tmp copy with DIFFERENT contents (simulating
    # an in-session upload already persisted).
    existing_tmp_copy = fake_tmp / "marketplace.db"
    existing_tmp_copy.write_bytes(b"mutated-since-startup")

    real_access = os.access

    def fake_access(path: object, mode: int) -> bool:
        if Path(str(path)) == source and mode == os.W_OK:
            return False
        return real_access(path, mode)

    monkeypatch.setattr(os, "access", fake_access)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(fake_tmp))

    result = _resolve_writable_db_path(source)

    assert result == existing_tmp_copy
    # The pre-existing contents must survive -- otherwise we'd clobber
    # in-session uploads on every Streamlit rerun.
    assert result.read_bytes() == b"mutated-since-startup"
