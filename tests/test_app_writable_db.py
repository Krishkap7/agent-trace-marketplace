"""Regression tests for :func:`trace_marketplace.ui.app._resolve_writable_db_path`.

Streamlit Cloud checks the repo into a read-only filesystem, which
caused ``OperationalError: attempt to write a readonly database`` on
the first upload attempt against the hosted demo (May 2026). The
original hotfix used ``os.access(..., os.W_OK)`` to detect the
read-only mount, which is a known POSIX gotcha: ``access(2)`` only
checks the file's permission bits + UID, NOT whether the underlying
filesystem is read-only. The bug re-surfaced because the seed DB on
Streamlit Cloud has user-writable permission bits (regular file
created by git checkout) but the mount itself is read-only -- so
``os.access`` returned True, the copy was skipped, and the INSERT
trapped on ``EROFS``.

The current implementation uses ``os.open(path, os.O_RDWR)`` instead.
That actually asks the kernel for write access without modifying any
bytes; a read-only mount fails immediately with ``EROFS`` (surfaced
here as ``OSError``).

These tests pin five behaviours that matter for the demo to keep
working:

* writable source -> returned in place (local dev is unaffected),
* read-only source + no existing tmp copy -> copy created and tmp
  path returned,
* read-only source + EXISTING tmp copy -> tmp path returned WITHOUT
  overwriting (so an in-session upload isn't clobbered by a rerun),
* **permission bits look writable but kernel rejects O_RDWR** ->
  this is the exact Streamlit Cloud failure mode that the original
  ``os.access`` check missed; pin it as a regression guard,
* successful probe closes its file descriptor (FD-leak guard for
  long-running containers that hit this path on every cache miss).

We monkeypatch ``os.open`` and ``tempfile.gettempdir`` to exercise
the read-only branch deterministically without depending on the
actual file permissions of the test environment.
"""

from __future__ import annotations

import errno
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


def _make_fake_open_that_rejects(source: Path, errno_code: int):
    """Build a monkeypatch for ``os.open`` that simulates a read-only
    mount: any ``O_RDWR`` against the seed DB raises ``OSError`` with
    the given errno. Every other ``os.open`` call passes through to
    the real implementation so unrelated I/O (writes into tmp_path,
    log file rotation, etc.) keeps working.
    """
    real_open = os.open

    def fake_open(path: object, flags: int, *args, **kwargs):  # type: ignore[no-untyped-def]
        if Path(str(path)) == source and (flags & os.O_RDWR):
            raise OSError(errno_code, os.strerror(errno_code))
        return real_open(path, flags, *args, **kwargs)

    return fake_open


def test_writable_source_returned_in_place(tmp_path) -> None:
    """Local dev path: ``data/marketplace.db`` is writable, no copy."""
    source = tmp_path / "data" / "marketplace.db"
    _seed_db_file(source)
    # tmp_path is always writable, no monkeypatch needed.
    result = _resolve_writable_db_path(source)
    assert result == source


def test_readonly_source_triggers_copy_to_tempdir(tmp_path, monkeypatch) -> None:
    """Streamlit Cloud path: source mount is read-only -> copy into tempdir."""
    source = tmp_path / "data" / "marketplace.db"
    _seed_db_file(source, content=b"seed-corpus-bytes")

    monkeypatch.setattr(os, "open", _make_fake_open_that_rejects(source, errno.EROFS))

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
    existing_tmp_copy = fake_tmp / "marketplace.db"
    existing_tmp_copy.write_bytes(b"mutated-since-startup")

    monkeypatch.setattr(os, "open", _make_fake_open_that_rejects(source, errno.EROFS))
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(fake_tmp))

    result = _resolve_writable_db_path(source)

    assert result == existing_tmp_copy
    # The pre-existing contents must survive -- otherwise we'd clobber
    # in-session uploads on every Streamlit rerun.
    assert result.read_bytes() == b"mutated-since-startup"


def test_writable_permission_bits_but_readonly_mount_still_falls_back(
    tmp_path, monkeypatch
) -> None:
    """The exact Streamlit Cloud regression: the file's permission bits
    say "user-writable" (so ``os.access(..., os.W_OK)`` returns True),
    but the filesystem itself is mounted read-only -- so the actual
    ``O_RDWR`` ``open(2)`` syscall fails with ``EROFS``. The previous
    implementation missed this case entirely and shipped a "fix" that
    was effectively a no-op on the deployed demo.

    Pin it: even when ``os.access`` would lie, we MUST still detect
    the mount-level read-only via ``os.O_RDWR`` and fall back to
    /tmp. Otherwise the upload page hits
    ``OperationalError: attempt to write a readonly database`` on
    every INSERT.
    """
    source = tmp_path / "data" / "marketplace.db"
    _seed_db_file(source, content=b"seed-corpus")
    # Sanity check: the test harness's tmp_path file IS writable
    # by permission bits (so a hypothetical ``os.access`` check
    # would happily return True, just like on Streamlit Cloud).
    assert os.access(source, os.W_OK)

    monkeypatch.setattr(os, "open", _make_fake_open_that_rejects(source, errno.EROFS))

    fake_tmp = tmp_path / "fake-tmp"
    fake_tmp.mkdir()
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(fake_tmp))

    result = _resolve_writable_db_path(source)

    # Must have fallen back to /tmp despite the permission bits
    # claiming the source is writable.
    assert result == fake_tmp / "marketplace.db"
    assert result.read_bytes() == b"seed-corpus"


def test_open_succeeds_then_descriptor_is_closed(tmp_path) -> None:
    """The success path opens an FD and must close it -- otherwise a
    long-running Streamlit container would leak FDs on every rerun
    (cache_resource caches the connection, but the writability check
    still runs once per cache miss).

    We loop 50 times to make any leak visible quickly: if ``os.close``
    were skipped, this test would still pass on most platforms (FD
    table is large), but the implementation contract is clear and
    the loop documents the intent.
    """
    source = tmp_path / "data" / "marketplace.db"
    _seed_db_file(source)

    for _ in range(50):
        result = _resolve_writable_db_path(source)
        assert result == source
