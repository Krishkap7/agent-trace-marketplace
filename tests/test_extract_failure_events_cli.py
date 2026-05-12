"""CLI-level tests for :mod:`scripts.extract_failure_events`.

Focused on the dry-run / live branching contract that
:func:`scripts.extract_failure_events.main` exposes. We don't exercise
``_run_live`` (that would require either real Opus 4.7 calls or
extensive mocking of the SDK + ThreadPoolExecutor); instead we pin the
two invariants Cursor Bugbot flagged on PR #13:

* ``--limit N --dry-run`` actually previews N traces (was previously
  showing the full-corpus cost),
* ``--limit 0`` short-circuits cleanly without invoking either path.

Both invariants are part of the user-facing contract: dry-run must be
a faithful preview of what ``--confirm`` would burn, otherwise an
operator could see ``$47`` and not realise their ``--limit 5`` reduced
the actual spend to under a dollar.
"""

from __future__ import annotations

from pathlib import Path

from scripts.extract_failure_events import main
from trace_marketplace.storage.db import connect, init_schema, insert_trace


def _seed_targets(db_path: Path, n: int) -> None:
    """Create a DB with ``n`` selectable target rows.

    Selection criteria from ``_select_targets``: parseable ATIF,
    ``has_error = 1`` (or ``failure_label IS NOT NULL``), and
    ``failure_events IS NULL``. We satisfy the first two and leave the
    third untouched (NULL by default on a fresh row).
    """
    conn = connect(db_path)
    try:
        init_schema(conn)
        for i in range(n):
            insert_trace(
                conn,
                trace_id=f"target-{i:03d}",
                source_format="synth",
                raw_blob="{}",
                atif='{"schema_version": "ATIF-v1.7", "steps": []}',
                agent_name="test-agent",
                model="test-model",
                num_steps=1,
                num_tool_calls=0,
                has_error=1,
            )
        conn.commit()
    finally:
        conn.close()


def test_dry_run_respects_limit_flag(tmp_path, capsys) -> None:
    """``--limit N --dry-run`` must report cost projections for N
    traces, not the full corpus.

    Previously this returned the full-corpus cost because ``--limit``
    was only applied inside ``_run_live``. The fix moved the limit
    application up to ``main()`` so both branches see the same target
    list.
    """
    db = tmp_path / "limit.db"
    _seed_targets(db, n=10)

    rc = main(["--db", str(db), "--dry-run", "--limit", "3"])
    assert rc == 0
    output = capsys.readouterr().out
    # The dry-run output frames every line with a [DRY RUN] prefix and
    # writes ``Total traces to extract:  <N>``. Assert the trimmed count
    # appears, not the seeded count.
    assert "Total traces to extract:  3" in output
    assert "Total traces to extract:  10" not in output


def test_dry_run_without_limit_uses_full_target_count(tmp_path, capsys) -> None:
    """Sanity check the other direction: no ``--limit`` -> full corpus."""
    db = tmp_path / "no_limit.db"
    _seed_targets(db, n=7)

    rc = main(["--db", str(db), "--dry-run"])
    assert rc == 0
    output = capsys.readouterr().out
    assert "Total traces to extract:  7" in output


def test_limit_zero_short_circuits_cleanly(tmp_path, capsys) -> None:
    """``--limit 0`` (and negative; argparse won't accept negative
    natively but max(0, ...) clamps anyway) must not invoke either
    branch -- the user gets a friendly message and a clean exit."""
    db = tmp_path / "zero.db"
    _seed_targets(db, n=5)

    rc = main(["--db", str(db), "--dry-run", "--limit", "0"])
    assert rc == 0
    output = capsys.readouterr().out
    assert "reduces the selection to zero traces" in output
    # No dry-run banner should be emitted on the empty path.
    assert "Total traces to extract:" not in output


def test_no_selection_targets_short_circuits_before_limit_check(
    tmp_path, capsys
) -> None:
    """When the DB has nothing to extract at all (empty selection), the
    script prints the "no traces need extraction" message and exits.
    This is the path that runs when the operator re-invokes the script
    after a successful live run -- everything is already populated."""
    db = tmp_path / "empty.db"
    conn = connect(db)
    try:
        init_schema(conn)
        # Insert a row that does NOT match the selection criteria
        # (failure_events already populated => idempotent skip).
        insert_trace(
            conn,
            trace_id="already-done",
            source_format="synth",
            raw_blob="{}",
            atif='{"schema_version": "ATIF-v1.7"}',
            agent_name="a",
            model="m",
            num_steps=1,
            num_tool_calls=0,
            has_error=1,
        )
        conn.execute(
            "UPDATE traces SET failure_events = ? WHERE id = ?",
            ("[]", "already-done"),
        )
        conn.commit()
    finally:
        conn.close()

    rc = main(["--db", str(db), "--dry-run", "--limit", "5"])
    assert rc == 0
    output = capsys.readouterr().out
    assert "No traces need extraction" in output


def test_missing_db_returns_non_zero_exit(tmp_path, capsys) -> None:
    """Defensive: a missing DB shouldn't crash with a stack trace;
    we want a non-zero exit and a clear error message."""
    db = tmp_path / "does-not-exist.db"
    rc = main(["--db", str(db), "--dry-run"])
    assert rc == 1
