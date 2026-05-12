"""Integration tests for the slice-5 upload pipeline.

The Streamlit page itself (``trace_marketplace.ui.upload_view``) is
exercised by hand on a running app -- there's no headless renderer in
the test stack. What we *can* pin without a runtime is the
upload-equivalent sequence the page invokes for each file:

    write_bytes -> ingest_file -> persist_signals_for_trace

If that wiring is correct and idempotent end-to-end on a real Claude
Code fixture, the page is a thin Streamlit wrapper that adds error
isolation + UI affordances around it. Those affordances are exercised
manually; the contract this test guarantees is that the page can't
silently drop a trace because of an integration bug between the
ingest pipeline and the rule-pass helper.

Three behaviours pinned:

1. **Happy path.** A real Claude Code JSONL upload lands in the DB
   with the expected ``source_format``, an ATIF blob, and the
   ``failure_signals`` column populated by the inline rule pass.
2. **Unknown-format graceful fallback.** Random plain-text payload
   still produces a row (raw_blob only), the rule helper returns
   ``None`` without raising, and the DB column for signals stays NULL.
3. **Idempotency.** Re-running the same payload through the same
   sequence updates the existing row instead of duplicating it; the
   ``failure_signals`` column reflects the latest pass.
"""

from __future__ import annotations

import json
from pathlib import Path

from trace_marketplace.enrich.failure_rules import persist_signals_for_trace
from trace_marketplace.ingest.pipeline import ingest_file
from trace_marketplace.storage.db import connect, count_traces, init_schema

_CLAUDE_CODE_FIXTURE = Path("tests/fixtures/claude_code_minimal.jsonl")


def _ingest_and_run_rules(
    conn, source_path: Path
) -> tuple[str, dict[str, bool] | None]:
    """The upload page's per-file sequence, expressed without Streamlit.

    Mirrors :func:`trace_marketplace.ui.upload_view._process_upload`
    minus the try/except scaffolding and Streamlit session-state
    handling -- the part that's pure business logic.
    """
    result = ingest_file(source_path, conn)
    atif_text: str | None = None
    if result.validated:
        row = conn.execute(
            "SELECT atif FROM traces WHERE id = ?", (result.trace_id,)
        ).fetchone()
        atif_text = row["atif"] if row else None
    signals = persist_signals_for_trace(conn, result.trace_id, atif_text)
    conn.commit()
    return result.trace_id, signals


def test_upload_pipeline_ingests_claude_code_and_runs_rules(tmp_path: Path) -> None:
    """Happy path: a real Claude Code JSONL produces an ingested row
    with ATIF + populated failure_signals."""
    raw = _CLAUDE_CODE_FIXTURE.read_text(encoding="utf-8")
    upload_path = tmp_path / "uploaded.jsonl"
    upload_path.write_text(raw, encoding="utf-8")
    db = tmp_path / "marketplace.db"

    with connect(db) as conn:
        init_schema(conn)
        trace_id, signals = _ingest_and_run_rules(conn, upload_path)
        row = conn.execute("SELECT * FROM traces WHERE id = ?", (trace_id,)).fetchone()

    assert row is not None, "upload didn't reach the DB"
    assert row["source_format"] == "claude_code"
    assert row["atif"] is not None, "Claude Code adapter must produce ATIF"
    assert row["failure_signals"] is not None, "inline rule pass should write JSON"

    # The rule-pass JSON must be the same dict we returned in-memory --
    # if these diverge it means the helper wrote one thing and returned
    # another, which would be a subtle bug.
    persisted = json.loads(row["failure_signals"])
    assert isinstance(signals, dict)
    assert persisted == signals

    # All five rule keys must be present so the UI's signal-summary
    # rendering can rely on a stable shape.
    assert set(signals.keys()) == {
        "hit_max_turns",
        "repeated_tool_call_loop",
        "ended_on_error",
        "abandonment",
        "incomplete_session",
    }


def test_upload_pipeline_handles_unknown_format_as_raw_blob(tmp_path: Path) -> None:
    """A random .txt file with no detectable shape still produces a row
    (raw_blob only), and the rule helper returns ``None`` without raising."""
    upload_path = tmp_path / "random_notes.txt"
    upload_path.write_text(
        "Just some notes from a meeting. Nothing JSON-y here.\n", encoding="utf-8"
    )
    db = tmp_path / "marketplace.db"

    with connect(db) as conn:
        init_schema(conn)
        trace_id, signals = _ingest_and_run_rules(conn, upload_path)
        row = conn.execute("SELECT * FROM traces WHERE id = ?", (trace_id,)).fetchone()

    assert row is not None
    assert row["source_format"] == "unknown"
    assert row["atif"] is None, "unknown format must leave atif NULL"
    assert row["raw_blob"], "raw_blob must be populated even when format is unknown"
    # The helper's contract: ``atif_text=None`` short-circuits to None
    # so the upload view can detect "no signals computable" without
    # parsing a NULL.
    assert signals is None
    # And the column stays NULL because the helper short-circuits before
    # the UPDATE ever runs.
    assert row["failure_signals"] is None


def test_upload_pipeline_is_idempotent_across_repeat_uploads(tmp_path: Path) -> None:
    """Re-uploading the same payload must update the existing row
    rather than duplicate. INSERT OR REPLACE is the storage-layer
    invariant; this test guarantees the upload-equivalent sequence
    preserves it (i.e. nothing in the rule pass accidentally inserts
    a sibling row)."""
    raw = _CLAUDE_CODE_FIXTURE.read_text(encoding="utf-8")
    db = tmp_path / "marketplace.db"

    with connect(db) as conn:
        init_schema(conn)

        first_upload = tmp_path / "uploaded.jsonl"
        first_upload.write_text(raw, encoding="utf-8")
        trace_id_a, _ = _ingest_and_run_rules(conn, first_upload)

        # Simulate the user re-uploading by writing the same bytes to
        # a differently-named tempfile (the upload view stages each
        # submission into its own TemporaryDirectory).
        second_upload = tmp_path / "uploaded-again.jsonl"
        second_upload.write_text(raw, encoding="utf-8")
        trace_id_b, _ = _ingest_and_run_rules(conn, second_upload)

        assert trace_id_a == trace_id_b, "same payload must hash to same trace_id"
        assert count_traces(conn) == 1, (
            "second upload should INSERT OR REPLACE, not add a sibling row"
        )
