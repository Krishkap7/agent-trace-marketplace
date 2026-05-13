"""Unit tests for the inline-classification helpers used by the upload flow.

The upload page (``trace_marketplace.ui.upload_view._process_upload``)
runs the full classification pipeline on every validated upload:

    rule pass -> Sonnet judge -> Opus events extractor ->
    recommendations cache

The two LLM passes are wrapped in helpers that match the no-raise
contract the upload pipeline expects: every error path returns
``None`` instead of raising, so one failed pass can't tank the rest
of the row.

These tests pin those contracts:

* :func:`persist_judge_for_trace` --
  - happy path persists ``failure_label`` + ``failure_label_reasoning``
    + ``failure_label_source = 'sonnet_judge'`` and returns a
    :class:`JudgeResult`.
  - missing trace, NULL atif, and missing ``ANTHROPIC_API_KEY`` all
    return ``None`` without writing anything.
  - judge call raising ``JudgeError`` (or any unexpected exception)
    is swallowed; helper returns ``None``.

* :func:`persist_events_for_trace` --
  - happy path persists ``failure_events``, OVERWRITES
    ``failure_label`` with the events-derived summary, sets
    ``failure_label_source = 'opus_events_derived'``, sets
    ``ultimately_succeeded`` from ``derive_ultimately_succeeded``,
    and returns an :class:`EventsResult`.
  - When Opus reports unrecovered events, ``has_error`` gets bumped
    to ``1`` even if the rule pass had it as ``0`` (defensive
    write-back so the recoveries-panel gate stays consistent).
  - When Opus reports only recovered events (or empty list),
    ``has_error`` is preserved (we don't downgrade the rule pass).
  - missing trace, NULL atif, missing ``ANTHROPIC_API_KEY``, and API
    failures all return ``None`` without writing anything (or, in the
    API-failure case, without overwriting an earlier judge result).

We use an in-memory SQLite DB for both helpers because they do real
SELECT/UPDATE work; mocking out the connection would over-couple the
tests to query strings. Anthropic clients are mocked using the same
``_make_client`` / ``_response`` / ``_tool_use_block`` pattern as
``test_failure_events.py``.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any
from unittest.mock import MagicMock

import pytest

from trace_marketplace.enrich.failure_events import persist_events_for_trace
from trace_marketplace.enrich.failure_judge import (
    JudgeError,
    JudgeResult,
    persist_judge_for_trace,
)
from trace_marketplace.enrich.failure_taxonomy import FailureLabel
from trace_marketplace.storage.db import init_schema


# ---------- shared fixtures ----------


_TRIVIAL_ATIF: dict[str, Any] = {
    "schema_version": "ATIF-v1.7",
    "agent": {"name": "test-agent", "model_name": "test-model"},
    "steps": [
        {"source": "user", "message": "Fix the import error in foo.py."},
        {"source": "agent", "message": "Looking at it now."},
    ],
}


@pytest.fixture
def conn(monkeypatch) -> sqlite3.Connection:
    """In-memory SQLite with the production schema applied + one
    pre-inserted trace row at id ``trace-under-test``.

    Also clears ``ANTHROPIC_API_KEY`` by default so tests have to
    explicitly set it when they want the helper's API path to fire.
    Tests that want the no-API-key branch get the right behaviour
    automatically; tests that want the happy path opt in via
    ``monkeypatch.setenv``.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    init_schema(c)
    c.execute(
        """
        INSERT INTO traces (id, source_format, raw_blob, atif, ingested_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "trace-under-test",
            "claude_code",
            b"raw",
            json.dumps(_TRIVIAL_ATIF),
            "2026-05-12T00:00:00Z",
        ),
    )
    c.commit()
    return c


def _row(conn: sqlite3.Connection, trace_id: str = "trace-under-test") -> sqlite3.Row:
    return conn.execute("SELECT * FROM traces WHERE id = ?", (trace_id,)).fetchone()


# ---------- judge helper ----------


def _judge_text_response_block(label: str, reasoning: str) -> Any:
    """Build the ``submit_label`` tool_use block the Sonnet judge
    returns -- the production prompt forces this exact tool name."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = "submit_label"
    block.input = {"label": label, "reasoning": reasoning}
    return block


def _response(*content_blocks: Any, in_tok: int = 100, out_tok: int = 50) -> Any:
    response = MagicMock()
    response.content = list(content_blocks)
    response.usage = MagicMock(input_tokens=in_tok, output_tokens=out_tok)
    return response


def _make_client(responses: list[Any]) -> Any:
    client = MagicMock()
    iterator = iter(responses)

    def _create(**_kwargs: Any) -> Any:
        try:
            return next(iterator)
        except StopIteration as exc:
            raise AssertionError("Exhausted mocked responses") from exc

    client.messages.create.side_effect = _create
    return client


def test_judge_helper_happy_path_persists_label_reasoning_and_source(
    conn, monkeypatch
) -> None:
    """When the judge returns a valid label, the helper writes
    failure_label + reasoning + source = 'sonnet_judge' and returns
    a populated JudgeResult."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = _make_client(
        [
            _response(
                _judge_text_response_block(
                    label=FailureLabel.LOOP.value,
                    reasoning="The agent retried the same failed test 8 times "
                    "without changing strategy.",
                )
            )
        ]
    )

    result = persist_judge_for_trace(conn, "trace-under-test", client=client)

    assert isinstance(result, JudgeResult)
    assert result.label is FailureLabel.LOOP

    row = _row(conn)
    assert row["failure_label"] == "loop"
    assert "retried the same failed test" in (row["failure_label_reasoning"] or "")
    assert row["failure_label_source"] == "sonnet_judge"


def test_judge_helper_returns_none_when_api_key_missing(conn) -> None:
    """No ``ANTHROPIC_API_KEY`` -> graceful skip, no DB write, no
    Anthropic client construction. This is the Streamlit Cloud demo
    case (no secrets configured)."""
    result = persist_judge_for_trace(conn, "trace-under-test")
    assert result is None
    row = _row(conn)
    assert row["failure_label"] is None
    assert row["failure_label_source"] is None


def test_judge_helper_returns_none_for_missing_trace(conn, monkeypatch) -> None:
    """Trace id not in the table -> None. No client constructed even
    though the API key is set, because the early SELECT short-circuits."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    sentinel_client = MagicMock()
    result = persist_judge_for_trace(conn, "does-not-exist", client=sentinel_client)
    assert result is None
    sentinel_client.messages.create.assert_not_called()


def test_judge_helper_returns_none_for_null_atif(conn, monkeypatch) -> None:
    """Raw-blob upload (``atif IS NULL``) -> None. There's no semantic
    content to judge, so calling Anthropic would burn tokens for no
    benefit."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    conn.execute("UPDATE traces SET atif = NULL WHERE id = ?", ("trace-under-test",))
    sentinel_client = MagicMock()
    result = persist_judge_for_trace(conn, "trace-under-test", client=sentinel_client)
    assert result is None
    sentinel_client.messages.create.assert_not_called()


def test_judge_helper_swallows_judge_error(conn, monkeypatch) -> None:
    """A JudgeError (or any unexpected exception from the judge call)
    must NOT propagate. The upload pipeline depends on the no-raise
    contract; a per-trace failure should leave failure_label NULL
    without breaking the upload."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    failing_client = MagicMock()
    failing_client.messages.create.side_effect = JudgeError("simulated")

    result = persist_judge_for_trace(conn, "trace-under-test", client=failing_client)
    assert result is None
    row = _row(conn)
    assert row["failure_label"] is None


def test_judge_helper_swallows_unexpected_exceptions(conn, monkeypatch) -> None:
    """Network errors / SDK bugs etc. all need to be swallowed too --
    same no-raise contract as JudgeError but exercised via a generic
    exception subclass to guarantee we don't accidentally narrow the
    except clause to only catch JudgeError."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    failing_client = MagicMock()
    failing_client.messages.create.side_effect = RuntimeError("network down")
    result = persist_judge_for_trace(conn, "trace-under-test", client=failing_client)
    assert result is None
    row = _row(conn)
    assert row["failure_label"] is None


# ---------- events helper ----------


def _events_tool_use_block(events_payload: list[dict[str, Any]]) -> Any:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "record_failure_events"
    block.input = {"events": events_payload}
    return block


def test_events_helper_happy_path_persists_events_and_overwrites_label(
    conn, monkeypatch
) -> None:
    """When Opus returns events, helper persists the JSON list,
    OVERWRITES failure_label with summarize_label(events).value, sets
    failure_label_source = 'opus_events_derived', and sets
    ultimately_succeeded from derive_ultimately_succeeded."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Pre-set a Sonnet judge value so we can confirm Opus overwrites it.
    conn.execute(
        """
        UPDATE traces
           SET failure_label = 'gave_up',
               failure_label_source = 'sonnet_judge'
         WHERE id = ?
        """,
        ("trace-under-test",),
    )
    client = _make_client(
        [
            _response(
                _events_tool_use_block(
                    [
                        {
                            "label": FailureLabel.LOOP.value,
                            "step_start": 1,
                            "step_end": 2,
                            "recovered": True,
                            "recovery_step": 2,
                            "recovery_summary": "Switched strategy and re-read the error.",
                        }
                    ]
                )
            )
        ]
    )

    result = persist_events_for_trace(conn, "trace-under-test", client=client)

    assert result is not None
    assert len(result.events) == 1

    row = _row(conn)
    persisted = json.loads(row["failure_events"])
    assert len(persisted) == 1
    assert persisted[0]["label"] == "loop"
    assert persisted[0]["recovered"] is True
    # All-recovered -> summary label is SUCCESS, overwrites Sonnet's "gave_up".
    assert row["failure_label"] == FailureLabel.SUCCESS.value
    assert row["failure_label_source"] == "opus_events_derived"
    # All-recovered -> ultimately_succeeded = 1.
    assert row["ultimately_succeeded"] == 1


def test_events_helper_bumps_has_error_when_unrecovered_event_present(
    conn, monkeypatch
) -> None:
    """The rule pass might miss a failure that Opus catches. Defensive
    write-back: if Opus sees an unrecovered event, has_error must be
    bumped to 1 so the recoveries-panel gate (has_error = 1) stays
    consistent with what Opus actually saw. Otherwise the panel would
    silently skip a failed trace just because the rule pass missed it."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    conn.execute("UPDATE traces SET has_error = 0 WHERE id = ?", ("trace-under-test",))
    client = _make_client(
        [
            _response(
                _events_tool_use_block(
                    [
                        {
                            "label": FailureLabel.GAVE_UP.value,
                            "step_start": 1,
                            "step_end": 2,
                            "recovered": False,
                            "recovery_step": None,
                            "recovery_summary": None,
                        }
                    ]
                )
            )
        ]
    )

    persist_events_for_trace(conn, "trace-under-test", client=client)

    row = _row(conn)
    assert row["has_error"] == 1
    assert row["ultimately_succeeded"] == 0


def test_events_helper_does_not_downgrade_has_error_from_1_to_0(
    conn, monkeypatch
) -> None:
    """Inverse of the bump-up case: if the rule pass set has_error=1
    but Opus reports only recovered events (or no events at all), we
    PRESERVE has_error=1. The rule pass is conservative + may have
    caught something Opus dismissed; preserving the higher-recall
    signal is safer than blindly downgrading."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    conn.execute("UPDATE traces SET has_error = 1 WHERE id = ?", ("trace-under-test",))
    client = _make_client(
        [
            _response(
                _events_tool_use_block(
                    [
                        {
                            "label": FailureLabel.LOOP.value,
                            "step_start": 1,
                            "step_end": 2,
                            "recovered": True,
                            "recovery_step": 2,
                            "recovery_summary": "Read the error and retried.",
                        }
                    ]
                )
            )
        ]
    )

    persist_events_for_trace(conn, "trace-under-test", client=client)

    row = _row(conn)
    # has_error stays 1 even though Opus saw only a recovered event.
    assert row["has_error"] == 1


def test_events_helper_returns_none_when_api_key_missing(conn) -> None:
    result = persist_events_for_trace(conn, "trace-under-test")
    assert result is None
    row = _row(conn)
    assert row["failure_events"] is None
    assert row["failure_label_source"] is None


def test_events_helper_returns_none_for_missing_trace(conn, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    sentinel_client = MagicMock()
    result = persist_events_for_trace(conn, "does-not-exist", client=sentinel_client)
    assert result is None
    sentinel_client.messages.create.assert_not_called()


def test_events_helper_returns_none_for_null_atif(conn, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    conn.execute("UPDATE traces SET atif = NULL WHERE id = ?", ("trace-under-test",))
    sentinel_client = MagicMock()
    result = persist_events_for_trace(conn, "trace-under-test", client=sentinel_client)
    assert result is None
    sentinel_client.messages.create.assert_not_called()


def test_events_helper_swallows_extraction_failures(conn, monkeypatch) -> None:
    """If Opus blows up (rate limit, network error, schema validation
    failure after retry), the helper must swallow it and leave the
    pre-existing row state intact. The upload pipeline still wants a
    successful upload + Sonnet label even if Opus failed."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Pre-set a Sonnet result so we can confirm it survives the
    # Opus failure -- the events helper must NOT clobber failure_label
    # when its own call fails.
    conn.execute(
        """
        UPDATE traces
           SET failure_label = 'loop',
               failure_label_reasoning = 'Sonnet said so.',
               failure_label_source = 'sonnet_judge'
         WHERE id = ?
        """,
        ("trace-under-test",),
    )
    failing_client = MagicMock()
    failing_client.messages.create.side_effect = RuntimeError("rate limit")

    result = persist_events_for_trace(conn, "trace-under-test", client=failing_client)
    assert result is None
    row = _row(conn)
    # Sonnet's prior result is intact -- the helper did not write
    # anything to the row when its own call failed.
    assert row["failure_label"] == "loop"
    assert row["failure_label_reasoning"] == "Sonnet said so."
    assert row["failure_label_source"] == "sonnet_judge"
    assert row["failure_events"] is None
