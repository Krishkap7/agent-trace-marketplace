"""Unit tests for :mod:`trace_marketplace.ui.queries`.

These run against the ``populated_db`` fixture in ``conftest.py`` -- a
tmp SQLite DB seeded with 4 ingested fixtures + 2 hand-inserted rows so
we can pin every filter knob without monkey-patching.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from trace_marketplace.ui.queries import (
    DISTINCT_VALUE_COLUMNS,
    ListFilters,
    bounds,
    distinct_values,
    get_trace,
    list_traces,
    trace_counts,
)


# ---------- list_traces ----------


def test_list_traces_no_filters_returns_every_row(
    populated_db: sqlite3.Connection,
) -> None:
    rows, total = list_traces(populated_db)
    # 4 ingested fixtures + 2 hand-inserted rows.
    assert total == 6
    assert len(rows) == 6


def test_list_traces_filters_by_source_format(
    populated_db: sqlite3.Connection,
) -> None:
    rows, total = list_traces(populated_db, ListFilters(source_formats=("codex",)))
    assert total == 1
    assert rows[0]["source_format"] == "codex"


def test_list_traces_filter_multiple_source_formats(
    populated_db: sqlite3.Connection,
) -> None:
    rows, total = list_traces(
        populated_db,
        ListFilters(source_formats=("codex", "cursor")),
    )
    assert total == 2
    got = {row["source_format"] for row in rows}
    assert got == {"codex", "cursor"}


def test_list_traces_has_error_errors_only(
    populated_db: sqlite3.Connection,
) -> None:
    # Of the 6 fixtures, swe_agent is the only ground-truth-labelled
    # format. The shipped row has ``target=False`` (i.e. unresolved
    # / failure), so the "errors only" view should find exactly it.
    rows, total = list_traces(populated_db, ListFilters(has_error="errors"))
    assert total == 1
    assert rows[0]["has_error"] == 1
    assert rows[0]["source_format"] == "swe_agent"


def test_list_traces_has_error_successes_only(
    populated_db: sqlite3.Connection,
) -> None:
    rows, total = list_traces(populated_db, ListFilters(has_error="successes"))
    # Only the hand-inserted __test_success__ row carries has_error=0
    # in the seed mix.
    assert total == 1
    assert rows[0]["id"] == "__test_success__"
    assert rows[0]["has_error"] == 0


def test_list_traces_has_error_unknown_only(
    populated_db: sqlite3.Connection,
) -> None:
    rows, total = list_traces(populated_db, ListFilters(has_error="unknown"))
    # claude_code + codex + cursor (all leave has_error NULL) + the
    # __test_unknown__ hand-inserted row.
    assert total == 4
    for row in rows:
        assert row["has_error"] is None


def test_list_traces_has_error_rejects_garbage_value(
    populated_db: sqlite3.Connection,
) -> None:
    with pytest.raises(ValueError):
        list_traces(populated_db, ListFilters(has_error="lol"))


def test_list_traces_has_atif_only(populated_db: sqlite3.Connection) -> None:
    rows, total = list_traces(populated_db, ListFilters(has_atif=True))
    # All 4 ingested fixtures validate + the __test_success__ atif row.
    # The __test_unknown__ row has atif IS NULL and must be excluded.
    assert total == 5
    for row in rows:
        # ``atif`` isn't in LIST_COLUMNS, but every kept row must NOT be
        # the unknown one.
        assert row["id"] != "__test_unknown__"


def test_list_traces_has_atif_false_finds_only_null_atif(
    populated_db: sqlite3.Connection,
) -> None:
    rows, total = list_traces(populated_db, ListFilters(has_atif=False))
    assert total == 1
    assert rows[0]["id"] == "__test_unknown__"


def test_list_traces_search_substring_is_case_insensitive(
    populated_db: sqlite3.Connection,
) -> None:
    # The synthetic success row uses agent_name="test-agent". Search
    # with a different case to prove case-insensitivity.
    _, total = list_traces(populated_db, ListFilters(search="TEST-AGENT"))
    assert total >= 1


def test_list_traces_search_matches_id(populated_db: sqlite3.Connection) -> None:
    _, total = list_traces(populated_db, ListFilters(search="__test_success__"))
    assert total == 1


def test_list_traces_min_steps_filter_excludes_short_rows(
    populated_db: sqlite3.Connection,
) -> None:
    # The synthetic unknown row has num_steps=NULL and SQL NULL never
    # satisfies ``num_steps >= ?``; the synthetic success row has 2.
    _, total = list_traces(populated_db, ListFilters(min_steps=999))
    assert total == 0


def test_list_traces_pagination_offset_skips_first_row(
    populated_db: sqlite3.Connection,
) -> None:
    all_rows, _ = list_traces(populated_db, limit=10, offset=0)
    paged, _ = list_traces(populated_db, limit=10, offset=1)
    assert [r["id"] for r in paged] == [r["id"] for r in all_rows[1:]]


def test_list_traces_pagination_limit_caps_result_size(
    populated_db: sqlite3.Connection,
) -> None:
    rows, total = list_traces(populated_db, limit=2, offset=0)
    assert len(rows) == 2
    # ``total`` is the UN-PAGED count and must NOT shrink with LIMIT.
    assert total == 6


# ---------- get_trace ----------


def test_get_trace_returns_full_row_with_raw_blob_and_decodable_atif(
    populated_db: sqlite3.Connection,
) -> None:
    row = get_trace(populated_db, "__test_success__")
    assert row is not None
    assert row["raw_blob"] == '{"hello": "world"}'
    decoded = json.loads(row["atif"])
    assert decoded["schema_version"] == "ATIF-v1.7"


def test_get_trace_returns_none_for_unknown_id(
    populated_db: sqlite3.Connection,
) -> None:
    assert get_trace(populated_db, "does-not-exist") is None


def test_get_trace_with_null_atif_still_returns_raw_blob(
    populated_db: sqlite3.Connection,
) -> None:
    row = get_trace(populated_db, "__test_unknown__")
    assert row is not None
    assert row["atif"] is None
    assert row["raw_blob"] == "not actually a trace"


# ---------- distinct_values ----------


def test_distinct_values_returns_sorted_non_null_set(
    populated_db: sqlite3.Connection,
) -> None:
    formats = distinct_values(populated_db, "source_format")
    # We don't pin every value (depends on fixtures), but the well-known
    # ones must be present and the list must be sorted.
    assert formats == sorted(formats)
    assert "codex" in formats
    assert "cursor" in formats


def test_distinct_values_rejects_column_not_in_allowlist(
    populated_db: sqlite3.Connection,
) -> None:
    with pytest.raises(ValueError):
        distinct_values(populated_db, "raw_blob")


def test_distinct_value_columns_does_not_include_dangerous_columns() -> None:
    # Sanity: pure data assertion, no DB needed. Codifies the SQL-safety
    # invariant so a future contributor can't widen the allow-list without
    # also updating this test.
    assert "raw_blob" not in DISTINCT_VALUE_COLUMNS
    assert "atif" not in DISTINCT_VALUE_COLUMNS
    assert "id" not in DISTINCT_VALUE_COLUMNS


# ---------- trace_counts / bounds ----------


def test_trace_counts_aggregates_match_the_fixture(
    populated_db: sqlite3.Connection,
) -> None:
    counts = trace_counts(populated_db)
    assert counts["total"] == 6
    assert counts["with_atif"] == 5  # everything except __test_unknown__
    assert counts["errors"] == 1  # swe_agent fixture (target=False)
    assert counts["successes"] == 1  # __test_success__
    assert (
        counts["unknown_outcome"] == 4
    )  # claude_code + codex + cursor + __test_unknown__


def test_bounds_returns_min_and_max_num_steps(
    populated_db: sqlite3.Connection,
) -> None:
    b = bounds(populated_db)
    assert "min_steps" in b and "max_steps" in b
    assert b["min_steps"] <= b["max_steps"]
