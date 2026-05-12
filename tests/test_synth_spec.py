"""Tests for :mod:`trace_marketplace.synth.spec`.

Spec sampling is pure Python -- no mocks needed. We pin:

1. ``sample_specs(N, seed=S)`` is deterministic across runs.
2. Format distribution is even within +/-1.
3. Default Slice 7 demo distribution matches the documented ratios.
4. Custom ``label_distribution`` overrides the default cleanly.
5. Labels absent from the distribution NEVER get sampled.
6. Secondary failures never duplicate the primary AND only draw from
   labels present in the active distribution.
7. Legacy ``failure_ratio`` path still works (back-compat).
8. Edge cases: count=0, count<formats, invalid distribution.
"""

from __future__ import annotations

import pytest

from trace_marketplace.synth.spec import (
    ALL_FORMATS,
    DEFAULT_LABEL_DISTRIBUTION,
    NON_SUCCESS_LABELS,
    TraceSpec,
    sample_specs,
)


def test_sample_specs_is_deterministic_for_same_seed() -> None:
    a = sample_specs(50, seed=7)
    b = sample_specs(50, seed=7)
    assert a == b


def test_sample_specs_differs_across_seeds() -> None:
    a = sample_specs(50, seed=1)
    b = sample_specs(50, seed=2)
    assert a != b


def test_format_distribution_is_even_within_one() -> None:
    specs = sample_specs(500, seed=42)
    counts = {fmt: 0 for fmt in ALL_FORMATS}
    for s in specs:
        counts[s.output_format] += 1
    # 500 / 3 = 166.66 -> expect each bucket in {166, 167}.
    for fmt, c in counts.items():
        assert 165 <= c <= 168, f"{fmt}: {c}"


def test_default_distribution_matches_demo_spec() -> None:
    """Default Slice 7 demo: 30/5/15/10/10/10/10/10 across 8 labels."""
    specs = sample_specs(500, seed=42)
    counts: dict[str, int] = {}
    for s in specs:
        counts[s.primary_failure] = counts.get(s.primary_failure, 0) + 1
    # +/- 1 tolerance for rounding.
    assert counts["success"] == 150
    assert counts["partial_success"] == 25
    assert counts["loop"] == 75
    assert counts["gave_up"] == 50
    assert counts["hallucinated_api"] == 50
    assert counts["misread_error"] == 50
    assert counts["environment_issue"] == 50
    assert counts["wrong_approach"] == 50


def test_default_distribution_failure_ratio_is_65pct() -> None:
    """30% success + 5% partial -> 65% failures."""
    specs = sample_specs(500, seed=42)
    fail_count = sum(1 for s in specs if s.is_failure)
    assert 320 <= fail_count <= 330, fail_count


def test_labels_absent_from_distribution_never_appear() -> None:
    """tool_spamming / incomplete / unclear are NOT in the default."""
    specs = sample_specs(500, seed=42)
    seen = {s.primary_failure for s in specs}
    assert "tool_spamming" not in seen
    assert "incomplete" not in seen
    assert "unclear" not in seen


def test_custom_distribution_overrides_default() -> None:
    """Pass a pure-success distribution; every spec should be success."""
    specs = sample_specs(
        100,
        label_distribution={"success": 1.0},
        seed=42,
    )
    assert all(s.primary_failure == "success" for s in specs)
    assert all(not s.is_failure for s in specs)


def test_invalid_distribution_label_raises() -> None:
    with pytest.raises(ValueError, match="unknown labels"):
        sample_specs(
            10,
            label_distribution={"not_a_real_label": 1.0},
            seed=0,
        )


def test_distribution_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to ~1.0"):
        sample_specs(
            10,
            label_distribution={"success": 0.5, "loop": 0.2},
            seed=0,
        )


def test_legacy_failure_ratio_path_still_works() -> None:
    """Empty dict triggers the legacy even-split-across-9 path."""
    specs = sample_specs(
        200,
        label_distribution={},
        failure_ratio=0.75,
        seed=11,
    )
    fail_count = sum(1 for s in specs if s.is_failure)
    assert 148 <= fail_count <= 152, fail_count
    # All 9 non-success labels should appear under the legacy split.
    labels_seen = {s.primary_failure for s in specs if s.is_failure}
    for label in NON_SUCCESS_LABELS:
        assert label in labels_seen, label


def test_secondary_failures_never_duplicate_primary() -> None:
    specs = sample_specs(500, seed=42)
    for s in specs:
        assert s.primary_failure not in s.secondary_failures


def test_secondary_failures_only_use_active_labels() -> None:
    """Secondary failures must be drawn from the active distribution."""
    specs = sample_specs(500, seed=42)
    forbidden = {"tool_spamming", "incomplete", "unclear"}
    for s in specs:
        for sec in s.secondary_failures:
            assert sec not in forbidden


def test_secondary_failures_distribution_has_both_zero_and_multi() -> None:
    """Sanity: 50/30/20 split should produce zero-secondary AND >1-secondary."""
    specs = sample_specs(500, seed=42)
    failing = [s for s in specs if s.is_failure]
    zero = sum(1 for s in failing if len(s.secondary_failures) == 0)
    two = sum(1 for s in failing if len(s.secondary_failures) == 2)
    assert zero > 0
    assert two > 0


def test_distribution_with_single_failure_label_has_no_secondary() -> None:
    """Only one failure label active -> secondary pool is empty."""
    specs = sample_specs(
        50,
        label_distribution={"success": 0.5, "loop": 0.5},
        seed=42,
    )
    failing = [s for s in specs if s.is_failure]
    assert failing
    for s in failing:
        assert s.secondary_failures == ()


def test_default_label_distribution_sums_to_one() -> None:
    assert abs(sum(DEFAULT_LABEL_DISTRIBUTION.values()) - 1.0) < 1e-6


def test_traces_are_frozen() -> None:
    spec = sample_specs(1, seed=0)[0]
    with pytest.raises(Exception):
        spec.output_format = "other"  # type: ignore[misc]


def test_count_zero_returns_empty_list() -> None:
    assert sample_specs(0) == []


def test_invalid_failure_ratio_raises() -> None:
    """failure_ratio is only consulted on the legacy path (empty dict)."""
    with pytest.raises(ValueError):
        sample_specs(10, label_distribution={}, failure_ratio=1.5)
    with pytest.raises(ValueError):
        sample_specs(10, label_distribution={}, failure_ratio=-0.1)


def test_count_smaller_than_format_pool_still_balanced() -> None:
    """count=2 with 3 formats -> two distinct formats represented."""
    specs = sample_specs(2, seed=42)
    assert len({s.output_format for s in specs}) == 2


def test_target_step_count_is_threaded_through() -> None:
    specs = sample_specs(5, target_step_count=12, seed=42)
    for s in specs:
        assert s.target_step_count == 12


def test_success_traces_have_no_secondary_failures() -> None:
    specs = sample_specs(200, seed=42)
    for s in specs:
        if not s.is_failure:
            assert s.secondary_failures == ()


def test_dataclass_hashable_for_dedupe() -> None:
    spec = TraceSpec(
        output_format="codex",
        primary_failure="loop",
        task="x",
        target_step_count=5,
    )
    assert hash(spec) is not None
    s = {spec, spec}
    assert len(s) == 1
