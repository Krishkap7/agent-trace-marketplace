# Codex synthetic fixtures

These 10 `.jsonl` files are **synthetic** rollouts produced by
`scripts/generate_synthetic_codex.py`, not captured from a real
`~/.codex/sessions/<date>/rollout-*.jsonl`.

## Why synthetic?

Real Codex sessions carry personal cwd paths, real API spend, and
project-specific content. For an open-source prototype we generate a
small, varied set of rollouts that exactly match the wire format
documented in [`codex-rs/protocol/src/protocol.rs`](https://github.com/openai/codex/blob/main/codex-rs/protocol/src/protocol.rs)
(the `RolloutItem` enum) but contain no real-world data.

Every file is tagged `"synthetic": true` on the `session_meta` line, and
the adapter propagates this into `trajectory.extra.synthetic` so
downstream tooling can filter synthetic rows out of "real" corpora.

## Variety target

Ten files covering the 4 dimensions called out in the slice 2.6 spec:

| Count | Kind      | Filename suffix |
| ----: | --------- | --------------- |
|     4 | success   | `_success`      |
|     3 | failure   | `_failure`      |
|     2 | short     | `_short`        |
|     1 | long      | `_long`         |

## Failure-mode ground truth

The 3 failure files each carry a `synthetic_failure_mode` field (sibling
of `synthetic` on `session_meta`) naming the *kind* of failure. The
adapter propagates this into `trajectory.extra.synthetic_failure_mode`
so failure-mode search can be evaluated against ground truth.

| File                              | `synthetic_failure_mode` | What goes wrong                                                |
| --------------------------------- | ------------------------ | -------------------------------------------------------------- |
| `codex_synth_05_failure.jsonl`    | `hallucinated_api`       | Calls `requests.fetch_json` (doesn't exist), ignores AttributeError |
| `codex_synth_06_failure.jsonl`    | `infinite_retry_loop`    | Re-runs the same broken `pytest -q` 4 times without varying    |
| `codex_synth_07_failure.jsonl`    | `ignored_test_failure`   | pytest shows `1 failed`, agent declares "all tests pass"       |

## Regenerate

```bash
uv run python scripts/generate_synthetic_codex.py            # all 10
uv run python scripts/generate_synthetic_codex.py --only 5   # just file #5
```

The RNG seed is fixed (`SEED = 20260511`) so regeneration is a no-op in
git unless `SCENARIOS` is edited.
