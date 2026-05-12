# Cursor synthetic fixtures

These 10 `.cursor.json` files are **synthetic** sessions produced by
`scripts/generate_synthetic_cursor.py`. They are *not* extracted from a
real Cursor workspace SQLite DB or from `cursor-agent --output-format
stream-json` output.

## Why synthetic, why this format?

Cursor exposes two paths to get a chat out of the IDE:

1. The per-workspace SQLite DB (`state.vscdb`) — undocumented schema,
   shifts between IDE versions.
2. The official CLI `cursor-agent --print --output-format stream-json`
   — a documented NDJSON event stream.

For v1 we accept neither directly. Instead the adapter reads a small,
stable **project-defined JSON envelope** that callers can produce from
either source. Native ingest of the two canonical Cursor shapes is a
deliberate v2 feature so we can ship the failure-mode search loop
first. See `trace_marketplace/adapters/cursor.py` for the full envelope
spec.

Every file is tagged `"synthetic": true` at the envelope top level, and
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

The 3 failure files each carry a `synthetic_failure_mode` field at the
envelope top level naming the *kind* of failure. The adapter propagates
this into `trajectory.extra.synthetic_failure_mode` so failure-mode
search can be evaluated against ground truth.

| File                                  | `synthetic_failure_mode` | What goes wrong                                                          |
| ------------------------------------- | ------------------------ | ------------------------------------------------------------------------ |
| `cursor_synth_05_failure.cursor.json` | `broke_working_code`     | "Fix" replaces multi-format parser with `fromisoformat`; 2 tests regress |
| `cursor_synth_06_failure.cursor.json` | `wrong_file_edited`      | Patches deprecated `src/legacy/billing_v1.py`, ignores live `src/billing.py:42` |
| `cursor_synth_07_failure.cursor.json` | `premature_conclusion`   | Bumps a flaky-test sleep, declares done without re-running tests         |

## Regenerate

```bash
uv run python scripts/generate_synthetic_cursor.py            # all 10
uv run python scripts/generate_synthetic_cursor.py --only 5   # just file #5
```

The RNG seed is fixed (`SEED = 20260511`) so regeneration is a no-op in
git unless `SCENARIOS` is edited.
