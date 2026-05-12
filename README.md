# Trace Marketplace

A failure-mode search engine for coding-agent traces. See [trace-marketplace-context.md](trace-marketplace-context.md) for the project brief and architectural decisions.

## Status

**Slice 5 (current):** Browser uploads. The Streamlit UI gained an upload page (`?page=upload`) that drag-drops or pastes a trace through the existing `ingest_file()` pipeline + the rule-based failure pass, so a freshly uploaded trace lands in the list view with `has_error` and `failure_signals` populated within a single page render. LLM-judge stays a batch job, deliberately -- per-upload Sonnet calls would be slow and unbounded; fresh uploads pick up a `failure_label` on the next `scripts/judge_failures.py` run via the existing `WHERE failure_label IS NULL` gate. Per-file isolation means one bad payload in a multi-file submission can't tank the rest.

**Slice 4:** Failure-detection enrichment. Deterministic rule pass (`scripts/detect_failures.py`) populates a new `failure_signals` JSON column and fills `has_error` for unlabelled traces; `scripts/validate_detection.py` benchmarks the rules against SWE-agent ground truth (precision ~0.78, recall ~0.48 aggregate, full table in [data/validation_report.md](data/validation_report.md)). LLM judge (`scripts/judge_failures.py`, Sonnet 4.6) classifies each flagged + sampled trace into an 11-way `FailureLabel` enum plus a rich shop-window paragraph of reasoning. Both passes are idempotent.

**Slice 3:** Streamlit UI viewer. Filterable, paginated list view + URL-linkable detail view that renders the ATIF as a readable conversation thread (italic reasoning, encrypted-reasoning badge, collapsible tool calls / observations). Falls back to raw blob for `atif IS NULL` traces. SQL helpers live in `trace_marketplace/ui/queries.py` and are unit-tested without Streamlit.

**Slice 2.6:** Codex (OpenAI `~/.codex/sessions/*.jsonl` rollouts) and Cursor (project-defined v1 export envelope) adapters. Both ship with deterministic synthetic-fixture generators (4 success / 3 failure / 2 short / 1 long per format) plus 6 named ground-truth failure modes (`hallucinated_api`, `infinite_retry_loop`, `ignored_test_failure`, `broke_working_code`, `wrong_file_edited`, `premature_conclusion`) propagated into `trajectory.extra.synthetic_failure_mode` for downstream search evaluation.

**Slice 2.5:** SWE-agent HuggingFace adapter. Bulk-ingests a stratified 500-row sample (250 success / 250 failure) from `nebius/SWE-agent-trajectories` and wires the `has_error` column generically off `trajectory.extra["resolved"]`.

**Slice 2:** Claude Code JSONL → ATIF adapter, modular ingest pipeline (detect → adapt → redact → store) covering both native ATIF and Claude Code sessions.

**Slice 1:** ingest native ATIF traces from HuggingFace into a SQLite row, with `raw_blob` + validated `atif` columns populated.

### Why two corpora

The marketplace pulls from two distinct sources, each filling a different gap:

* **Real-user traces** (Claude Code JSONL, Cursor v1 envelope, Codex rollout JSONL): a long, messy, high-fidelity record of what an agent actually did on someone's laptop. Step counts in the thousands, model thinking blocks intact, no ground-truth labels. This is the *target* of the failure-mode search engine — what users will actually upload.
* **Benchmark traces** (ATIF MCP, SWE-agent): comparatively short, structured, and crucially carry a *labelled outcome* (success/failure against held-out tests). This is the *evaluation set* — slice 3's failure-detection rules and judges get calibrated against `has_error` here before being applied to the unlabelled real-user corpus.
* **Synthetic failure-mode fixtures** (Codex / Cursor `_failure_*` files): a small set of deterministically generated traces with `trajectory.extra.synthetic_failure_mode` tagging the *kind* of failure (hallucinated API, ignored test failure, edited the wrong file, …). Bridges the gap between unlabelled real-user data and binary success/failure benchmark labels — gives the search engine ground-truth named-mode targets to evaluate against.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
uv sync
```

This creates `.venv/` with Python 3.12 and installs `harbor` (ATIF Pydantic models), `huggingface-hub`, and dev tools (`pytest`, `ruff`).

## Run the pipeline

```bash
uv run python scripts/download_corpus.py        # pulls ATIF trajectory.json files into data/raw/
uv run python scripts/ingest_atif.py            # 5 ATIF traces      → data/marketplace.db
uv run python scripts/ingest_claude_code.py     # *.jsonl from data/raw/claude_code/

# SWE-agent corpus (~85 MB parquet pull, ~30 s):
HF_HOME="$PWD/.cache/huggingface" uv run python scripts/download_swe_agent.py   # 250 success + 250 failure
uv run python scripts/ingest_swe_agent.py                                       # → 500 rows into the DB

# Codex + Cursor (synthetic fixtures, ~0.5 s):
uv run python scripts/generate_synthetic_codex.py     # 10 *.jsonl into data/raw/codex/
uv run python scripts/generate_synthetic_cursor.py    # 10 *.cursor.json into data/raw/cursor/
uv run python scripts/ingest_codex.py                 # 10 codex rows
uv run python scripts/ingest_cursor.py                # 10 cursor rows

uv run python scripts/verify.py                 # SELECT + per-format / per-has_error breakdown
uv run python scripts/breakdown.py              # success/failure + synthetic failure-mode counts
uv run python scripts/inspect_trace.py <id>     # detailed view of one trace
```

End-to-end output should show 5 ATIF + 2 Claude Code + 500 SWE-agent + 10 Codex + 10 Cursor = **527 rows**, all with non-NULL `raw_blob` and `atif` columns. SWE-agent rows additionally carry `has_error ∈ {0, 1}` (250 each); the other formats stay `has_error=NULL` until slice 3 lands the failure-detection pass. The 3 `*_failure_*.jsonl` Codex files and 3 `cursor_synth_*_failure.cursor.json` files carry a `synthetic_failure_mode` ground-truth label inside `atif.extra` (queryable via `json_extract(atif, '$.extra.synthetic_failure_mode')`).

For Claude Code ingest, drop session JSONL files into `data/raw/claude_code/` (these come from `~/.claude/projects/<dir>/<uuid>.jsonl` on a real install). Codex sessions live under `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`; point `scripts/ingest_codex.py --raw-dir` at that directory to ingest real Codex rollouts. Cursor v1 envelope files use the `*.cursor.json` extension and follow the spec in `trace_marketplace/adapters/cursor.py`. All three real-data directories are gitignored.

## Enrich failures (slice 4)

Two passes turn the raw `traces` table into a queryable failure-mode index:

```bash
# Pass 1: deterministic rule-based signals (free, ~1 s for 500+ traces).
uv run python scripts/detect_failures.py            # writes failure_signals
uv run python scripts/detect_failures.py --force    # recompute after threshold tweaks

# Validate against SWE-agent ground truth (writes data/validation_report.md):
uv run python scripts/validate_detection.py

# Pass 2: LLM judge (Sonnet 4.6, ~$5 for ~280 calls).
export ANTHROPIC_API_KEY=sk-ant-...
uv run python scripts/judge_failures.py --dry-run   # ALWAYS check expected count + cost first
uv run python scripts/judge_failures.py             # real run; <$10 hard cap built in
```

`detect_failures.py` populates a new `failure_signals` JSON column (`{"loop": true, "ended_on_error": false, ...}`) and fills `has_error` from the aggregate "any signal fired" only when it was previously NULL (`COALESCE` preserves SWE-agent's `target`-derived ground truth).

`judge_failures.py` selects every `has_error = 1` row plus a deterministic 10% sample of `has_error = 0` rows (seed 42), calls Sonnet 4.6 via the Anthropic SDK's structured tool-use feature so the JSON response is enum-validated, and writes the chosen `FailureLabel` + a one-sentence reasoning. The script is idempotent (skips traces that already have `failure_label`), respects a `--max-spend-usd` hard cap, and refuses to start without `ANTHROPIC_API_KEY` in the environment.

The 11 failure labels live in [`trace_marketplace/enrich/failure_taxonomy.py`](trace_marketplace/enrich/failure_taxonomy.py); add a label there and its description gets fed verbatim into the next judge prompt with no other changes.

## View the marketplace

After ingest, launch the Streamlit viewer:

```bash
bash scripts/run_ui.sh
# or equivalently:
uv run streamlit run trace_marketplace/ui/app.py
```

Opens `http://localhost:8501` on the list view. Sidebar filters narrow by `source_format`, `agent_name`, `model`, three-way `has_error` (errors / successes / unknown), `num_steps` range, and an `atif IS NULL` toggle for the raw-blob fallback path. The search box does case-insensitive substring matches across `id` / `agent_name` / `model`.

Click any row to navigate to `?trace_id=<id>` — the URL is bookmarkable. The detail view renders the ATIF as a conversation thread (user / agent / system / tool sources visually distinguished), with tool calls and observations collapsed by default. Encrypted reasoning placeholders (`[encrypted reasoning, N bytes]`) show a lock badge so the slice 2.7 fix is visible in the UI. Raw blob + normalised ATIF are always available behind expanders for inspection. `atif IS NULL` traces (unknown formats / adapter failures) fall back to the raw blob automatically.

![List view](docs/screenshots/list_view.png)

![Detail view (ATIF)](docs/screenshots/detail_view.png)

> Screenshots regenerated after each UI change — drop replacements into `docs/screenshots/`.

## Upload from the UI (slice 5)

Click the **Upload trace ->** link in the top-right of the list view (or navigate directly to `?page=upload`). The page accepts two flows:

* **Drag-drop** -- multi-file. Each file is staged into its own tempdir, fed through `ingest_file()`, then the rule-based failure pass runs inline so the result block can report which signals fired. Per-file try/except means one bad upload doesn't abort the rest.
* **Paste** -- single-shot textarea for ad-hoc demos. Same pipeline, same idempotency guarantees.

Format detection is content-based, so the file extension doesn't have to match the format (a Claude Code session saved as `.txt` still works). Unknown formats land as raw blobs -- they remain searchable via the list-view text search, they just lack structured filters until an adapter for that shape is added.

`failure_label` (the Sonnet 4.6 taxonomy classification) is **deliberately not** populated on upload -- the synchronous LLM call would add 5-30 s of user-facing latency and uncapped cost. Fresh uploads land with `failure_label = NULL`; the next `scripts/judge_failures.py` run picks them up via its existing gate.

![Upload page](docs/screenshots/upload_view.png)

> Drop a replacement screenshot into `docs/screenshots/upload_view.png` after each UI change.

## Tests

```bash
uv run pytest
```

Adapter tests use synthetic fixtures in `tests/fixtures/` — no real session content lives in the repo.

## Layout

```
trace_marketplace/
  adapters/
    base.py                # BaseAdapter ABC
    atif.py                # passthrough + v1.2→v1.7 schema normalization
    claude_code.py         # JSONL → ATIF (walks message.content blocks)
    swe_agent.py           # SWE-agent HF row → ATIF (system_prompt + text)
    codex.py               # ~/.codex/sessions/*.jsonl rollouts → ATIF
    cursor.py              # Cursor v1 export envelope → ATIF
  ingest/
    detect.py              # cheap format sniffer (5 formats)
    pipeline.py            # detect → adapt → redact → insert (+ has_error)
  storage/
    db.py                  # SQLite schema + insert helper
trace_marketplace/
  enrich/
    __init__.py
    failure_taxonomy.py    # FailureLabel enum + descriptions (judge prompt source)
    failure_rules.py       # 5 pure signal_* fns + compute_signals aggregator
    failure_judge.py       # Anthropic SDK wrapper; tool-use structured output
  ui/
    __init__.py
    app.py                 # 3-route router (page=upload / trace_id / list)
    queries.py             # parameterised SQL helpers; no Streamlit import
    components.py          # render_step / render_header / render_filters_summary
    list_view.py           # sidebar filters + st.dataframe + manual pagination
    detail_view.py         # header + conversation thread + raw-blob fallback
    upload_view.py         # slice 5: drag-drop + paste -> ingest_file -> inline rule pass
scripts/
  download_corpus.py            # ATIF MCP traces → data/raw/<agent>/
  download_swe_agent.py         # nebius/SWE-agent-trajectories parquet → data/raw/swe_agent/
  generate_synthetic_codex.py   # 10 synthetic Codex rollouts → data/raw/codex/
  generate_synthetic_cursor.py  # 10 synthetic Cursor envelopes → data/raw/cursor/
  ingest_atif.py                # data/raw/<agent>/trajectory.json → DB
  ingest_claude_code.py         # data/raw/claude_code/*.jsonl → DB
  ingest_swe_agent.py           # data/raw/swe_agent/*.json → DB
  ingest_codex.py               # data/raw/codex/*.jsonl → DB (recursive)
  ingest_cursor.py              # data/raw/cursor/*.cursor.json → DB
  verify.py                     # SELECT + pretty-print + format/has_error breakdown
  breakdown.py                  # success/failure + synthetic failure-mode counts
  inspect_trace.py              # one-trace detail view
  detect_failures.py            # slice 4: rule-based failure signal pass
  validate_detection.py         # slice 4: precision/recall vs SWE-agent ground truth
  judge_failures.py             # slice 4: LLM-judge pass (Sonnet 4.6)
  run_ui.sh                     # bash wrapper around `uv run streamlit run`
tests/
  fixtures/
    claude_code_minimal.jsonl   # synthetic ~14-event Claude Code session
    swe_agent_sample.json       # synthetic SWE-agent HF row
    codex_sample.jsonl          # synthetic 13-line Codex rollout
    cursor_sample.cursor.json   # synthetic 6-message Cursor envelope
  test_claude_code_adapter.py
  test_codex_adapter.py
  test_cursor_adapter.py
  test_detect.py
  test_pipeline_has_error.py
  test_swe_agent_adapter.py
  test_upload_integration.py    # slice 5: ingest + rule pass on a Claude Code JSONL fixture
data/
  raw/
    codex/                 # synthetic rollouts (committed; see data/raw/codex/README.md)
    cursor/                # synthetic envelopes (committed; see data/raw/cursor/README.md)
    swe_agent/             # downloaded HF rows (gitignored)
    claude_code/           # user-supplied Claude sessions (gitignored)
  marketplace.db           # SQLite (gitignored)
```

## Ingest pipeline

```
file path → detect_format → adapter.to_atif → redact (stub) → insert_trace
                            ↓                                      ↑
                       raw_blob always stored          has_error derived from
                       even when adapter returns       trajectory.extra["resolved"]
                       None
```

Adding a new format = drop a `BaseAdapter` subclass into `trace_marketplace/adapters/`, register it in `ADAPTERS`, extend `detect_format`. To populate `has_error`, set `trajectory.extra["resolved"] = bool(...)` in the adapter; the pipeline does the inversion (`has_error = int(not resolved)`) once, generically. To carry a named failure-mode ground-truth label, set `trajectory.extra["synthetic_failure_mode"] = "..."` (or have the adapter pick up a `synthetic_failure_mode` field from the raw input, as Codex/Cursor do).
