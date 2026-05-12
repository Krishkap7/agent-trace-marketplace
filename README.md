# Trace Marketplace

A failure-mode search engine for coding-agent traces. See [trace-marketplace-context.md](trace-marketplace-context.md) for the project brief and architectural decisions.

## Status

**Slice 3 (current):** Streamlit UI viewer. Filterable, paginated list view + URL-linkable detail view that renders the ATIF as a readable conversation thread (italic reasoning, encrypted-reasoning badge, collapsible tool calls / observations). Falls back to raw blob for `atif IS NULL` traces. SQL helpers live in `trace_marketplace/ui/queries.py` and are unit-tested without Streamlit.

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
  ui/
    __init__.py
    app.py                 # 30-line router; reads st.query_params["trace_id"]
    queries.py             # parameterised SQL helpers; no Streamlit import
    components.py          # render_step / render_header / render_filters_summary
    list_view.py           # sidebar filters + st.dataframe + manual pagination
    detail_view.py         # header + conversation thread + raw-blob fallback
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
