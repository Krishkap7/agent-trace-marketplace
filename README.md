# Trace Marketplace

A failure-mode search engine for coding-agent traces. See [trace-marketplace-context.md](trace-marketplace-context.md) for the project brief and architectural decisions.

## Status

**Slice 2.5 (current):** SWE-agent HuggingFace adapter. Bulk-ingests a stratified 500-row sample (250 success / 250 failure) from `nebius/SWE-agent-trajectories` and wires the `has_error` column generically off `trajectory.extra["resolved"]`.

**Slice 2:** Claude Code JSONL → ATIF adapter, modular ingest pipeline (detect → adapt → redact → store) covering both native ATIF and Claude Code sessions.

**Slice 1:** ingest native ATIF traces from HuggingFace into a SQLite row, with `raw_blob` + validated `atif` columns populated.

### Why two corpora

The marketplace pulls from two distinct sources, each filling a different gap:

* **Real-user traces** (Claude Code JSONL, future Cursor exports): a long, messy, high-fidelity record of what an agent actually did on someone's laptop. Step counts in the thousands, model thinking blocks intact, no ground-truth labels. This is the *target* of the failure-mode search engine — what users will actually upload.
* **Benchmark traces** (ATIF MCP, SWE-agent): comparatively short, structured, and crucially carry a *labelled outcome* (success/failure against held-out tests). This is the *evaluation set* — slice 3's failure-detection rules and judges get calibrated against `has_error` here before being applied to the unlabelled real-user corpus.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
uv sync
```

This creates `.venv/` with Python 3.12 and installs `harbor` (ATIF Pydantic models), `huggingface-hub`, and dev tools (`pytest`, `ruff`).

## Run the pipeline

```bash
uv run python scripts/download_corpus.py        # pulls ATIF trajectory.json files into data/raw/
uv run python scripts/ingest_atif.py            # ingests 5 ATIF traces into data/marketplace.db
uv run python scripts/ingest_claude_code.py     # ingests *.jsonl from data/raw/claude_code/

# SWE-agent corpus (~85 MB parquet pull, ~30 s):
HF_HOME="$PWD/.cache/huggingface" uv run python scripts/download_swe_agent.py   # 250 success + 250 failure
uv run python scripts/ingest_swe_agent.py                                       # → 500 rows into the DB

uv run python scripts/verify.py                 # SELECT + per-format / per-has_error breakdown
uv run python scripts/breakdown.py              # success/failure split by source_format and model
uv run python scripts/inspect_trace.py <id>     # detailed view of one trace
```

End-to-end output should show 5 ATIF + N Claude Code + 500 SWE-agent rows, all with non-NULL `raw_blob` and `atif` columns. SWE-agent rows additionally carry `has_error ∈ {0, 1}` (250 each); the other formats stay `has_error=NULL` until slice 3 lands the failure-detection pass.

For Claude Code ingest, drop session JSONL files into `data/raw/claude_code/` (these come from `~/.claude/projects/<dir>/<uuid>.jsonl` on a real install). The directory is gitignored so test sessions never end up committed.

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
    cursor.py              # stub (see module docstring for v2 plan)
  ingest/
    detect.py              # cheap format sniffer
    pipeline.py            # detect → adapt → redact → insert (+ has_error)
  storage/
    db.py                  # SQLite schema + insert helper
scripts/
  download_corpus.py       # HuggingFace → data/raw/
  download_swe_agent.py    # nebius/SWE-agent-trajectories parquet → data/raw/swe_agent/
  ingest_atif.py           # data/raw/<agent>/trajectory.json → DB
  ingest_claude_code.py    # data/raw/claude_code/*.jsonl → DB
  ingest_swe_agent.py      # data/raw/swe_agent/*.json → DB
  verify.py                # SELECT + pretty-print + format/has_error breakdown
  breakdown.py             # success/failure stats by source_format and model
  inspect_trace.py         # one-trace detail view
  explore_swe_agent.py     # one-shot Step-0 inspection (uncommitted scratch)
tests/
  fixtures/
    claude_code_minimal.jsonl   # synthetic ~14-event Claude Code session
    swe_agent_sample.json       # synthetic SWE-agent HF row (10 traj entries)
  test_claude_code_adapter.py
  test_detect.py
  test_pipeline_has_error.py
  test_swe_agent_adapter.py
data/
  raw/                     # downloaded inputs (gitignored)
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

Adding a new format = drop a `BaseAdapter` subclass into `trace_marketplace/adapters/`, register it in `ADAPTERS`, extend `detect_format`. To populate `has_error`, set `trajectory.extra["resolved"] = bool(...)` in the adapter; the pipeline does the inversion (`has_error = int(not resolved)`) once, generically.
