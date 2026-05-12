# Trace Marketplace

A failure-mode search engine for coding-agent traces. See [trace-marketplace-context.md](trace-marketplace-context.md) for the project brief and architectural decisions.

## Status

**Slice 2 (current):** Claude Code JSONL → ATIF adapter, modular ingest pipeline (detect → adapt → redact → store) covering both native ATIF and Claude Code sessions.

**Slice 1:** ingest native ATIF traces from HuggingFace into a SQLite row, with `raw_blob` + validated `atif` columns populated.

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
uv run python scripts/verify.py                 # SELECT + pretty-print all rows
uv run python scripts/inspect_trace.py <id>     # detailed view of one trace
```

End-to-end output should show 5 ATIF rows + N Claude Code rows, all with non-NULL `raw_blob` and `atif` columns.

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
    cursor.py              # stub (see module docstring for v2 plan)
  ingest/
    detect.py              # cheap format sniffer
    pipeline.py            # detect → adapt → redact → insert
  storage/
    db.py                  # SQLite schema + insert helper
scripts/
  download_corpus.py       # HuggingFace → data/raw/
  ingest_atif.py           # data/raw/<agent>/trajectory.json → DB
  ingest_claude_code.py    # data/raw/claude_code/*.jsonl → DB
  verify.py                # SELECT + pretty-print
  inspect_trace.py         # one-trace detail view
tests/
  fixtures/
    claude_code_minimal.jsonl   # synthetic ~14-event Claude Code session
  test_claude_code_adapter.py
  test_detect.py
data/
  raw/                     # downloaded inputs (gitignored)
  marketplace.db           # SQLite (gitignored)
```

## Ingest pipeline

```
file path → detect_format → adapter.to_atif → redact (stub) → insert_trace
                            ↓
                       raw_blob is always stored,
                       even when adapter returns None
```

Adding a new format = drop a `BaseAdapter` subclass into `trace_marketplace/adapters/`, register it in `ADAPTERS`, extend `detect_format`.
