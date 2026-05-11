# Trace Marketplace

A failure-mode search engine for coding-agent traces. See [trace-marketplace-context.md](trace-marketplace-context.md) for the project brief and architectural decisions.

## Status

**Slice 1 (current):** ingest one native ATIF trace from HuggingFace into a SQLite row, with `raw_blob` + validated `atif` columns populated.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
uv sync
```

This creates `.venv/` with Python 3.12 and installs `harbor` (for the ATIF Pydantic models) and `huggingface-hub`.

## Run the pipeline

```bash
uv run python scripts/download_corpus.py   # pulls trajectory.json files from HF into data/raw/
uv run python scripts/ingest_atif.py       # validates + inserts 5 traces into data/marketplace.db
uv run python scripts/verify.py            # SELECT and print rows
```

End-to-end output should show 5 rows with non-NULL `raw_blob` and `atif` columns.

## Layout

```
trace_marketplace/
  storage/db.py            # SQLite schema + insert helper
scripts/
  download_corpus.py       # HuggingFace -> data/raw/
  ingest_atif.py           # data/raw/ -> data/marketplace.db
  verify.py                # SELECT + pretty-print
data/
  raw/                     # downloaded trajectory.json files (gitignored)
  marketplace.db           # SQLite, gitignored
```
