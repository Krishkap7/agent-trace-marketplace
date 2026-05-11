# Trace Marketplace — Project Context

> Context dump from planning conversation. Drop this in your Cursor project root (or in `.cursor/rules/` as a `.mdc` file) so the AI has full context for every chat.

---

## The brief

Open-ended Fleet AI work trial: **"Trace Marketplace."**

People generate agent traces (Claude Code, Cursor, Codex, etc.) while their AIs do work for them. Labs and businesses would pay for:
- Rare data / rare experiences
- Failure modes — where the AI is screwing up

The brief explicitly says marketplace functionality is **nice-to-have**. They mostly care about **design and systems thinking** around: **ingest → search → process/surface interesting things**.

## Reframed (after clarifying with reviewer)

The real product is **a failure-mode search engine for coding-agent traces**:

- **Sellers**: users running Claude Code / Cursor whose agents fail in interesting ways
- **Buyers**: labs training coding agents who want to find "traces like this failure we're debugging" or "traces where the agent hit error type X"
- **Killer use case**: *"I have a failure in my agent, find me similar failures in the corpus."*

Marketplace mechanics (pricing, payments, bounties) are out of scope for the prototype — covered in the design doc only.

## Constraints

- **Time**: 2 days, hackathon-style, side project (independent of fleet-sdk/harbor)
- **Deliverable**: working prototype primary, doc secondary
- **LLM access**: paid keys available (use sparingly — see economics decisions below)
- **Reviewer**: technical, wants to see scoping decisions and systems thinking, not feature count

---

## Key architectural decisions

### 1. Canonical format: ATIF (Agent Trajectory Interchange Format)

**Don't reinvent the schema.** ATIF already exists — open spec from the Harbor team (Terminal-Bench creators).

- **Spec**: https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md
- **Docs**: https://www.harborframework.com/docs/agents/trajectory-format
- **Reference impl**: `harbor` Python library has Pydantic models in `harbor.models.trajectories` and a validator in `harbor.utils.trajectory_validator`

ATIF structure (roughly):
```
Trajectory:
  schema_version: "ATIF-v1.4"
  session_id: str
  agent: Agent(name, version, model_name, extra)
  steps: list[Step]
  final_metrics: FinalMetrics
  extra: dict  # escape hatch for source-specific stuff

Step:
  step_id: int (sequential from 1)
  timestamp: ISO 8601
  source: "user" | "agent" | "system"
  message: str
  reasoning_content: str (optional, agent only)
  tool_calls: list[ToolCall]
  observation: Observation
  metrics: Metrics
```

Agents that already emit ATIF (via Harbor): Terminus-2, OpenHands, Mini-SWE-Agent, Gemini CLI, Claude Code, Codex.
**Cursor is NOT on that list** — we'd write our own adapter.

### 2. Two-layer storage (medallion / bronze-silver-gold)

```
traces table:
  id              UUID
  source_format   TEXT     -- "atif" | "claude_code" | "swe_agent" | "unknown"
  raw_blob        JSON     -- original upload, untouched (bronze)
  atif            JSON     -- best-effort canonical version (silver)
  ingested_at     TIMESTAMP
  -- derived/extracted columns for search (gold):
  agent_name      TEXT
  model           TEXT
  num_steps       INT
  num_tool_calls  INT
  has_error       BOOL
  failure_label   TEXT
  embedding_id    REF
```

**Why both raw + ATIF**: ATIF normalization is lossy by design (reviewer explicitly said "if you can't fully recover all information that may still be okay"). Raw blob is the source of truth for re-processing; ATIF is the query surface. Standard pattern — Snowflake, Databricks, Phoenix, LangSmith all do this.

### 3. Adapter pattern with format auto-detection

User uploads whatever they have. System sniffs format, picks adapter, converts. User never has to think about format.

```python
# Pseudocode
def ingest(path):
    fmt = detect_format(path)            # sniff file extension, top-level keys, etc.
    adapter = ADAPTERS[fmt]
    raw = read_raw(path)
    atif = adapter.to_atif(raw)          # may be sparse if fmt == "unknown"
    store(raw_blob=raw, atif=atif, source_format=fmt)
```

**Detection signatures**:
- ATIF: `schema_version` starts with "ATIF" at root
- Claude Code: JSONL with `sessionId` + `type` ∈ {user, assistant, tool_use, tool_result}
- SWE-agent: known JSON shape from nebius dataset
- Cursor: SQLite with composer/chat table names
- Unknown: none of the above

### 4. Unknown-format handling: store raw, leave ATIF sparse, no LLM

**Decision: no LLM in the conversion path for the prototype.**

For unknown formats:
- Store raw blob (always)
- Produce minimal/empty ATIF (with whatever fields can be safely inferred)
- Trace is still searchable via full-text search on raw blob and embedding-based search on raw text
- Structured search (filter by tool name, count steps) doesn't work for these traces until someone writes an adapter

This is the cheap-and-honest answer: nothing is rejected, nothing costs LLM dollars per trace, and the user can still find their data in the system.

**Long-term (in design doc, not built)**: Pattern 2 — LLM synthesizes a deterministic Python adapter once per unknown format, caches it, reuses for all future traces of that format. Pays LLM cost once-per-format, not once-per-trace. Requires sandboxing + validation + format fingerprinting — too much for 2 days.

**Explicitly rejected**: Pattern 1 — LLM converting each trace at runtime. Per-trace LLM cost is a trap at marketplace scale.

### 5. Three search modes, one canonical source

```
Raw upload
    ↓
[Adapter] ──► ATIF (lossy if needed, but always produced — possibly sparse)
    ↓
    ├──► Structured store (Postgres/SQLite columns)   — filter-style search
    ├──► Embedding store (vector index over text)     — "find similar" search
    └──► Full-text index (FTS5 over raw + ATIF text)  — keyword/grep search
```

All three indexes derived from ATIF (FTS also indexes raw blob for safety). Search API orchestrates them into ranked results.

---

## LLM usage policy

**Deterministic code for structural mapping. LLM for semantic judgment.**

| Task | Tool | Why |
|------|------|-----|
| Format detection | Code | Cheap, fast, reliable |
| Field-to-field conversion (known formats) | Code | Deterministic, testable, free |
| Canonical tool-name resolution (`Bash` → `run_command`) | Code with cached LLM disambiguation for novel tool names | Mostly rule-based; LLM only for unseen tools, cached |
| Failure-mode classification | LLM-as-judge | Inherently semantic, can't be rules-based; sample don't full-scan |
| "Find similar failures" search | Embeddings | Standard semantic similarity |
| Adapter synthesis for unknown formats | LLM (one-shot, cached) | Pattern 2 — design only, not built |

Approximate budget: LLM calls only for (a) failure classification on a sample of traces, (b) maybe novel tool-name resolution. Everything else is deterministic Python.

---

## Corpus plan

**Primary**: `nebius/SWE-agent-trajectories` on HuggingFace — 80k SWE-agent trajectories solving GitHub issues, rich metadata, real failures pre-labeled with exit rates and correctness. Sample ~500–1000 for the demo.

Why this corpus does double duty:
- Real failure signal for failure-detection and search to find
- Forces writing a real non-ATIF → ATIF adapter (the systems-thinking work)

**Secondary**: `obaydata/mcp-agent-trajectory-benchmark` — 49 native ATIF v1.2 trajectories. Proves passthrough handles real ATIF.

**Optional**: Generate ~10–20 fresh traces with Harbor + Anthropic key on a SWE-bench subset, if time on day 1.

---

## What to build vs. stub

### Build (deterministic, no LLM at ingest):
- ATIF passthrough adapter (validates + stores)
- SWE-agent → ATIF adapter (real conversion work)
- Format auto-detection
- SQLite storage with raw_blob + atif columns + extracted derived columns
- Failure detection layer (cheap rule signals + LLM judge on sample)
- Embeddings + vector index
- Search API (structured + semantic + FTS)
- Minimal UI (Streamlit is fine) — paste a trace, see ranked similar failures

### Stub (interface only, raises NotImplementedError):
- Cursor adapter — proves the abstraction extends
- LLM-synthesized adapter (Pattern 2) — design-doc-only

---

## Failure-mode taxonomy (preliminary, refine during build)

**Cheap structural signals** (rule-based):
- Hit max turns / context limit
- Repeated identical tool call N times (loop detection)
- Ended on tool error with no recovery attempt
- Agent abandoned task (final message expresses giving up)
- Non-zero exit / failed verifier (if available from corpus)

**Semantic labels** (LLM-as-judge on suspicious traces):
- Misread error message
- Wrong API/library assumption
- Hallucinated function or file
- Got stuck on environment setup
- Correct approach but wrong implementation detail
- Gave up on hard test / hard subproblem
- ...etc, can grow with corpus

**Design call**: hybrid taxonomy — small set of structural types detected by code, plus embedding-based clustering for the long tail. Buyers can search by named type OR by example.

---

## Project structure

```
trace_marketplace/
  adapters/        # one file per source format
    base.py        # BaseAdapter abstract class
    atif.py        # passthrough
    swe_agent.py   # real conversion
    cursor.py      # stub
    llm_synth.py   # stub (Pattern 2)
  models/          # re-exports harbor.models.trajectories
  ingest/
    detect.py      # format sniffer
    pipeline.py    # detect → adapt → store
  storage/
    db.py          # SQLite + schema
    blob.py        # raw blob storage
  enrich/
    failure_rules.py    # cheap structural signals
    failure_judge.py    # LLM-as-judge wrapper
    embeddings.py
  search/
    structured.py
    semantic.py
    fts.py
    api.py         # combines all three
  ui/
    app.py         # streamlit
tests/
data/
  raw/             # uploaded files, untouched
  processed/       # normalized ATIF
pyproject.toml
README.md
DESIGN.md          # the design doc deliverable
```

---

## Day-by-day (rough)

### Day 1
- AM: Project scaffold, install harbor lib, ATIF passthrough, format detection skeleton, SQLite schema
- PM: SWE-agent adapter, ingest ~500-1000 sampled traces, validate
- EOD: Working ingest pipeline end-to-end on real data

### Day 2
- AM: Failure detection — rule-based signals first, then LLM judge on a sample
- Midday: Embeddings + vector index, search API
- PM: Streamlit UI — paste trace, see similar failures + structured filters
- EOD: Polish, write DESIGN.md, push

### If time on day 2 evening (polish only):
- Live Harbor-generated trace demo
- OR: small working demo of Pattern 2 (LLM-synthesized adapter) on one specific format

---

## What "good" looks like for the reviewer

The reviewer is grading **scoping + systems thinking**, not feature count. The wins are:

- Chose ATIF as canonical (didn't reinvent the wheel) — articulated *why*
- Raw + ATIF dual storage (data engineering pattern, not novel) — articulated *why*
- Two real adapters + one stub (proves the abstraction without overcommitting)
- LLM used only where semantically necessary, not for cheap-code-replaceable work
- Pattern 2 designed for unknown formats, not built, with explicit reasoning about per-format vs. per-trace LLM economics
- Failure detection hybrid (structural + semantic) acknowledges the long-tail problem
- Killer demo: paste a failing trace, get back ranked similar failures

What the reviewer explicitly said:
- "Make sense to try to convert to it [ATIF] for certain purposes, and if you can't fully recover all information that may still be okay" — green light to be lossy and pragmatic.

---

## Reference links

- ATIF spec: https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md
- ATIF docs: https://www.harborframework.com/docs/agents/trajectory-format
- Harbor: https://github.com/harbor-framework/harbor
- SWE-agent trajectories dataset: https://huggingface.co/datasets/nebius/SWE-agent-trajectories
- MCP ATIF dataset: https://huggingface.co/datasets/obaydata/mcp-agent-trajectory-benchmark
- Phoenix ATIF ingestion reference: https://arize.com/docs/phoenix/release-notes/04-2026/04-03-2026-atif-trajectory-upload
