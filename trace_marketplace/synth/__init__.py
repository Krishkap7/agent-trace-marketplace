"""Sonnet-authored synthetic trace generation.

Slice 7 home for everything that turns "I want 500 diverse failure
traces" into actual files on disk under ``data/raw/<format>/`` that
the existing ingest scripts pick up unchanged.

The pipeline is intentionally narrow:

    TraceSpec  -->  prompt + tool schema  -->  Anthropic call
                                            |
                                            v
                                       raw JSONL/JSON
                                            |
                                            v
                                  validate via adapter
                                            |
                                       valid? write to disk

We DELIBERATELY do not own ingestion or DB writes -- those are the
existing ``scripts/ingest_*.py`` paths and they already work. This
package's contract ends when bytes land on disk.
"""
