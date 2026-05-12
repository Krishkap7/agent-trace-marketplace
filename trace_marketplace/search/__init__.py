"""Search primitives for the trace marketplace.

This package is the slice 6 home for everything that lets a user
*find* traces: text extraction (turning an ATIF dict into a blob
suitable for embedding), vector NN over those embeddings, and the
NL-query parser that converts a free-form English question into a
structured filter + semantic intent.

The actual embedding generation lives in
``trace_marketplace.enrich.embeddings`` because it's a write-side
enrichment step (same conceptual neighbourhood as
``failure_rules``/``failure_judge``); this package is read-side only.
"""
