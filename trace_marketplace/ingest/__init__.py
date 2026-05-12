"""Ingest orchestration: detect format, dispatch to adapter, persist row."""

from trace_marketplace.ingest.detect import detect_format
from trace_marketplace.ingest.pipeline import IngestResult, ingest_file, redact

__all__ = ["IngestResult", "detect_format", "ingest_file", "redact"]
