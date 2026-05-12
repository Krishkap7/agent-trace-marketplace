"""ATIF passthrough adapter.

For files whose content is already in ATIF format, the adapter's job is to
JSON-parse, normalise minor schema drift, and validate against harbor's
Pydantic models. Extracted from slice 1's ``scripts/ingest_atif.py``.

The one non-trivial bit is :func:`_normalize_step_metrics`, which handles
v1.2 corpora (like ``obaydata/mcp-agent-trajectory-benchmark``) that ship
two field names harbor's current v1.7 model rejects with extra_forbidden:

    step.usage              -> step.metrics
    step.usage.cache_tokens -> step.metrics.cached_tokens

Same numeric semantics, just renamed. We apply the rename so v1.2 documents
validate cleanly; ``raw_blob`` retains the original un-normalised bytes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from harbor.models.trajectories import Trajectory
from trace_marketplace.adapters.base import BaseAdapter

log = logging.getLogger(__name__)


def _normalize_step_metrics(raw_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``raw_dict`` with ``step.usage`` renamed to ``step.metrics``.

    See module docstring for the rationale. Idempotent: if ``metrics`` is
    already present, ``usage`` is dropped and not re-renamed.
    """
    out = dict(raw_dict)
    new_steps: list[Any] = []
    for step in raw_dict.get("steps") or []:
        if not isinstance(step, dict):
            new_steps.append(step)
            continue
        new_step = dict(step)
        if "usage" in new_step and "metrics" not in new_step:
            metrics = dict(new_step.pop("usage") or {})
            if "cache_tokens" in metrics and "cached_tokens" not in metrics:
                metrics["cached_tokens"] = metrics.pop("cache_tokens")
            new_step["metrics"] = metrics
        new_steps.append(new_step)
    out["steps"] = new_steps
    return out


class ATIFAdapter(BaseAdapter):
    """Passthrough adapter for files that are already ATIF.

    Despite the "passthrough" label this is not the identity function: we
    still apply :func:`_normalize_step_metrics` to bridge minor schema
    drift between ATIF v1.2 and the v1.7 Pydantic models harbor ships.
    """

    source_format = "atif"

    def to_atif(
        self,
        raw_text: str,
        source_path: Path | None = None,
    ) -> Trajectory | None:
        try:
            raw_dict = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            log.warning(
                "ATIFAdapter: invalid JSON in %s: %s",
                source_path or "<no path>",
                exc,
            )
            return None

        if not isinstance(raw_dict, dict):
            log.warning(
                "ATIFAdapter: top-level value is %s, expected object (%s)",
                type(raw_dict).__name__,
                source_path or "<no path>",
            )
            return None

        normalized = _normalize_step_metrics(raw_dict)

        try:
            return Trajectory.model_validate(normalized)
        except ValidationError as exc:
            log.warning(
                "ATIFAdapter: validation failed for %s (%d error(s))",
                source_path or "<no path>",
                len(exc.errors()),
            )
            log.debug("Full validation error: %s", exc)
            return None


__all__ = ["ATIFAdapter", "_normalize_step_metrics"]
