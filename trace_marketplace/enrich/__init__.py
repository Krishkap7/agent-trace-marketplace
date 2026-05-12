"""Trace enrichment: structural rules + LLM judges that decorate
already-ingested traces with failure signals and labels.

Layered intentionally so the deterministic pieces (taxonomy + rules)
have no I/O and no network deps; only :mod:`failure_judge` reaches
out to the Anthropic API.

- :mod:`failure_taxonomy` -- the ``FailureLabel`` enum + per-label
  descriptions; single source of truth for the judge prompt.
- :mod:`failure_rules`    -- pure ``signal_*(traj) -> bool`` functions
  plus a ``compute_signals`` aggregator.
- :mod:`failure_judge`    -- Anthropic SDK wrapper that asks Sonnet
  for a structured ``(label, reasoning)`` decision.
"""
