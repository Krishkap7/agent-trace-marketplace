"""Single-trace Sonnet call with retry, cost tracking, prompt caching.

The orchestrator (``scripts/generate_synthetic_traces.py``) parallelises
across many :class:`~trace_marketplace.synth.spec.TraceSpec` instances;
this module owns the per-spec work: build the prompt, call Anthropic,
validate the result, retry once with the adapter's error in the
follow-up, and return everything the orchestrator needs to log and
account.

Why retry exactly once?

* Empirically the first failure is almost always a small structural
  bug (mismatched ``call_id`` pair, ``tool_use`` block without
  matching ``tool_result``). Showing Sonnet the adapter's complaint
  fixes it 70-90% of the time.
* A second retry costs as much as one fresh attempt and almost never
  saves the trace. Cheaper to drop and let the orchestrator
  oversample.
* Bounded retries make the cost projection tight: every spec costs
  at most 2 generation passes.

Cost accounting follows the public Anthropic pricing for
``claude-sonnet-4-6``:
* Base input:        $3.00 / MTok
* Cache write:       $3.75 / MTok  (1.25x base, per Anthropic docs)
* Cache read:        $0.30 / MTok  (0.10x base)
* Output:            $15.00 / MTok

The orchestrator sums :class:`CostEstimate` across all generations
to enforce a wall-clock budget cap.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from anthropic import Anthropic, APIStatusError, RateLimitError

from trace_marketplace.synth.prompts import (
    OUTPUT_TOOL_NAME,
    build_prompt,
    extract_file_bytes,
    extract_notes,
)
from trace_marketplace.synth.spec import TraceSpec
from trace_marketplace.synth.validator import ValidationResult, validate

log = logging.getLogger(__name__)


GENERATOR_MODEL: str = "claude-sonnet-4-6"
"""The model the user explicitly chose. Sonnet 4.6 gives the diversity
+ instruction-following balance we need for varied failure modes.
Down-tiering to Haiku produced visibly templated traces in early
testing."""

GENERATOR_TEMPERATURE: float = 1.0
"""High temperature on purpose: every spec asks for the SAME failure
mode many times, so without thermal diversity Sonnet collapses to
2-3 stock templates per label. 1.0 keeps surface variation across
realisations of the same spec."""

GENERATOR_MAX_TOKENS: int = 16000
"""Synthetic traces with ~25 steps land around 6-10K output tokens.
16K leaves headroom for the chattier failure modes (tool_spamming,
loop) where the model emits many repeated tool calls."""

DEFAULT_RATE_LIMIT_RETRIES: int = 6
"""How many times we retry on HTTP 429 before giving up. Exponential
backoff means total wait is up to ~2 minutes across 6 attempts --
plenty of slack for transient quota blips, short enough that a true
quota wall trips the orchestrator's overall timeout."""


# ---------------------------------------------------------------------------
# Anthropic SDK protocol.
# ---------------------------------------------------------------------------


class _Messages(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _AnthropicLike(Protocol):
    messages: _Messages


# ---------------------------------------------------------------------------
# Cost / outcome dataclasses.
# ---------------------------------------------------------------------------

PRICE_PER_MTOK_INPUT: float = 3.00
PRICE_PER_MTOK_CACHE_WRITE: float = 3.75
PRICE_PER_MTOK_CACHE_READ: float = 0.30
PRICE_PER_MTOK_OUTPUT: float = 15.00


@dataclass(frozen=True)
class CostEstimate:
    """Token + dollar costs for one Anthropic call.

    Fields:
    * ``input_tokens``: non-cached input tokens billed at base rate.
    * ``cache_write_tokens``: tokens added to the prompt cache this
      call. Higher on the first request in a window, near-zero
      afterwards.
    * ``cache_read_tokens``: tokens served from the prompt cache.
    * ``output_tokens``: completion tokens.
    * ``usd``: cents-precision aggregated dollar cost.
    """

    input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0

    @property
    def usd(self) -> float:
        return (
            self.input_tokens * PRICE_PER_MTOK_INPUT
            + self.cache_write_tokens * PRICE_PER_MTOK_CACHE_WRITE
            + self.cache_read_tokens * PRICE_PER_MTOK_CACHE_READ
            + self.output_tokens * PRICE_PER_MTOK_OUTPUT
        ) / 1_000_000

    def __add__(self, other: "CostEstimate") -> "CostEstimate":
        return CostEstimate(
            input_tokens=self.input_tokens + other.input_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


def _usage_to_cost(usage: Any) -> CostEstimate:
    """Translate the SDK's ``Usage`` object into a :class:`CostEstimate`."""
    if usage is None:
        return CostEstimate()

    def _get(field_name: str) -> int:
        if isinstance(usage, dict):
            raw = usage.get(field_name)
        else:
            raw = getattr(usage, field_name, None)
        try:
            return int(raw or 0)
        except (TypeError, ValueError):
            return 0

    return CostEstimate(
        input_tokens=_get("input_tokens"),
        cache_write_tokens=_get("cache_creation_input_tokens"),
        cache_read_tokens=_get("cache_read_input_tokens"),
        output_tokens=_get("output_tokens"),
    )


@dataclass
class GenerationResult:
    """Outcome of :func:`generate_trace` for one spec.

    Mutable on purpose -- the orchestrator stitches in
    ``output_path`` after writing to disk. Everything else is set by
    this module.
    """

    spec: TraceSpec
    ok: bool
    file_bytes: str | None = None
    validation: ValidationResult | None = None
    attempts: int = 0
    error: str = ""
    cost: CostEstimate = field(default_factory=CostEstimate)
    notes: str | None = None
    output_path: str | None = None


# ---------------------------------------------------------------------------
# Anthropic call with rate-limit retry.
# ---------------------------------------------------------------------------


def _sleep_for_rate_limit(attempt: int, base: float = 2.0, jitter: float = 0.5) -> None:
    """Sleep using exponential backoff with jitter for HTTP 429.

    ``attempt`` is 1-indexed (1 -> base seconds, 2 -> base*2, ...).
    Jitter keeps the worker pool from re-firing in lockstep after a
    coordinated 429 wave.
    """
    delay = base * (2 ** (attempt - 1))
    delay += random.uniform(0, jitter * delay)
    time.sleep(delay)


def _create_with_retry(
    client: _AnthropicLike,
    *,
    system: list[dict[str, Any]],
    user_message: str,
    tools: list[dict[str, Any]],
    max_429_retries: int = DEFAULT_RATE_LIMIT_RETRIES,
) -> Any:
    """Call ``client.messages.create`` with exponential backoff on 429s.

    All other API errors (auth, 500, malformed request) propagate
    immediately -- they aren't transient.

    Returns the raw SDK response. Caller handles content extraction.
    """
    last_429: RateLimitError | None = None
    for attempt in range(1, max_429_retries + 1):
        try:
            return client.messages.create(
                model=GENERATOR_MODEL,
                max_tokens=GENERATOR_MAX_TOKENS,
                temperature=GENERATOR_TEMPERATURE,
                system=system,
                tools=tools,
                tool_choice={"type": "tool", "name": OUTPUT_TOOL_NAME},
                messages=[{"role": "user", "content": user_message}],
            )
        except RateLimitError as exc:
            last_429 = exc
            log.warning(
                "Anthropic rate-limited (attempt %d/%d); backing off",
                attempt,
                max_429_retries,
            )
            if attempt == max_429_retries:
                break
            _sleep_for_rate_limit(attempt)
        except APIStatusError as exc:
            # 5xx are sometimes transient too; retry like 429 but
            # only twice. 4xx (other than 429) are configuration
            # errors -- bubble up.
            status = getattr(exc, "status_code", None)
            if status is not None and 500 <= int(status) < 600 and attempt < 3:
                log.warning(
                    "Anthropic %s on attempt %d; retrying", status, attempt
                )
                _sleep_for_rate_limit(attempt)
                continue
            raise
    assert last_429 is not None  # noqa: S101
    raise last_429


# ---------------------------------------------------------------------------
# Tool-use extraction.
# ---------------------------------------------------------------------------


def _first_tool_use_block(message_obj: Any) -> dict[str, Any] | None:
    """Return the first ``submit_trace`` tool_use block on the response.

    Normalises across "SDK dataclass" and "plain dict" content blocks
    so tests can mock with simple dicts while production uses the
    Anthropic SDK objects.
    """
    content = getattr(message_obj, "content", None) or []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type is None and isinstance(block, dict):
            block_type = block.get("type")
        if block_type != "tool_use":
            continue
        name = getattr(block, "name", None)
        if name is None and isinstance(block, dict):
            name = block.get("name")
        if name != OUTPUT_TOOL_NAME:
            continue
        raw_input = getattr(block, "input", None)
        if raw_input is None and isinstance(block, dict):
            raw_input = block.get("input")
        if not isinstance(raw_input, dict):
            return None
        return {"type": "tool_use", "name": name, "input": raw_input}
    return None


# ---------------------------------------------------------------------------
# Retry-with-feedback prompt builder.
# ---------------------------------------------------------------------------


def _retry_brief(original_brief: str, prior_error: str) -> str:
    """Build the follow-up brief that asks Sonnet to fix what broke.

    We re-state the original task verbatim (cheap) and append the
    adapter's complaint. Putting the error inside the brief rather
    than as a separate assistant turn keeps the call single-shot --
    no multi-turn token blowup for the retry.
    """
    return (
        f"{original_brief}\n\n"
        f"IMPORTANT: A previous attempt at this trace was rejected "
        f"by the adapter with this error:\n"
        f"  > {prior_error}\n\n"
        f"Re-generate the trace from scratch. Pay close attention "
        f"to ID cross-references (every ``tool_use.id`` /  "
        f"``tool_call.id`` / ``call_id`` MUST be referenced by "
        f"exactly one matching result entry later in the file). "
        f"Submit the complete fixed trace via ``submit_trace``."
    )


# ---------------------------------------------------------------------------
# Single-trace public API.
# ---------------------------------------------------------------------------


def generate_trace(
    spec: TraceSpec,
    *,
    client: _AnthropicLike | None = None,
    max_429_retries: int = DEFAULT_RATE_LIMIT_RETRIES,
    validate_fn: Any = validate,
) -> GenerationResult:
    """Generate ONE synthetic trace for ``spec``.

    Returns a :class:`GenerationResult` capturing the outcome:

    * ``ok=True``: ``file_bytes`` carries the validated content (the
      adapter has already parsed it once successfully) and
      ``validation.trajectory`` is the parsed ATIF. The caller is
      expected to write ``file_bytes`` to disk and let the existing
      ingest scripts re-parse on demand.
    * ``ok=False``: ``error`` describes why we gave up. ``file_bytes``
      may carry the LAST raw response so the dry-run debugger can
      inspect why validation failed.

    Parameters
    ----------
    spec
        The recipe to act on.
    client
        Anthropic-like client; tests inject a mock. Production callers
        pass ``None`` and we instantiate ``Anthropic()`` so the SDK
        picks up ``ANTHROPIC_API_KEY`` from the environment.
    max_429_retries
        Forwarded to :func:`_create_with_retry`.
    validate_fn
        Test seam -- defaults to the real validator. Tests can pass a
        stub to avoid having to construct valid adapter input.

    Errors are NEVER raised for "model didn't produce valid output"
    -- that's an outcome, not a programming bug. We do let SDK
    failures (auth, persistent 429s, network) bubble out so the
    orchestrator can decide whether to fail the whole run.
    """
    if client is None:
        client = Anthropic()

    built = build_prompt(spec)
    last_validation: ValidationResult | None = None
    last_bytes: str | None = None
    last_notes: str | None = None
    total_cost = CostEstimate()
    attempts = 0
    error_for_retry = ""

    for attempt in (1, 2):
        attempts = attempt
        brief = (
            built.user_brief
            if attempt == 1
            else _retry_brief(built.user_brief, error_for_retry)
        )

        response = _create_with_retry(
            client,
            system=built.system,
            user_message=brief,
            tools=built.tools,
            max_429_retries=max_429_retries,
        )
        total_cost = total_cost + _usage_to_cost(getattr(response, "usage", None))

        tool_block = _first_tool_use_block(response)
        file_bytes = extract_file_bytes(tool_block) if tool_block else None
        last_notes = extract_notes(tool_block) if tool_block else last_notes

        if file_bytes is None:
            error_for_retry = "no submit_trace tool_use block in response"
            log.warning(
                "spec[%s/%s] attempt %d: %s",
                spec.output_format,
                spec.primary_failure,
                attempt,
                error_for_retry,
            )
            continue

        last_bytes = file_bytes
        result = validate_fn(spec.output_format, file_bytes)
        last_validation = result
        if result.ok:
            return GenerationResult(
                spec=spec,
                ok=True,
                file_bytes=file_bytes,
                validation=result,
                attempts=attempt,
                cost=total_cost,
                notes=last_notes,
            )

        error_for_retry = result.error
        log.info(
            "spec[%s/%s] attempt %d failed validation: %s",
            spec.output_format,
            spec.primary_failure,
            attempt,
            result.error,
        )

    return GenerationResult(
        spec=spec,
        ok=False,
        file_bytes=last_bytes,
        validation=last_validation,
        attempts=attempts,
        error=error_for_retry or "exhausted retries with no valid trace",
        cost=total_cost,
        notes=last_notes,
    )


__all__ = [
    "CostEstimate",
    "DEFAULT_RATE_LIMIT_RETRIES",
    "GENERATOR_MAX_TOKENS",
    "GENERATOR_MODEL",
    "GENERATOR_TEMPERATURE",
    "GenerationResult",
    "PRICE_PER_MTOK_CACHE_READ",
    "PRICE_PER_MTOK_CACHE_WRITE",
    "PRICE_PER_MTOK_INPUT",
    "PRICE_PER_MTOK_OUTPUT",
    "generate_trace",
]
