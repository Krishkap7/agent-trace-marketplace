"""The failure-label vocabulary used by the LLM judge.

Single source of truth: the ``FailureLabel`` enum and its companion
:data:`LABEL_DESCRIPTIONS` mapping. The judge prompt builds its label
list from these two together, so descriptions cannot drift from the
values they describe (a class of bug that bit slice 2.7 when adapter
constants were duplicated across modules).

The vocabulary is intentionally small (11 values). A larger taxonomy
gives the model more chances to hedge into noise labels; an 11-way
classification with an explicit ``unclear`` member covers the demo
without sacrificing reproducibility.

Why a separate ``LABEL_DESCRIPTIONS`` dict rather than per-member
docstrings? Python's ``Enum`` does not attach bare-literal strings
following a member assignment to that member's ``__doc__``. Putting
descriptions in a dict next to the enum keeps the data unambiguous
and import-cheap, and :func:`labels_with_descriptions` is the only
allowed way to format them for the prompt.
"""

from __future__ import annotations

from enum import Enum


class FailureLabel(str, Enum):
    """The 11-way failure-mode classification for any ingested trace."""

    # behavioural patterns
    LOOP = "loop"
    TOOL_SPAMMING = "tool_spamming"
    GAVE_UP = "gave_up"
    INCOMPLETE = "incomplete"

    # cognitive patterns
    WRONG_APPROACH = "wrong_approach"
    HALLUCINATED_API = "hallucinated_api"
    MISREAD_ERROR = "misread_error"

    # environment / external
    ENVIRONMENT_ISSUE = "environment_issue"

    # negative class
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    UNCLEAR = "unclear"


LABEL_DESCRIPTIONS: dict[FailureLabel, str] = {
    FailureLabel.LOOP: "Same tool call repeated N+ times in a row with no progress.",
    FailureLabel.TOOL_SPAMMING: "Many tool calls with low signal-to-noise; agent thrashes.",
    FailureLabel.GAVE_UP: "Agent explicitly abandoned the task with give-up language.",
    FailureLabel.INCOMPLETE: "Session ended mid-action (pending tool call, truncated output).",
    FailureLabel.WRONG_APPROACH: (
        "Agent pursued an incorrect plan that would never succeed."
    ),
    FailureLabel.HALLUCINATED_API: (
        "Agent invented a function / file / library that does not exist."
    ),
    FailureLabel.MISREAD_ERROR: (
        "Agent misinterpreted an error message and acted on the wrong cause."
    ),
    FailureLabel.ENVIRONMENT_ISSUE: (
        "Tool / network / setup failure outside the agent's control."
    ),
    FailureLabel.SUCCESS: "Task completed correctly.",
    FailureLabel.PARTIAL_SUCCESS: (
        "Some progress made but the task was not fully completed."
    ),
    FailureLabel.UNCLEAR: "The trace does not carry enough signal to classify.",
}


def labels_with_descriptions() -> str:
    """Return the prompt-ready label list.

    One line per label, formatted ``"- value: description"``, in the
    enum's declared order. The judge prompt embeds the output verbatim
    so the model and our validator can never disagree about which
    labels exist or what they mean.
    """
    lines: list[str] = []
    for member in FailureLabel:
        description = LABEL_DESCRIPTIONS.get(member, "(no description)")
        lines.append(f"- {member.value}: {description}")
    return "\n".join(lines)


__all__ = ["FailureLabel", "LABEL_DESCRIPTIONS", "labels_with_descriptions"]
