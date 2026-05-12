"""Adapter abstract base class.

An adapter takes raw text from a single source format and returns a validated
ATIF :class:`Trajectory`. The interface is intentionally minimal:

* ``to_atif(raw_text, source_path=None)`` is a pure function. It MUST NOT
  raise on bad data -- return ``None`` and log a warning instead. The caller
  (the ingest pipeline) always stores ``raw_blob``; ``None`` simply means the
  ``atif`` column stays NULL for that row.
* ``source_format`` is a class attribute every adapter sets. The string must
  match a key produced by :func:`trace_marketplace.ingest.detect.detect_format`.
* ``source_path`` is optional context. Some adapters synthesise a trace id
  from the filename when the file's content lacks one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from harbor.models.trajectories import Trajectory


class BaseAdapter(ABC):
    """Convert one raw trace into an ATIF :class:`Trajectory`."""

    source_format: str

    @abstractmethod
    def to_atif(
        self,
        raw_text: str,
        source_path: Path | None = None,
    ) -> Trajectory | None:
        """Parse ``raw_text`` and return a validated ATIF Trajectory or ``None``.

        Implementations MUST NOT raise on malformed input. Logging at WARNING
        when returning ``None`` is required so operators can diagnose.
        """
