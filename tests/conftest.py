"""Pytest configuration and shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Absolute path to ``tests/fixtures``."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def claude_code_minimal_path(fixtures_dir: Path) -> Path:
    """Path to the trimmed Claude Code JSONL fixture."""
    return fixtures_dir / "claude_code_minimal.jsonl"


@pytest.fixture(scope="session")
def swe_agent_sample_path(fixtures_dir: Path) -> Path:
    """Path to the synthetic SWE-agent HF row fixture."""
    return fixtures_dir / "swe_agent_sample.json"
