"""Shared pytest fixtures for Benchtop tests."""

import shutil
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
DATA_DIR = TESTS_DIR / "data"
BENCHMARK_YAML = DATA_DIR / "LR-benchmark.yaml"


@pytest.fixture
def benchmark_yaml() -> Path:
    return BENCHMARK_YAML


@pytest.fixture
def fresh_cache(tmp_path) -> str:
    """Fresh cache directory path for each test."""
    cache = tmp_path / ".cache"
    cache.mkdir()
    return str(cache)
