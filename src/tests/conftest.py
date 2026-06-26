"""Shared pytest fixtures for Benchtop tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchtop.registry import load_simulator

TESTS_DIR = Path(__file__).resolve().parent
DATA_DIR = TESTS_DIR / "data"
BENCHMARK_YAML = DATA_DIR / "LR-benchmark.yaml"
SBML_PATH = DATA_DIR / "LR-model.xml"


@pytest.fixture
def benchmark_yaml() -> Path:
    return BENCHMARK_YAML


@pytest.fixture
def fresh_cache(tmp_path) -> str:
    """Fresh cache directory path for each test."""
    cache = tmp_path / ".cache"
    cache.mkdir()
    return str(cache)


@pytest.fixture
def tellurium_simulator():
    """Return TelluriumSimulator class if tellurium is installed."""
    try:
        return load_simulator("tellurium")
    except ImportError:
        pytest.skip("tellurium not installed")


@pytest.fixture
def tellurium_args():
    return SimpleNamespace(model_paths=[str(SBML_PATH)])
