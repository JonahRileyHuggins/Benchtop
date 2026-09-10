"""Tests for the read-only `benchtop design` preview."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchtop.design import render_design
from benchtop.launcher import launch_design, main

DATA_DIR = Path(__file__).resolve().parent / "data"
BENCHMARK_YAML = DATA_DIR / "LR-benchmark.yaml"


def test_render_design_lr_benchmark():
    text = render_design(str(BENCHMARK_YAML))

    assert "PROBLEM  test-benchmark" in text
    assert "PROBLEM  test-benchmark-2" in text
    assert "simulator  bngsim" in text
    assert "simulator  tellurium" in text
    assert "heterogenize" in text
    assert "primary-condition" in text
    assert "adjacent-primary" in text
    assert "x1" in text
    assert "x3" in text
    assert "TOTAL: 12 simulations" in text
    assert "preview only" in text
    assert "simulations: 3 conditions x 1 cell = 3" in text
    assert "simulations: 3 conditions x 3 cells = 9" in text

    # heterogenize is the parent of both simulation conditions
    hetero_at = text.index("heterogenize")
    primary_at = text.index("primary-condition")
    adjacent_at = text.index("adjacent-primary")
    assert hetero_at < primary_at < adjacent_at
    assert "+-- primary-condition" in text
    assert "+-- adjacent-primary" in text


def test_render_design_verbose_includes_overrides():
    text = render_design(str(BENCHMARK_YAML), verbose=True)
    assert "cyt_prot__LIGAND_" in text
    assert "conditionId" in text


def test_design_does_not_construct_experiment(monkeypatch):
    constructed = []

    def boom(*_args, **_kwargs):
        constructed.append(True)
        raise AssertionError("Experiment should not be constructed")

    monkeypatch.setattr("benchtop.experiment.Experiment.__init__", boom)
    render_design(str(BENCHMARK_YAML))
    assert constructed == []


def test_design_creates_no_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    render_design(str(BENCHMARK_YAML))
    assert not (tmp_path / ".cache").exists()
    assert list(tmp_path.rglob("*.pkl")) == []
    assert not (tmp_path / "results").exists()


def test_launch_design_requires_path():
    with pytest.raises(AssertionError, match="PEtab YAML path"):
        launch_design(SimpleNamespace(path=None, verbose=False))


def test_missing_yaml_fails_before_simulation():
    with pytest.raises(FileNotFoundError):
        render_design("does-not-exist.yaml")

    with pytest.raises(FileNotFoundError, match="Experiment not found"):
        launch_design(SimpleNamespace(path="does-not-exist.yaml", verbose=False))


def test_main_design_does_not_run_experiment(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchtop", "design", "-p", str(BENCHMARK_YAML)],
    )
    called = []
    monkeypatch.setattr(
        "benchtop.launcher.launch_experiment",
        lambda _args: called.append("experiment"),
    )

    main()

    assert called == []
    out = capsys.readouterr().out
    assert "TOTAL: 12 simulations" in out
    assert "preview only" in out
