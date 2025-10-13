import os
import sys
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import pytest

from src.benchtop.file_loader import FileLoader
from src.benchtop.Record import Record
from src.benchtop.Worker import Worker
from wrappers.tellurium_wrapper import WrapTellurium

cache_dir = "data/.cache/"
if not os.path.exists(cache_dir) or len(os.listdir(cache_dir)) != 13:
    from test_benchtop import test_run
    test_run()

petab_yaml = "LR-Benchmark/LR-benchmark.yaml"

loader = FileLoader(petab_yaml)
loader._petab_files()

record = Record(
    problem=loader.problems[0],
    cache_dir="./data/.cache",
    load_index=True,
)

sbml_path = "LR-Benchmark/LR-model.xml"
task = "heterogenize+1"


def test_worker() -> None:
    grunt = Worker(
        task=task,
        record=record,
        simulator=WrapTellurium,
        args=sbml_path,
        start=0.0,
        step=30.0,
    )

    # Sanity check
    assert grunt is not None
    assert isinstance(grunt, Worker)
    assert grunt.simulator is None


def test_find_preequilibration_results(monkeypatch) -> None:
    """Test Worker.__extract_preequilibration_results logic in isolation."""
    grunt = Worker(
        task=task,
        record=record,
        simulator=WrapTellurium,
        args=sbml_path,
        start=0.0,
        step=30.0,
    )

    cond_id, cell_num = task.split("+")
    cell_num = int(cell_num)

    # --- mock measurement DataFrame ---
    measurement_df = pd.DataFrame({
        "simulationConditionId": ["heterogenize"],
        "preequilibrationConditionId": ["baseline"]
    })

    # --- mock record.results_lookup() ---
    def fake_results_lookup(condition_id, cell):
        if condition_id == "baseline" and cell == cell_num:
            return pd.DataFrame({
                "time": [0, 100],
                "species_A": [1.0, 2.0],
                "species_B": [3.0, 4.0],
            })
        return None

    # Apply mocks
    grunt.record.problem.measurement_files = [measurement_df]
    monkeypatch.setattr(grunt.record, "results_lookup", fake_results_lookup)

    # --- Execute ---
    results = grunt._Worker__extract_preequilibration_results(cond_id, cell_num)

    # --- Validate ---
    assert isinstance(results, list)
    assert results == [2.0, 4.0], "Should extract final row (excluding time column)."


def test_find_preequilibration_results_no_match(monkeypatch) -> None:
    """Ensure an empty list is returned when no preequilibration condition exists."""
    grunt = Worker(
        task=task,
        record=record,
        simulator=WrapTellurium,
        args=sbml_path,
        start=0.0,
        step=30.0,
    )

    measurement_df = pd.DataFrame({
        "simulationConditionId": ["heterogenize"],
        "preequilibrationConditionId": [None],
    })

    grunt.record.problem.measurement_files = [measurement_df]
    monkeypatch.setattr(grunt.record, "results_lookup", lambda *_: None)

    results = grunt._Worker__extract_preequilibration_results("heterogenize", 1)
    assert results == [], "Should return empty list when no valid preequilibration condition."
