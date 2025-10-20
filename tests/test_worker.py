import os
import sys
from multiprocessing import Manager
from unittest.mock import patch, MagicMock

import pandas as pd

sys.path.append(os.path.dirname(__file__))
from src.benchtop.Worker import Worker

cache_dir = "./tests/data/.cache/"
if not os.path.exists(cache_dir) or len(os.listdir(cache_dir)) < 13:
    from test_benchtop import test_run
    test_run()

def make_dummy_worker():
    """Creates a worker instance with minimal mocks required for testing"""
    # fake simulator with dummy methods for class
    dummy_simulator = MagicMock()
    dummy_simulator.load = MagicMock()
    dummy_simulator.modify = MagicMock()
    dummy_simulator.getStateIds = MagicMock()
    results_df = pd.DataFrame({
        "good_var1": [1.0], 
        "good_var2": [2.0],
    })
    dummy_simulator.simulate = MagicMock(return_value=results_df)

    # fake record objet for testing
    dummy_record = MagicMock()
    dummy_record.problem = MagicMock()

    measurement_df = pd.DataFrame({
        "preequilibrationConditionId": ["None", "heterogenize"],
        "simulationConditionId": ["primary-condition", "heterogenize"],
        "time": [0, 1]
    })
    dummy_record.problem.measurement_files = [measurement_df]

    dummy_record.results_lookup = MagicMock(return_value=None)

    series = pd.Series(
        data=[ "primary-condition", "base values", 0, 2 ],
        index=["conditionId", "conditionName", "good_var1", "good_var2"]
    )

    dummy_record.condition_cell_id = MagicMock(
        return_value=(series, "1", "primary-condition")
    )

    # with Manager() as manager:
    #     lock = manager.Lock()

    # Construct Worker using MagicMock simulator type
    grunt = Worker(
        task="primary-condition+1",
        record=dummy_record,
        simulator=dummy_simulator,
        # lock=lock,
        args="dummy_path.xml",
        start=0.0,
        step=30.0,
    )

    assert grunt.simulator is None
    
    # Replace simulator (since constructor may override it)
    grunt.simulator = dummy_simulator

    return grunt, dummy_simulator


def test_worker_constructor() -> None:
    grunt, dummy_simulator = make_dummy_worker()

    # Sanity check
    assert grunt is not None
    assert isinstance(grunt, Worker)


def test_find_preequilibration_results() -> None:
    """Test Worker.__extract_preequilibration_results logic in isolation."""
    grunt, dummy_simulator = make_dummy_worker()

    task = "heterogenize+1"
    cond_id, cell_num = task.split("+")
    cell_num = int(cell_num)

    # --- mock record.results_lookup() ---
    def fake_results_lookup(condition_id, cell):
        if condition_id == "heterogenize" and cell == cell_num:
            return pd.DataFrame({
                "time": [0, 100],
                "species_A": [1.0, 2.0],
                "species_B": [3.0, 4.0],
            })
        return None

    # Apply mocks
    with patch.object(grunt.record, "results_lookup", fake_results_lookup):
        results = grunt._Worker__extract_preequilibration_results(cond_id, cell_num)

    assert isinstance(results, list)
    assert results == [2.0, 4.0]

def test_find_preequilibration_results_no_match() -> None:
    """Ensure an empty list is returned when no preequilibration condition exists."""
    grunt, dummy_simulator = make_dummy_worker()

    with patch.object(grunt.record, "results_lookup", lambda *_: None):
        results = grunt._Worker__extract_preequilibration_results("heterogenize", 1)
    assert results == [], "Should return empty list when no valid preequilibration condition."

def test_setModelState_basic():
    grunt, dummy_simulator = make_dummy_worker()

    names = ["conditionId", "good_var1", "bad_var", "conditionName"]
    states = [0.0, 1.23, 4.56, 0.0]

    # Configure modify() to raise ValueError once
    def fake_modify(name, value):
        if name == "bad_var":
            raise ValueError("Invalid variable name")
    dummy_simulator.modify.side_effect = fake_modify

    # Run the private method directly
    try:
        grunt._Worker__setModelState(names, states)
    except ValueError as e:
        pass

    # Verify modify() calls
    expected_calls = [
        ("good_var1", 1.23),
        ("bad_var", 4.56),
    ]
    actual_calls = [tuple(call.args) for call in dummy_simulator.modify.call_args_list]

    assert actual_calls == expected_calls, (
        f"Expected modify calls {expected_calls}, got {actual_calls}"
    )

def test_get_simulation_time():

    grunt, _ = make_dummy_worker()
    
    measurement_df = pd.DataFrame({
        "simulationConditionId": ["not_id", "heterogenize"],
        "time": [0, 48]
    })
    grunt.record.problem.measurement_files = [measurement_df]

    series = pd.Series(
        data=[ "heterogenize", "base values", 0, 2 ],
        index=["conditionId", "conditionName", "good_var1", "good_var2"]
    )

    time = grunt._Worker__get_simulation_time(series)

    assert time == 48, (
        f"Expected time returned 48, got {time}"
    )

def test_model_state_assignment():
    # --- Setup ---
    grunt, _ = make_dummy_worker()

    # Replace simulator with mock
    grunt.simulator = MagicMock()
    grunt.simulator.modify = MagicMock()

    # 10 mock model variable names (excluding blacklisted ones)
    model_names = [f"var_{i}" for i in range(10)]
    model_states = list(range(1, 11))  # new values, all nonzero

    # --- Action ---
    grunt._Worker__setModelState(model_names, model_states)

    # --- Assert ---
    # ensure modify() called once per variable
    assert grunt.simulator.modify.call_count == 10, \
        f"Expected 10 modify calls, got {grunt.simulator.modify.call_count}"

    # confirm all variables were modified with nonzero values
    for (name, value), call in zip(zip(model_names, model_states),
                                   grunt.simulator.modify.call_args_list):
        args, _ = call
        assert args == (name, value), f"Unexpected modify args {args}"
        assert value != 0, f"Variable {name} was not reassigned properly"

    print("✅ All 10 model states reassigned without error.")


