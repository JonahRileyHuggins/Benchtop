import os
import sys
from unittest.mock import MagicMock
from types import SimpleNamespace

import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(__file__))
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "src",
        "benchtop"
    )
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
YAML_PATH = os.path.join(DATA_DIR, "LR-benchmark.yaml")

from file_loader import FileLoader
from Record import Record
from ObservableCalculator import ObservableCalculator

def dummy_experiment() -> SimpleNamespace:
    """Observable_calculator uses composition, requiring experiment class object"""

    data_path = os.path.join(os.path.dirname(__file__), "data", "LR-benchmark.yaml")

    exp = SimpleNamespace()
    exp.loader = FileLoader(data_path)
    exp.loader._petab_files()
    exp.record = Record(exp.loader.problems[0])

    return exp

def test_obscalc_constructor():
    """Ensure constructor completes with validated test benchmark."""

    exp = dummy_experiment()

    try:
        obs = ObservableCalculator(exp)
    except Exception as e:
        # Hard fail if constructor raises anything
        raise AssertionError(f"ObservableCalculator constructor failed: {e}")

    # Basic sanity check
    assert isinstance(obs, ObservableCalculator), \
        "Constructor did not return an ObservableCalculator instance"


def test_calculate_formula() -> None:
    """Test method for calculating a specific formula."""

    exp = dummy_experiment()

    arr1 = np.arange(1, 11)
    arr2 = np.arange(1, 11)
    time = np.arange(0, 60, 6)

    keys = ["nuc_gene_a__RECEPTOR_", "nuc_gene_i__RECEPTOR_"]

    dataset = pd.DataFrame({
        keys[0]: arr1,
        keys[1]: arr2,
        "time": time
    })

    obs = ObservableCalculator(exp)

    formula = f"{keys[0]} + {keys[1]}"

    # From your test benchmark: ("primary-condition", "R_gene_activity")
    group = obs.data_groups.get_group(("primary-condition", "R_gene_activity"))

    results = obs._calculate_formula(dataset, formula, group)

    # expected time indices: 0, 30, 60 seconds → dataset indices [0,5,9]
    time_idx = [0, 5, 9]
    answer = (arr1 + arr2)[time_idx]

    # Check exact ordering and equality
    np.testing.assert_array_equal(
        results,
        answer,
        err_msg="_calculate_formula() returned results out of order"
    )
    print("✅ test_calculate_formula passed")

if __name__ == "__main__":

    test_calculate_formula()
