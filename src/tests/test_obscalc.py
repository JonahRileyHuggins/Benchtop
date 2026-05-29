import os
import sys
import shutil
import random
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
    # Clean up remaining data to avoid errrors.
    cache_dir = "./.cache"

    try: 
        os.makedirs(cache_dir, exist_ok=False)
    except OSError as e:
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=False)

    data_path = os.path.join(os.path.dirname(__file__), "data", "LR-benchmark.yaml")

    exp = SimpleNamespace()
    exp.loader = FileLoader(data_path)
    exp.loader._petab_files()
    exp.record = Record(exp.loader.problems[0])

    return exp

def make_dummy_data(exp: SimpleNamespace):
    """
    Populates .cache directory with deterministic values for each results_dict key.
    Each key gets a DataFrame of integers for simple arithmetic testing.

    args:
        - exp (SimpleNamespace): needed for dictionary unique identifiers

    """
    lr_model_sp = [
        "cyt_prot__LIGAND_", "cyt_prot__RECEPTOR_", "cyt_prot__LIGAND__RECEPTOR_",
        "nuc_gene_a__LIGAND_", "nuc_gene_i__LIGAND_", "nuc_gene_a__RECEPTOR_",
        "nuc_gene_i__RECEPTOR_", "cyt_mrna__LIGAND_", "cyt_mrna__RECEPTOR_"
    ]

    for key, entry in exp.record.cache.results_dict.items():
        condition_id = entry["conditionId"]
        cell = entry["cell"]

        # deterministic integers based on cell index + species index
        data = {}
        for i, species in enumerate(lr_model_sp):
            # simple formula: cell * 10 + species index → deterministic per entry
            data[species] = [cell * 10 + i for i in range(10)]  # 10 timesteps

        # include a time column
        data["time"] = list(range(10))

        dummy_df = pd.DataFrame(data)

        # Save to cache
        exp.record.cache.save(key, dummy_df)

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

def test_obscalc_run():
    """Unit test for ObservableCalculator.run() with deterministic data."""

    # Prepare experiment and deterministic cache data
    exp = dummy_experiment()
    make_dummy_data(exp)

    # Create ObservableCalculator instance
    obs = ObservableCalculator(exp)

    # Run the calculation
    obs.run()

    # --- Basic sanity checks ---

    # 1) Ensure results_dict still has same keys
    assert len(obs.observable_results) > 0, "No entries in results_dict after run()"

    # 2) Pick a random key and check its cached DataFrame
    random_key = random.choice(list(obs.observable_results.keys()))

    sim = obs.observable_results[random_key]
    cond_id = sim["conditionId"]
    if cond_id == "heterogenize":
        # heterogenize has blank observable, only should store 0:
        assert sim["blank"]["simulation"][0] == 0, "heterogenize storing wrong values"

    else: 
        assert np.all(
            np.diff(sim["LR-complex"]["simulation"]) >= 0
            ), \
            f"Data {random_key} is not sorted ascending"
        

    print("✅ test_obscalc_run passed successfully")






if __name__ == "__main__":

    test_calculate_formula()
    test_obscalc_run()

    shutil.rmtree(".cache")
