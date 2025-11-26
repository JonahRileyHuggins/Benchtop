import os
import sys
import uuid
import shutil
from types import SimpleNamespace
sys.path.append(os.path.dirname(__file__))
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "src",
        "benchtop"
    )
)
import pandas as pd

from Record import Record

problem = SimpleNamespace()
problem.cell_count= 3
problem.condition_files = [pd.DataFrame({
        "conditionId": ["heterogenize", "primary-condition"],
        "conditionName": ["base values", "some start"],
        "cyt_prot__LIGAND_": [0, 10],
        "nuc_gene_a__LIGAND_": [2, 2],
        "nuc_gene_i__LIGAND_": [0, 0],
        "nuc_gene_a__RECEPTOR_": [2, 2],
        "nuc_gene_i__RECEPTOR_": [0, 0],
        "cyt_mrna__LIGAND_": [5, 5],
        "cyt_mrna__RECEPTOR_": [5, 5],
    })]
problem.measurement_files = [pd.DataFrame({
        "observableId": ["blank", "R_gene_activity"],
        "preequilibrationConditionId": ["None", "heterogenize"],
        "simulationConditionId": ["heterogenize", "primary-condition"],
        "measurement": ["None", "60"],
        "time": ["0", "20"]
    })]


def make_dummy_record() -> Record:
    cache_dir = "./.cache"

    try: 
        os.makedirs(cache_dir, exist_ok=False)
    except OSError as e:
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=False)

    # Create record
    dummy_record = Record(
        problem=problem,
        cache_dir=cache_dir,
        load_index=False
    )

    # Create a simple DataFrame to cache
    df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})

    # Pick one valid key from the auto-generated results_dict

    return_key = [key for key in dummy_record.cache.results_dict.keys()\
                  if dummy_record.cache.results_dict[key]["conditionId"] == "primary-condition"\
                    and dummy_record.cache.results_dict[key]["cell"]== 2][0]

    # Save the dataframe into the cache so a pickle file is actually written
    dummy_record.cache.save(return_key, df)

    return dummy_record

def test_record_constructor() -> None:
    """Verify Record builds a correct results dictionary and cache."""
    rec = make_dummy_record()

    # --- Check for dict structure ---
    assert isinstance(rec.cache.results_dict, dict), \
        "Record.results_dict must be a dictionary"

    # There are 2 conditions × 3 cells = 6 entries expected
    assert len(rec.cache.results_dict) == 6, \
        "Record should create one entry per (condition × cell)"

    # --- Check each entry structure ---
    for identifier, entry in rec.cache.results_dict.items():
        assert "conditionId" in entry
        assert "cell" in entry
        assert "complete" in entry

        # Identifiers should be UUIDs because measurement_df has no datasetId
        try:
            uuid.UUID(identifier)
        except ValueError:
            raise AssertionError(f"Identifier {identifier} is not a valid UUID")

    # --- Cache directory exists ---
    assert os.path.exists("./.cache"), \
        "Cache directory should exist after constructing a Record"

    print("✅ test_record_constructor passed")


def test_results_lookup() -> None:
    """Test loading back the result entry associated with a condition+cell."""
    rec = make_dummy_record()

    # Pick one known condition/cell pair
    target_condition = "primary-condition"
    target_cell = 2

    # Grab the identifier that matches this pair
    matching_keys = [
        key for key, entry in rec.cache.results_dict.items()
        if entry["conditionId"] == target_condition and entry["cell"] == target_cell
    ]
    assert len(matching_keys) == 1, \
        "Expected one match for condition/cell pair"

    identifier = matching_keys[0]

    # The cache stores a DataFrame, but empty for now --> load() returns a pandas dataframe
    result = rec.cache.load(identifier)
    assert isinstance(result, pd.DataFrame), \
        "Record.load should return a pandas DataFrame for empty entries"

    print("✅ test_results_lookup passed")

if __name__ == "__main__":

    test_record_constructor()
    test_results_lookup()