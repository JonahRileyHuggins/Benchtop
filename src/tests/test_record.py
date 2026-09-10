import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.append(os.path.dirname(__file__))
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "src",
        "benchtop"
    )
)

from benchtop._record import Record

problem = SimpleNamespace()
problem.name = "test-problem"
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


def make_dummy_record(cache_dir: str) -> Record:
    os.makedirs(cache_dir, exist_ok=True)

    dummy_record = Record(
        problems=problem,
        cache_dir=cache_dir,
        load_index=False,
        no_confirm=True,
    )

    df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})

    return_key = [
        key for key in dummy_record.cache.job_keys()
        if dummy_record.cache.results_dict[key]["conditionId"] == "primary-condition"
        and dummy_record.cache.results_dict[key]["cell"] == 2
    ][0]

    dummy_record.cache.save(return_key, df)

    return dummy_record

def test_record_constructor(tmp_path) -> None:
    """Verify Record builds a correct results dictionary and cache."""
    cache_dir = str(tmp_path / ".cache")
    rec = make_dummy_record(cache_dir)

    assert isinstance(rec.cache.results_dict, dict), \
        "Record.results_dict must be a dictionary"

    assert len(rec.cache.job_keys()) == 6, \
        "Record should create one entry per (condition × cell)"

    for identifier in rec.cache.job_keys():
        entry = rec.cache.results_dict[identifier]
        assert "conditionId" in entry
        assert "cell" in entry
        assert "complete" in entry
        assert "problem" in entry

        try:
            uuid.UUID(identifier)
        except ValueError:
            raise AssertionError(f"Identifier {identifier} is not a valid UUID")

    assert os.path.exists(cache_dir), \
        "Cache directory should exist after constructing a Record"


def test_results_lookup(tmp_path) -> None:
    """Test loading back the result entry associated with a condition+cell."""
    rec = make_dummy_record(str(tmp_path / ".cache"))

    target_condition = "primary-condition"
    target_cell = 2

    matching_keys = [
        key for key in rec.cache.job_keys()
        if rec.cache.results_dict[key]["conditionId"] == target_condition
        and rec.cache.results_dict[key]["cell"] == target_cell
    ]
    assert len(matching_keys) == 1, \
        "Expected one match for condition/cell pair"

    identifier = matching_keys[0]

    result = rec.cache.load(identifier)
    assert isinstance(result, pd.DataFrame), \
        "Record.load should return a pandas DataFrame for empty entries"


if __name__ == "__main__":
    import tempfile

    test_record_constructor(Path(tempfile.mkdtemp()))
    test_results_lookup(Path(tempfile.mkdtemp()))
