import os
import sys
sys.path.append(os.path.dirname(__file__))

import pandas as pd
from unittest.mock import patch, MagicMock

from src.benchtop.file_loader import FileLoader
from src.benchtop.Record import Record

problem = {
    "cell_count": 3,
    "condition_files": [pd.DataFrame({
        "conditionId": ["heterogenize", "primary-condition"],
        "conditionName": ["base values", "some start"],
        "cyt_prot__LIGAND_": [0, 10],
        "nuc_gene_a__LIGAND_": [2, 2],
        "nuc_gene_i__LIGAND_": [0, 0],
        "nuc_gene_a__RECEPTOR_": [2, 2],
        "nuc_gene_i__RECEPTOR_": [0, 0],
        "cyt_mrna__LIGAND_": [5, 5],
        "cyt_mrna__RECEPTOR_": [5, 5],
        })
        ],
    "measurement_files": [pd.DataFrame({
        "observableId":["blank", "R_gene_activity"],
        "preequilibrationConditionId": ["None", "heterogenize"],
        "simulationConditionId": ["heterogenize", "primary-condition"],
        "measurement": ["None", "60"],
        "time": ["0", "20"]
        })
        ],

}

def make_dummy_record() -> Record:
    # fake record objet for testing
    dummy_record = Record(
        problem=problem,
        cache_dir="./tests/data/.cache",
        load_index=True
    )
    return dummy_record

def terst_record_constructor() -> None:

    dummy_record = make_dummy_record()

    assert type(dummy_record.results_dict) == dict
    
    assert os.path.exists("./tests/data/.cache")
 