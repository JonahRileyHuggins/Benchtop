
from unittest.mock import MagicMock

import pandas as pd

def dummy_simulator(*args, **kwargs) -> None:

    results_df = pd.DataFrame({
    "cyt_prot__LIGAND_": [0, 10, 4],
    "nuc_gene_a__LIGAND_": [2, 2, 5],
    "nuc_gene_i__LIGAND_": [0, 0, 0],
    "nuc_gene_a__RECEPTOR_": [2, 2, 3],
    "nuc_gene_i__RECEPTOR_": [0, 0, 0],
    "cyt_mrna__LIGAND_": [5, 5, 5],
    "cyt_mrna__RECEPTOR_": [5, 5, 5],
    })

    # fake simulator with dummy methods for class
    dummy_simulator = MagicMock()

    dummy_simulator.load = MagicMock()

    dummy_simulator.modify = MagicMock(return_value=None)

    dummy_simulator.getStateIds = MagicMock(return_value=results_df.keys())

    dummy_simulator.simulate = MagicMock(return_value=results_df)

    return dummy_simulator