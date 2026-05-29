import random
from unittest.mock import MagicMock

import pandas as pd

def dummy_simulator(*args, **kwargs) -> None:

    results_df = pd.DataFrame({
    "cyt_prot__LIGAND_": [random.uniform(0,1000) for _ in range(3)],
    "nuc_gene_a__LIGAND_": [random.uniform(0,1000) for _ in range(3)],
    "nuc_gene_i__LIGAND_": [random.uniform(0,1000) for _ in range(3)],
    "nuc_gene_a__RECEPTOR_": [random.uniform(0,1000) for _ in range(3)],
    "nuc_gene_i__RECEPTOR_": [random.uniform(0,1000) for _ in range(3)],
    "cyt_mrna__LIGAND_": [random.uniform(0,1000) for _ in range(3)],
    "cyt_mrna__RECEPTOR_": [random.uniform(0,1000) for _ in range(3)],
    })

    # fake simulator with dummy methods for class
    dummy_simulator = MagicMock()

    dummy_simulator.load = MagicMock()

    dummy_simulator.modify = MagicMock(return_value=None)

    dummy_simulator.getStateIds = MagicMock(return_value=results_df.keys())

    dummy_simulator.simulate = MagicMock(return_value=results_df)

    return dummy_simulator