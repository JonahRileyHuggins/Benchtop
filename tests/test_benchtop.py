import os
import sys
import json
import shutil
import random
from multiprocessing import Manager

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.benchtop.Experiment import Experiment
from src.benchtop.Worker import Worker
from src.benchtop.AbstractSimulator import AbstractSimulator
from wrappers.tellurium_wrapper import WrapTellurium
from make_dummy import dummy_simulator

def test_run() -> None: 

    assert os.path.basename(os.getcwd()) == 'Benchtop'

    cache_path = './tests/data/.cache'

    if os.path.exists(cache_path):
        shutil.rmtree(cache_path, ignore_errors=True)

    config_path = "./tests/data/LR-benchmark.yaml"
    experiment = Experiment(config_path, cache_dir=cache_path, cores=os.cpu_count(), verbose=True)

    cache_dir = os.path.join(os.path.dirname(config_path), '.cache')
    assert os.path.isdir(cache_dir)
    assert "cache_index.json" in os.listdir(cache_dir)
    assert len(experiment.record.results_dict.keys()) == 9

    sbml_path = os.path.abspath("./tests/data/LR-model.xml")
    experiment.run(WrapTellurium, (sbml_path,), step = 1)

    assert len(os.listdir(cache_dir)) == 10 # 9 simulations + cache index JSON
    for key in experiment.record.results_dict.keys():
        assert key + '.pkl' in os.listdir(cache_dir)

def test_reassigning_all_species() -> None:
    assert os.path.basename(os.getcwd()) == 'Benchtop'

    cache_path = './tests/data/.cache'
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path, ignore_errors=True)

    config_path = "./tests/data/LR-benchmark.yaml"
    experiment = Experiment(config_path, cache_dir=cache_path, cores=1, verbose=False)

    # Construct Worker using MagicMock simulator type
    with Manager() as manager:
        lock = manager.Lock()
        grunt = Worker(
            task=None,
            record=experiment.record,
            simulator=WrapTellurium,
            lock=lock,
            args=(os.path.abspath("./tests/data/LR-model.xml"), ),
            start=0.0,
            step=30.0,
        )

        sbml_path = os.path.abspath("./tests/data/LR-model.xml")
        grunt.simulator = WrapTellurium(sbml_path)

        # Get all species IDs from the model
        species_ids = grunt.simulator.tool.getFloatingSpeciesIds()
        assert len(species_ids) == 9, f"Expected 9 model species, got {len(species_ids)}"
        new_vals = [0.0 for i in range(10)]

        grunt._Worker__setModelState(species_ids, new_vals)

        # Verify all species concentrations are now zero
        for sid in species_ids:
            val = grunt.simulator.tool[sid]
            assert val == 0.0, f"Base tellurium did not reassign {sid} value properly (value={val})"

        print("✅ All 9 model species successfully reassigned to 0.0 without error.")

def test_results_dict_inheritance() -> None:
    """Verifies no method modifies results_dict object after initialization"""
    # 1. Create generic experiment
    assert os.path.basename(os.getcwd()) == 'Benchtop'

    cache_path = './tests/data/.cache'

    if os.path.exists(cache_path):
        shutil.rmtree(cache_path, ignore_errors=True)

    config_path = "./tests/data/LR-benchmark.yaml"
    experiment = Experiment(config_path, cache_dir=cache_path, cores=os.cpu_count(), verbose=True)

    # 2. Save aside simulation identifier keys
    initial_ids = list(experiment.record.results_dict.keys())

    # 3. Run generic experiment
    experiment.run(dummy_simulator, ("foobar",))

    # 4. Inspect difference between different child class results_dict members
    # 4.1 Changes to original member
    assert initial_ids == list(experiment.record.results_dict.keys()), f"Keys were modified during simulation: \
        Initial: {initial_ids}, \t ending: {list(experiment.record.results_dict.keys())}"
    # 4.2 changes to cache_index.json file
    with open(experiment.record.cache.cache_index_path) as f:
        cache_index = json.load(f)
    assert initial_ids == list(cache_index), f"Keys were modified during simulation: \
        Initial: {initial_ids}, \t cache index: {list(cache_index)}"
    # 4.3 changes to the cache object results dictionnary
    assert initial_ids == list(experiment.record.cache.results_dict.keys()), \
    f"Keys were modified during simulation: Initial: {initial_ids}, \t \
        cache results dictionary: {list(experiment.record.cache.results_dict.keys())}"

    print(f"✅ Simulation keys remained unchanged post-initialization")


def test_results_saving() -> None:
    """Verify that results aren't being overwritten by:
     1. randomizing initial values, 2. run a stochastic simulation, 3. assert no final values are the same."""
    assert os.path.basename(os.getcwd()) == 'Benchtop'

    cache_path = './tests/data/.cache'

    if os.path.exists(cache_path):
        shutil.rmtree(cache_path, ignore_errors=True)

    config_path = "./tests/data/LR-benchmark.yaml"
    experiment = Experiment(config_path, cache_dir=cache_path, cores=os.cpu_count(), verbose=True)

    # need a random value for conditions [primary-condition, secondary-condition, adjacent-primary]
    ligand_vals = [random.randint(1,10) for i in range(4)] # one extra for heterogenize condition
    receptor_vals = [random.randint(1,10) for i in range(4)] # one extra for heterogenize condition

    rand_conds_df = experiment.record.problem.condition_files[0]

    for idx in range(len(rand_conds_df)):
        rand_conds_df.loc[idx, 'cyt_prot__LIGAND_'] = ligand_vals[idx]
        rand_conds_df.loc[idx, 'cyt_prot__RECEPTOR_'] = receptor_vals[idx]

    # randomize heterogenize-condition mRNAs
    rand_conds_df.loc[0, "cyt_mrna__LIGAND_"] = random.randint(0,10)
    rand_conds_df.loc[0, "cyt_mrna__RECEPTOR_"] = random.randint(0,10)

    # change tellurium solver to "gillespie" in run:
    sbml_path = os.path.abspath("./tests/data/LR-model.xml")
    experiment.run(WrapTellurium, (sbml_path, "gillespie"), step = 10)

    # Collect final simulation outputs
    results_path = os.path.join(cache_path)
    final_values = []

    for key in experiment.record.results_dict.keys():
        result_file = os.path.abspath(os.path.join(results_path, f"{key}.pkl"))
        if not os.path.exists(result_file):
            raise FileNotFoundError(f"Expected results file not found: {result_file}")

        # Load pickled DataFrame
        data = pd.read_pickle(result_file)
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Expected DataFrame in {result_file}, got {type(data)}")

        # Grab last timepoint (assuming 'time' column exists)
        if "time" not in data.columns:
            raise KeyError(f"'time' column not found in results for {key}")

        last_row = data.sort_values("time").iloc[-1]
        final_values.append(last_row)

    # Convert list of Series → DataFrame for comparison
    verify_df = pd.DataFrame(final_values)

    # Check for duplicate final states
    duplicates = verify_df.duplicated().sum()
    assert duplicates == 0, f"Found {duplicates} duplicate final results; results are being overwritten."

    print(f"✅ {len(verify_df)} unique stochastic results verified — no overwriting detected.")

