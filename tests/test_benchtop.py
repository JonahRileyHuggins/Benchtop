import os
import sys
import shutil
from multiprocessing import Manager

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.benchtop.Experiment import Experiment
from src.benchtop.Worker import Worker
from src.benchtop.AbstractSimulator import AbstractSimulator
from wrappers.tellurium_wrapper import WrapTellurium

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
    assert len(experiment.record.results_dict.keys()) == 12

    sbml_path = os.path.abspath("./tests/data/LR-model.xml")
    experiment.run(WrapTellurium, (sbml_path,), step = 1)

    assert len(os.listdir(cache_dir)) == 13 # 12 simulations + cache index JSON
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


