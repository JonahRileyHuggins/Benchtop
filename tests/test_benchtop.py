import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.benchtop.Experiment import Experiment
from wrappers.tellurium_wrapper import WrapTellurium

def test_run() -> None: 

    assert os.path.basename(os.getcwd()) == 'Benchtop'


    config_path = "./tests/data/LR-benchmark.yaml"
    experiment = Experiment(config_path, cache_dir='./tests/data/.cache', cores=os.cpu_count(), verbose=True)

    cache_dir = os.path.join(os.path.dirname(config_path), '.cache')
    assert os.path.isdir(cache_dir)
    assert "cache_index.json" in os.listdir(cache_dir)
    assert len(experiment.record.results_dict.keys()) == 12


    sbml_path = os.path.abspath("./tests/data/LR-model.xml")
    experiment.run(WrapTellurium, (sbml_path,), step = 1)

    assert len(os.listdir(cache_dir)) == 13 # 12 simulations + cache index JSON
    for key in experiment.record.results_dict.keys():
        assert key + '.pkl' in os.listdir(cache_dir)


