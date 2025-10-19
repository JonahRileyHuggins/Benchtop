import os 
import sys
import json
import shutil

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.benchtop.ResultsCacher import ResultCache
from src.benchtop.Experiment import Experiment


def test_cache_constructor() -> None:
    """Validating constructor returns as intended"""

    assert os.path.basename(os.getcwd()) == 'Benchtop'

    cache_path = './tests/data/.cache'

    if os.path.exists(cache_path):
        shutil.rmtree(cache_path, ignore_errors=True)
    os.makedirs(cache_path)

    delete_cache = {
        "id1": [],
        'id2': [],
        "id3": []
    }

    with open(f"{cache_path}/cache_index.json", "w") as f:
        json.dump(delete_cache, f, indent=2)

    config_path = "./tests/data/LR-benchmark.yaml"
    experiment = Experiment(config_path, cache_dir=cache_path, cores=os.cpu_count(), verbose=True)

    cache_dir = os.path.join(os.path.dirname(config_path), '.cache')
    assert os.path.isdir(cache_dir)
    assert "cache_index.json" in os.listdir(cache_dir)

    bad_keys = ["id1", "id2", "id3"]

    same_key = False

    for key in experiment.record.cache.results_dict.keys(): 
        if key in bad_keys:
            same_key = True
    
    assert same_key == False, f"Cache not deleted from previous round: {experiment.record.cache.results_dict.keys()}"


def test_load_prior() -> None:

    test_reload = {
        "c28f333f-94d6-4d34-a09e-725ab30d4db8": {
            "conditionId": "heterogenize",
            "cell": 1,
            "complete": False
        },
        "b487ea17-9ae7-4176-bc26-7cdc4e983c90": {
            "conditionId": "heterogenize",
            "cell": 2,
            "complete": False
        },
        "590117d1-9973-4061-b205-24fc96e2ea13": {
            "conditionId": "heterogenize",
            "cell": 3,
            "complete": False
        },
        "35b8eaa5-5b7a-42b5-bb9f-f3405441143e": {
            "conditionId": "primary-condition",
            "cell": 1,
            "complete": False
        },
        "01674711-ec35-463f-9afd-5ddeeb99e39f": {
            "conditionId": "primary-condition",
            "cell": 2,
            "complete": False
        },
        "7279991d-9414-4783-a6f5-ee399559ce97": {
            "conditionId": "primary-condition",
            "cell": 3,
            "complete": False
        },
        "36cd9c60-97ff-4076-8f33-17d5d060430b": {
            "conditionId": "adjacent-primary",
            "cell": 1,
            "complete": False
        },
        "c1ecdd8f-e798-4853-a51f-cb98c7e9d38d": {
            "conditionId": "adjacent-primary",
            "cell": 2,
            "complete": False
        },
        "4ec00d19-6382-4e60-b8f3-f594c66def9c": {
            "conditionId": "adjacent-primary",
            "cell": 3,
            "complete": False
        }
    }

    assert os.path.basename(os.getcwd()) == 'Benchtop'

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cache_path = os.path.join(base_dir, "tests", "data", ".cache")
    print(cache_path)
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path, ignore_errors=True)
    os.makedirs(cache_path)

    with open(f"{cache_path}/cache_index.json", "w") as f:
        json.dump(test_reload, f, indent=2)

    config_path = "./tests/data/LR-benchmark.yaml"
    experiment = Experiment(config_path, cache_dir=cache_path, cores=os.cpu_count(), verbose=True, load_index=True)
    
    tester_keys = list(test_reload.keys())

    results_keys = list(experiment.record.cache.results_dict.keys())

    assert results_keys == tester_keys, f"Experiment did not reload cache index: \
        Original_keys: {tester_keys}, \n Loaded Keys: {results_keys}"
    