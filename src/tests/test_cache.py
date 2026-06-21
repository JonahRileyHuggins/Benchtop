import json
from pathlib import Path

import pytest

from benchtop._results_cacher import ResultCache
from benchtop.experiment import Experiment


def test_cache_constructor(fresh_cache, benchmark_yaml):
    """Stale cache index is replaced when load_index=False."""
    stale = {"id1": {}, "id2": {}, "id3": {}}
    with open(Path(fresh_cache) / "cache_index.json", "w") as f:
        json.dump(stale, f, indent=2)

    experiment = Experiment(
        str(benchmark_yaml),
        cache_dir=fresh_cache,
        cores=2,
        verbose=False,
    )

    loaded_keys = set(experiment.record.cache.job_keys())
    assert not loaded_keys.intersection({"id1", "id2", "id3"})


def test_load_prior(fresh_cache, benchmark_yaml):
    test_reload = {
        "__problems__": {
            "test-benchmark": {"complete": False},
            "test-benchmark-2": {"complete": False},
        },
        "c28f333f-94d6-4d34-a09e-725ab30d4db8": {
            "problem": "test-benchmark",
            "conditionId": "heterogenize",
            "cell": 1,
            "complete": False,
        },
        "b487ea17-9ae7-4176-bc26-7cdc4e983c90": {
            "problem": "test-benchmark",
            "conditionId": "heterogenize",
            "cell": 2,
            "complete": False,
        },
    }

    with open(Path(fresh_cache) / "cache_index.json", "w") as f:
        json.dump(test_reload, f, indent=2)

    experiment = Experiment(
        str(benchmark_yaml),
        cache_dir=fresh_cache,
        cores=2,
        verbose=True,
        load_index=True,
    )

    assert experiment.record.cache.job_keys() == [
        key for key in test_reload if key != "__problems__"
    ]


def test_problem_completion_tracking(fresh_cache, benchmark_yaml):
    experiment = Experiment(
        str(benchmark_yaml),
        cache_dir=fresh_cache,
        cores=2,
        verbose=False,
    )

    cache = experiment.record.cache
    assert cache.is_problem_complete("test-benchmark") is False

    cache.update_problem_status("test-benchmark", True)
    assert cache.is_problem_complete("test-benchmark") is True
    assert cache.is_problem_complete("test-benchmark-2") is False
