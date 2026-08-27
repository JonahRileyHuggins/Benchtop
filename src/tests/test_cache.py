import json
from pathlib import Path

import pandas as pd
import pytest

from benchtop._results_cacher import ResultCache
from benchtop.experiment import Experiment
from make_dummy import dummy_simulator


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
        no_confirm=True,
    )

    loaded_keys = set(experiment.record.cache.job_keys())
    assert not loaded_keys.intersection({"id1", "id2", "id3"})


def test_existing_cache_requires_no_confirm(fresh_cache, benchmark_yaml):
    """An existing cache index is not overwritten unless no_confirm=True."""
    with open(Path(fresh_cache) / "cache_index.json", "w") as f:
        json.dump({"id1": {}}, f, indent=2)

    with pytest.raises(FileExistsError, match="Cache index already exists"):
        Experiment(
            str(benchmark_yaml),
            cache_dir=fresh_cache,
            cores=2,
            verbose=False,
        )


def test_load_prior_merges_with_config(fresh_cache, benchmark_yaml):
    """Loaded index is merged with the full benchmark job list."""
    partial_index = {
        "__problems__": {
            "test-benchmark": {"complete": False},
            "test-benchmark-2": {"complete": False},
        },
        "c28f333f-94d6-4d34-a09e-725ab30d4db8": {
            "problem": "test-benchmark",
            "conditionId": "heterogenize",
            "cell": 1,
            "complete": True,
        },
        "b487ea17-9ae7-4176-bc26-7cdc4e983c90": {
            "problem": "test-benchmark",
            "conditionId": "primary-condition",
            "cell": 1,
            "complete": False,
        },
    }

    with open(Path(fresh_cache) / "cache_index.json", "w") as f:
        json.dump(partial_index, f, indent=2)

    experiment = Experiment(
        str(benchmark_yaml),
        cache_dir=fresh_cache,
        cores=2,
        verbose=False,
        load_index=True,
    )

    assert len(experiment.record.cache.job_keys()) == 12
    assert experiment.record.cache.results_dict[
        "c28f333f-94d6-4d34-a09e-725ab30d4db8"
    ]["complete"] is True
    assert set(experiment.record.incomplete_tasks_for_problem("test-benchmark")) == {
        "primary-condition+1",
        "adjacent-primary+1",
    }


def test_load_index_resumes_incomplete_jobs(fresh_cache, benchmark_yaml):
    """load_index=True should resume only incomplete jobs via run()."""
    experiment = Experiment(
        str(benchmark_yaml),
        cache_dir=fresh_cache,
        cores=2,
        verbose=False,
    )

    completed_key = None
    for key in experiment.record.cache.job_keys():
        entry = experiment.record.cache.results_dict[key]
        if (
            entry["problem"] == "test-benchmark"
            and entry["conditionId"] == "heterogenize"
            and entry["cell"] == 1
        ):
            completed_key = key
            break

    assert completed_key is not None
    experiment.record.cache.update_cache_index(completed_key, True)
    experiment.record.cache.save(
        completed_key,
        pd.DataFrame({"time": [0.0], "value": [1.0]}),
    )

    resumed = Experiment(
        str(benchmark_yaml),
        cache_dir=fresh_cache,
        cores=2,
        verbose=False,
        load_index=True,
    )

    assert resumed.load_index is True
    assert set(resumed.record.incomplete_tasks_for_problem("test-benchmark")) == {
        "adjacent-primary+1",
        "primary-condition+1",
    }

    resumed.run(dummy_simulator)

    for key in resumed.record.cache.job_keys():
        assert resumed.record.cache.results_dict[key]["complete"] is True

    reloaded = pd.read_pickle(Path(fresh_cache) / f"{completed_key}.pkl")
    assert reloaded.loc[0, "value"] == 1.0


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


def test_resume_skips_completed_problem(fresh_cache, benchmark_yaml):
    """Resume should skip problems marked complete and finish the rest."""
    experiment = Experiment(
        str(benchmark_yaml),
        cache_dir=fresh_cache,
        cores=2,
        verbose=False,
    )

    experiment.record.cache.update_problem_status("test-benchmark", True)

    incomplete_before = [
        key
        for key in experiment.record.cache.job_keys()
        if experiment.record.cache.results_dict[key]["problem"] == "test-benchmark-2"
        and not experiment.record.cache.results_dict[key]["complete"]
    ]
    assert len(incomplete_before) == 9

    experiment.run(dummy_simulator)

    for key in experiment.record.cache.job_keys():
        entry = experiment.record.cache.results_dict[key]
        if entry["problem"] == "test-benchmark-2":
            assert entry["complete"] is True
        elif entry["problem"] == "test-benchmark":
            assert entry["complete"] is False

    assert experiment.record.cache.is_problem_complete("test-benchmark") is True
    assert experiment.record.cache.is_problem_complete("test-benchmark-2") is True
