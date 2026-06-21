import json
import random
from pathlib import Path

import pandas as pd
import pytest

from benchtop.experiment import Experiment
from benchtop._worker import Worker
from make_dummy import dummy_simulator

try:
    from wrappers.tellurium_wrapper import WrapTellurium
    HAS_TELLURIUM = True
except ImportError:
    HAS_TELLURIUM = False

tellurium = pytest.mark.skipif(not HAS_TELLURIUM, reason="tellurium not installed")

DATA_DIR = Path(__file__).resolve().parent / "data"
BENCHMARK_YAML = DATA_DIR / "LR-benchmark.yaml"
SBML_PATH = DATA_DIR / "LR-model.xml"

# 3 conditions × (1 cell + 3 cells) across two problems
EXPECTED_JOB_COUNT = 12


def test_experiment_initializes_all_problem_jobs(fresh_cache, benchmark_yaml):
    experiment = Experiment(
        str(benchmark_yaml),
        cache_dir=fresh_cache,
        cores=2,
        verbose=False,
    )

    assert (Path(fresh_cache) / "cache_index.json").is_file()
    assert len(experiment.record.cache.job_keys()) == EXPECTED_JOB_COUNT

    problems_meta = experiment.record.cache.results_dict["__problems__"]
    assert problems_meta["test-benchmark"]["complete"] is False
    assert problems_meta["test-benchmark-2"]["complete"] is False


def test_run_all_problems(fresh_cache, benchmark_yaml):
    experiment = Experiment(
        str(benchmark_yaml),
        cache_dir=fresh_cache,
        cores=2,
        verbose=False,
    )
    experiment.run(dummy_simulator)

    cache_dir = Path(fresh_cache)
    assert len(list(cache_dir.glob("*.pkl"))) == EXPECTED_JOB_COUNT

    for key in experiment.record.cache.job_keys():
        assert (cache_dir / f"{key}.pkl").is_file()
        assert experiment.record.cache.results_dict[key]["complete"] is True

    problems_meta = experiment.record.cache.results_dict["__problems__"]
    assert problems_meta["test-benchmark"]["complete"] is True
    assert problems_meta["test-benchmark-2"]["complete"] is True


@tellurium
def test_reassigning_all_species(fresh_cache, benchmark_yaml):
    experiment = Experiment(
        str(benchmark_yaml),
        cache_dir=fresh_cache,
        cores=1,
        verbose=False,
    )

    grunt = Worker(
        task=None,
        record=experiment.record,
        simulator=WrapTellurium,
        args=(str(SBML_PATH),),
        start=0.0,
        step=30.0,
    )

    grunt.simulator = WrapTellurium(str(SBML_PATH))

    species_ids = grunt.simulator.tool.getFloatingSpeciesIds()
    assert len(species_ids) == 9
    new_vals = [0.0 for _ in species_ids]

    grunt._set_model_state(species_ids, new_vals)

    for sid in species_ids:
        assert grunt.simulator.tool[sid] == 0.0


@tellurium
def test_param_reassignment(fresh_cache, benchmark_yaml):
    experiment = Experiment(
        str(benchmark_yaml),
        cache_dir=fresh_cache,
        cores=1,
        verbose=False,
    )

    grunt = Worker(
        task=None,
        record=experiment.record,
        simulator=WrapTellurium,
        args=(str(SBML_PATH),),
        start=0.0,
        step=30.0,
    )

    grunt.simulator = WrapTellurium(str(SBML_PATH))

    param_ids = grunt.simulator.tool.getGlobalParameterIds()
    assert len(param_ids) == 15
    new_vals = [1.0 for _ in param_ids]

    grunt._set_model_state(param_ids, new_vals)

    for sid in param_ids:
        assert grunt.simulator.tool[sid] == 1.0


def test_results_dict_inheritance(fresh_cache, benchmark_yaml):
    experiment = Experiment(
        str(benchmark_yaml),
        cache_dir=fresh_cache,
        cores=2,
        verbose=False,
    )

    initial_ids = experiment.record.cache.job_keys()
    experiment.run(dummy_simulator)

    assert initial_ids == experiment.record.cache.job_keys()

    with open(experiment.record.cache.cache_index_path) as f:
        cache_index = json.load(f)

    assert initial_ids == [
        key for key in cache_index if key != "__problems__"
    ]


def test_results_saving(fresh_cache, benchmark_yaml):
    experiment = Experiment(
        str(benchmark_yaml),
        cache_dir=fresh_cache,
        cores=2,
        verbose=False,
    )

    rand_conds_df = experiment.record.problem.condition_files[0]

    for idx in range(len(rand_conds_df)):
        rand_conds_df.loc[idx, "cyt_prot__LIGAND_"] = 0
        rand_conds_df.loc[idx, "cyt_prot__RECEPTOR_"] = 0
        rand_conds_df.loc[idx, "nuc_gene_i__LIGAND_"] = 0
        rand_conds_df.loc[idx, "nuc_gene_a__LIGAND_"] = 0
        rand_conds_df.loc[idx, "nuc_gene_i__RECEPTOR_"] = 0
        rand_conds_df.loc[idx, "cyt_prot__LIGAND__RECEPTOR_"] = 0

    experiment.run(dummy_simulator)

    final_values = []
    for key in experiment.record.cache.job_keys():
        data = pd.read_pickle(Path(fresh_cache) / f"{key}.pkl")
        assert isinstance(data, pd.DataFrame)
        assert "time" in data.columns
        final_values.append(data.sort_values("time").iloc[-1])

    verify_df = pd.DataFrame(final_values)
    assert verify_df.duplicated().sum() == 0


def test_problems_run_sequentially(fresh_cache, benchmark_yaml, monkeypatch):
    """Each problem is marked complete in config order after its jobs finish."""
    call_log = []
    from benchtop._results_cacher import ResultCache

    original = ResultCache.update_problem_status

    def wrapped(self, problem_name, complete):
        call_log.append((problem_name, complete))
        return original(self, problem_name, complete)

    monkeypatch.setattr(ResultCache, "update_problem_status", wrapped)

    experiment = Experiment(
        str(benchmark_yaml),
        cache_dir=fresh_cache,
        cores=2,
        verbose=False,
    )
    experiment.run(dummy_simulator)

    assert call_log == [
        ("test-benchmark", True),
        ("test-benchmark-2", True),
    ]
