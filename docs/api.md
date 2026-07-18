# Python API

The public entry point is {py:class}`benchtop.experiment.Experiment`, also exported as:

```python
from benchtop import Experiment
```

Benchtop does not require you to import organizer, worker, or cache internals for normal use.

## `Experiment`

```python
Experiment(
    petab_yaml,
    cores=os.cpu_count(),
    cache_dir="./.cache",
    load_index=False,
    verbose=False,
)
```

| Parameter | Description |
|-----------|-------------|
| `petab_yaml` | Path to the benchmark YAML file |
| `cores` | Number of parallel worker processes |
| `cache_dir` | Directory for trajectory pickles and the cache index |
| `load_index` | If `True`, `run()` resumes incomplete jobs from the existing index |
| `verbose` | Enable debug logging |

On construction, Benchtop loads the YAML, resolves referenced TSV/SBML paths, and builds the job record for all problems in the file.

### `run(simulator, *args, start=0.0, step=30.0)`

Schedule and execute simulations for every incomplete problem.

| Argument | Description |
|----------|-------------|
| `simulator` | Registry name (`"tellurium"`, `"amici"`, `"bngsim"`, `"rover"`) **or** an `AbstractSimulator` subclass |
| `*args` | Optional namespace (e.g. CLI `args`) passed into workers; SBML paths are injected as `model_paths` |
| `start` | Simulation start time (default `0.0`) |
| `step` | Output / coupling step size (default `30.0`) |

Workers apply condition overrides via `modify()`, optionally seed state from a completed preequilibration run, then call `simulate(start, stop, step)` where `stop` is the maximum measurement time for that condition.

### `calculate_observables(*args)`

For each problem, evaluate `observableFormula` expressions against cached trajectories, downsample to measurement times, and write a results pickle next to the benchmark (or to `args.output` when provided). Clears the simulation cache afterward via `save_results`.

### `resume(simulator, *args, start=0.0, step=30.0)`

Re-run only incomplete cache entries. Invoked automatically from `run()` when `load_index=True`.

### `save_results(args)`

Serialize the current results dictionary to `results/<benchmark_name>.pkl` (or `args.output`) and delete the cache.

## Minimal example

```python
from benchtop import Experiment

experiment = Experiment(
    "src/tests/data/LR-benchmark.yaml",
    cores=4,
    cache_dir="./.cache",
    verbose=True,
)

experiment.run("tellurium")
experiment.calculate_observables()
```

Pass a wrapper class instead of a registry name:

```python
from wrappers.tellurium_wrapper import TelluriumSimulator

experiment.run(TelluriumSimulator)
```

## Simulator registry

Registered backends (see `benchtop.registry.SIMULATOR_REGISTRY`):

| Name | Wrapper |
|------|---------|
| `tellurium` | `TelluriumSimulator` (default dependency) |
| `amici` | `AmiciSimulator` (optional `.[amici]`) |
| `bngsim` | `BNGSimSimulator` |
| `rover` | `RoverSimulator` (hybrid; needs two SBML partitions) |

Unknown names raise `ValueError` listing available keys. Missing optional packages raise `ImportError` with an install hint.

## Results shape

After `calculate_observables()`, the pickle is a nested dictionary roughly:

```text
{
  "<problem_name>": {
    "<job_key>": {
      "problem": ...,
      "conditionId": ...,
      "cell": ...,
      "<observableId>": {
        "experiment": array,
        "simulation": array,
        "time": array,
      },
      ...
    },
    ...
  },
  ...
}
```

Raw trajectories (before observables) live in the cache directory as per-job pickle files plus a JSON index.

## Related

- {doc}`writing_wrappers` — implement `AbstractSimulator` for a new engine
- {doc}`cli` — same workflow from the command line
- {doc}`examples` — LR receptor benchmark walkthrough
