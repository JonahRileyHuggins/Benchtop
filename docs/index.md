# Benchtop

**Simulator-agnostic parallel execution of simulation experiments in PEtab format.**

Benchtop runs in-silico benchmarks defined with a YAML configuration and companion TSV/SBML files. It schedules conditions in dependency order, distributes simulations across CPU cores, caches trajectories, and evaluates observables for model–data comparison—without tying you to a single ODE or hybrid engine.

## What it does

- Load a [PEtab](https://petab.readthedocs.io/)-inspired experiment (conditions, measurements, observables, parameters, SBML).
- Simulate each condition × cell replicate in parallel via pluggable simulator wrappers.
- Cache raw trajectories and resume incomplete runs.
- Compute observable formulas and align them with experimental measurement times.

**Pipeline:** Benchmark YAML → FileLoader → Experiment → Organizer (task order) → worker pool → simulator wrapper → ResultCache → ObservableCalculator → results pickle.

## Features

- **PEtab-style benchmarks** — YAML plus standard TSV files for conditions, measurements, observables, and parameters.
- **Parallel execution** — Round-robin worker pool across processes.
- **Preequilibration** — Topological sort so preequilibration conditions run before dependents.
- **Single-cell replication** — Multiple stochastic replicates per condition via `cell_count`.
- **Result caching** — Pickle trajectories and a JSON index; resume with `--load_index`.
- **Pluggable simulators** — BNGsim by default; optional AMICI, Tellurium, Rover, or your own wrapper.

## Installation

Requires Python ≥ 3.11.

```bash
pip install benchtop
```

From a source checkout:

```bash
pip install -e .
# optional backends
pip install -e ".[amici]"
# documentation tools
pip install -e ".[docs]"
```

This installs the `benchtop` CLI entry point.

## Quick links

```{toctree}
:maxdepth: 2

api
examples
cli
writing_wrappers
developers
```

## Authors

- Jonah Huggins — JonahRileyHuggins@gmail.com
- Marc Birtwistle — marc.birtwistle@gmail.com

## License

GPL-2.0 — see the `LICENSE` file in the repository.
