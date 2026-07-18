# Examples

## LR receptor benchmark

The repository ships a small PEtab-style example under `src/tests/data/`:

| File | Role |
|------|------|
| `LR-benchmark.yaml` | Problem list, `cell_count`, paths to TSV/SBML |
| `conditions.tsv` | Condition IDs and parameter/species overrides |
| `measurements.tsv` | Observable ↔ condition links, times, optional preequilibration |
| `observables.tsv` | `observableId` and `observableFormula` |
| `LR Model - Parameters.tsv` | Nominal parameters |
| `LR-model.xml` | SBML model |

Example YAML:

```yaml
format_version: 1
parameter_file: LR Model - Parameters.tsv
problems:
  - name: 'test-benchmark'
    condition_files:
      - conditions.tsv
    measurement_files:
      - measurements.tsv
    observable_files:
      - observables.tsv
    sbml_files:
      - LR-model.xml
    cell_count: 1
```

`cell_count` is a Benchtop extension: number of replicate simulations per condition (useful for stochastic models).

### Command line

From the repository root (with Benchtop installed):

```bash
benchtop experiment -p src/tests/data/LR-benchmark.yaml -s tellurium -c 4
```

Results land in `src/tests/data/results/LR-benchmark.pkl` by default. Cache files go to `./.cache` unless you pass `--cache_dir`.

### Python

```python
from benchtop import Experiment

experiment = Experiment("src/tests/data/LR-benchmark.yaml", cores=4)
experiment.run("tellurium")
experiment.calculate_observables()
```

A notebook sketch lives in `demo/in-silico-experiment-demo.ipynb` (API names there may lag the package; prefer `Experiment.run` + `calculate_observables` as above).

## Benchmark file roles

| File | Purpose |
|------|---------|
| **Parameter TSV** | Nominal model parameters |
| **conditions.tsv** | Rows of overrides; columns (except `conditionId` / `conditionName`) are passed to `modify()` |
| **measurements.tsv** | Sets simulation horizon (`time`), links observables, optional `preequilibrationConditionId` |
| **observables.tsv** | Formulas over species column names in the trajectory |
| **SBML** | One or more model files listed under `sbml_files` |

When `preequilibrationConditionId` is set, Benchtop runs that condition first and seeds the dependent simulation from its final species state.

## Observables and time alignment

By default, Benchtop evaluates each `observableFormula` on the full trajectory, then **downsamples to the measurement times** in `measurements.tsv` so simulation and experiment vectors share the same time grid.

That matches classic time-series comparisons. It is **not** the right tool for population metrics such as “fraction of cells whose *maximum* of A+B+C exceeds a threshold.” For those, keep a full trajectory observable (empty/`None` measurements if needed) and compute the threshold / population statistic in post-processing. See also {doc}`writing_wrappers` and {doc}`developers` for extension points.

## Hybrid models (two SBML files)

Backends such as Rover expect **two** partitioned SBML files. List them in order under `sbml_files`:

```yaml
sbml_files:
  - deterministic-interactions.xml   # BNGsim partition
  - stochastic-gene-expression.xml   # StochMod partition
```

Then:

```bash
benchtop experiment -p path/to/benchmark.yaml -s rover -c 4
```

Species overrides in `conditions.tsv` are molecule counts for Rover’s shared store.
