# Command-line interface

Benchtop installs a console script named `benchtop`. The primary subcommand is `experiment`.

```bash
benchtop experiment -p path/to/benchmark.yaml -s tellurium -c 4
```

## Usage

```text
benchtop experiment [options]
```

If no valid subcommand is given, Benchtop prints a short help hint. Use `benchtop experiment --help` for the full flag list.

## Global options

| Flag | Description |
|------|-------------|
| `-v`, `--verbose` | Enable debug logging |
| `-p`, `--path` | Path to a single benchmark YAML (required unless `--run_all`) |
| `-n`, `--name` | Descriptive name for the run (reserved for tooling) |
| `-o`, `--output` | Output directory for the results pickle (default: `results/` next to the YAML) |

## Experiment options

| Flag | Default | Description |
|------|---------|-------------|
| `-s`, `--simulator` | unset | Force one backend for all problems (`tellurium`, `amici`, `bngsim`, `rover`). Overrides per-problem YAML `simulator`; if omitted, each problem uses its YAML value or `tellurium`. |
| `-c`, `--cores` | CPU count | Number of parallel worker processes |
| `--cache_dir` | `./.cache` | Directory for trajectory cache and index |
| `--load_index` | off | Resume incomplete jobs from an existing cache index |
| `--no_confirm` | off | Overwrite an existing cache directory without erroring. Unused when `--load_index` is set. |
| `--No_Observables` | off | Skip `calculate_observables` after simulation |
| `--catchall KEY=VALUE` | — | Extra key=value pairs forwarded into the experiment args namespace |
| `--run_all DIR` | — | Recursively run every `.yaml` / `.yml` under `DIR` |

## Examples

Single benchmark:

```bash
benchtop experiment -p src/tests/data/LR-benchmark.yaml -s tellurium -c 4
```

All YAML files under a directory:

```bash
benchtop experiment --run_all path/to/benchmarks/ -s tellurium
```

Simulate only (no observable pickle):

```bash
benchtop experiment -p path/to/benchmark.yaml --No_Observables
```

Resume after an interrupted run:

```bash
benchtop experiment -p path/to/benchmark.yaml --load_index --cache_dir ./.cache
```

## Exit behavior

- Missing `-p` without `--run_all` raises an assertion error asking for a YAML path.
- Unknown simulator names fail inside `Experiment.run` with the list of registered backends.
- Optional backends that are not installed fail at import with a `pip install benchtop[<name>]` hint.

Simulator selection precedence: explicit `-s` / `Experiment.run(simulator=...)` overrides all problems; otherwise each problem’s YAML `simulator` is used; otherwise `tellurium`.
