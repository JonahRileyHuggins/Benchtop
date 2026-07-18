# Developers

Notes for contributing to Benchtop itself. For adding a simulation engine, see {doc}`writing_wrappers`.

## Repository layout

```text
Benchtop/
├── .github/workflows/     # CI
├── .readthedocs.yaml      # Read the Docs build
├── docs/                  # Sphinx documentation (this site)
├── demo/                  # Notebook walkthroughs
├── src/
│   ├── benchtop/          # Core library
│   ├── wrappers/          # Simulator backends
│   └── tests/             # Pytest suite + LR example data
├── pyproject.toml
└── README.md
```

Core modules (under `src/benchtop/`):

| Module | Role |
|--------|------|
| `experiment.py` | Orchestrator: load, schedule, run, observables |
| `_organizer.py` | Topological task order and round-robin assignment |
| `_worker.py` | Per-process simulate + cache write |
| `_record.py` / `_results_cacher.py` | Job keys, pickle cache, index |
| `_observable_calculator.py` | Formula eval and time alignment |
| `_abstract_simulator.py` | Wrapper ABC |
| `file_loader.py` | YAML + TSV loading |
| `registry.py` | Name → wrapper class |
| `launcher.py` / `arguments.py` | CLI |

## Editable install

```bash
pip install -e ".[dev]"
# documentation
pip install -e ".[docs]"
```

Optional simulator extras are declared under `[project.optional-dependencies]` in `pyproject.toml`.

## Tests

```bash
pytest
# or
cd src/tests && python run_tests.py
```

Tests use the LR receptor files in `src/tests/data/` and exercise experiment execution, caching, workers, organizers, records, and observables.

## Building documentation locally

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`. Read the Docs builds from `.readthedocs.yaml` on each push once the GitHub project is connected.

## What not to commit

| Path | Why |
|------|-----|
| `build/`, `dist/`, `*.egg-info/` | setuptools / packaging artifacts |
| `docs/_build/` | Generated HTML |
| `.cache/`, `cache*.json`, `*.pkl` | Local simulation cache and results |

These patterns are listed in `.gitignore`.

## Cache and results conventions

- **Cache** — Per-job trajectory pickles plus a JSON index under `--cache_dir` (default `./.cache`). Use `--load_index` / `load_index=True` to resume.
- **Results** — After observables, a pickle is written under `results/` next to the benchmark YAML (or `-o` / `args.output`). The simulation cache is then deleted.

## Packaging

```bash
python -m build
```

Upload artifacts from `dist/` only; never publish the intermediate `build/` tree to GitHub.
