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

Local build (optional; CI builds on release):

```bash
python -m build
```

Artifacts land in `dist/`. Upload those only if publishing by hand; never commit the intermediate `build/` tree.

### One-time PyPI Trusted Publishing setup

Publishing uses GitHub OIDC (no long-lived PyPI API token in the repo).

1. On [pypi.org](https://pypi.org) for the **Benchtop** project, add a **Trusted Publisher** of type GitHub with:
   - **Owner** — GitHub user or org that owns the repo
   - **Repository** — `Benchtop`
   - **Workflow** — `publish.yml`
   - **Environment** — `pypi`
2. In the GitHub repository settings, create an Environment named **`pypi`** (optional protection rules / required reviewers).
3. Optionally repeat the same Trusted Publisher setup on [test.pypi.org](https://test.pypi.org) to validate the flow first.

### Releasing a new version

1. Bump `version` in `pyproject.toml` (PyPI rejects re-uploads of an existing version).
2. Commit and merge to `main`.
3. Create and push a matching tag (e.g. `v0.1.1`).
4. Publish a GitHub Release for that tag — this runs `.github/workflows/publish.yml`, which builds the sdist/wheel and uploads to PyPI.
5. Confirm the new version: `pip install Benchtop==X.Y.Z`.

CI builds from the tagged commit; local `dist/` files are not required for the automated path.
