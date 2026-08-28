# Writing a simulator wrapper

Benchtop is simulator-agnostic: workers only call three methods on an {py:class}`benchtop._abstract_simulator.AbstractSimulator` subclass. Benchmark semantics (conditions, measurements, observables) stay in PEtab files.

## Interface

```python
from benchtop._abstract_simulator import AbstractSimulator

class MySimulator(AbstractSimulator):
    def load(self, args, **kwargs):
        """Load/compile the model. args.model_paths is set by Experiment."""

    def modify(self, component: str, value: float) -> None:
        """Set a parameter or species before simulation."""

    def simulate(self, start: float, stop: float, step: float):
        """Integrate and return a pandas.DataFrame trajectory."""
```

The constructor calls `load(*args, **kwargs)` immediately. Prefer:

```python
def __init__(self, args, **kwargs):
    super().__init__(args, **kwargs)
```

## Contract checklist

1. **`args.model_paths`** — List of SBML paths from the YAML `sbml_files` entry. Most engines use `args.model_paths[0]`. Hybrid engines (e.g. Rover) require two paths: deterministic then stochastic.
2. **`modify(name, value)`** — Column headers from `conditions.tsv` (except `conditionId` / `conditionName`) are passed here as strings. Names must match model parameter or species IDs.
3. **`simulate(start, stop, step)`** — Return a `pandas.DataFrame` with a `time` column and one column per species ID referenced by `observableFormula`. Workers compute `stop` from the maximum measurement time for the condition.
4. **Units** — Whatever your engine expects (e.g. concentrations for BNGSim/Tellurium; molecule counts for Rover species overrides). Document this for users of your wrapper.

## Minimal skeleton

```python
"""My backend simulator wrapper."""

import logging
import pandas as pd

from benchtop._abstract_simulator import AbstractSimulator

logger = logging.getLogger(__name__)


class MySimulator(AbstractSimulator):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

    def load(self, args, **kwargs):
        # Build engine handle from args.model_paths[0] (or more paths)
        self.tool = ...

    def modify(self, component: str, value: int | float) -> None:
        logger.debug("Setting %s = %s", component, value)
        # set parameter or species on self.tool

    def simulate(self, start, stop, step) -> pd.DataFrame:
        # integrate; return DataFrame with columns ["time", ...species...]
        ...
```

Place the module under `src/wrappers/`, for example `src/wrappers/my_wrapper.py`.

## Register the backend

Add an entry in `src/benchtop/registry.py`:

```python
SIMULATOR_REGISTRY = {
    ...
    "mysim": "wrappers.my_wrapper.MySimulator",
}
```

Then:

```bash
benchtop experiment -p path/to/benchmark.yaml -s mysim
```

Or pass the class directly:

```python
from wrappers.my_wrapper import MySimulator

experiment.run(MySimulator)
```

If the engine needs optional dependencies, document an extras install (e.g. `pip install benchtop[mysim]`) and let the registry import fail with a clear `ImportError` (existing pattern for AMICI).

## Reference wrappers

| File | Notes |
|------|--------|
| `wrappers/bngsim_wrapper.py` | Default; branches param vs species; explicit `time` column |
| `wrappers/amici_wrapper.py` | Compiled AMICI module path handling |
| `wrappers/tellurium_wrapper.py` | Single SBML; `self.tool[component] = value` |
| `wrappers/rover_wrapper.py` | Two SBML paths; hybrid ODE + tau-leap |

## Observables vs wrappers

Wrappers do **not** declare observables. After trajectories are cached, `ObservableCalculator` evaluates formulas from `observables.tsv` using species column names. Keep formulas as arithmetic over those names; function names like `max(...)` are currently parsed as species identifiers and will fail. Population-level metrics belong in post-processing, not in the wrapper.
