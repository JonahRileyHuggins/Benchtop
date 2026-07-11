"""Simulator backend lookup: name → wrapper class."""

import importlib

SIMULATOR_REGISTRY = {
    "amici": "wrappers.amici_wrapper.AmiciSimulator",
    "tellurium": "wrappers.tellurium_wrapper.TelluriumSimulator",
    "bngsim": "wrappers.bngsim_wrapper.BNGSimSimulator",
    "rover": "wrappers.rover_wrapper.RoverSimulator",
}


def load_simulator(name: str):
    """Import and return a simulator class by registry name."""
    if name not in SIMULATOR_REGISTRY:
        raise KeyError(name)

    import_path = SIMULATOR_REGISTRY[name]
    module_path, class_name = import_path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ImportError(
            f"The simulator '{name}' requires optional dependencies. "
            f"Install with: pip install benchtop[{name}]"
        ) from e

    return getattr(module, class_name)
