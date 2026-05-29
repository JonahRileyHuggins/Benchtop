import importlib

SIMULATOR_REGISTRY = {
            "amici": "wrappers.amici_wrapper.AmiciSimulator",
            "tellurium": "wrappers.tellurium_wrapper.TelluriumSimulator",
}

def load_simulator(name: str):

    try:
        import_path = SIMULATOR_REGISTRY[name]

    except ModuleNotFoundError as e:

        raise ImportError(
            f"The simulator '{name}' requires optional dependencies. "
            f"Install with:\n\n"
            f"pip install benchtop[{name}]"
        ) from e

    module_path, class_name = import_path.rsplit(".", 1)

    module = importlib.import_module(module_path)

    cls = getattr(module, class_name)

    return cls

