from wrappers.amici_wrapper import AmiciSimulator
from wrappers.tellurium_wrapper import TelluriumSimulator

SIMULATOR_REGISTRY = {
            "amici": AmiciSimulator,
            "tellurium": TelluriumSimulator,
}
