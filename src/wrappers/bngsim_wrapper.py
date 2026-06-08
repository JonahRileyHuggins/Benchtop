"""BNGSim ODE simulator wrapper."""

import logging

import bngsim
import numpy as np
import pandas as pd

from benchtop._abstract_simulator import AbstractSimulator

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BNGSimSimulator(AbstractSimulator):
    """Load SBML via BNGSim; integrate with codegen ODE solver."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

    def load(self, args, **kwargs):
        self.tool = bngsim.Model.from_sbml(args.model_paths[0])

    def simulate(self, start, stop, step) -> pd.DataFrame:
        n_points = int(((stop + step) - start) / step)
        sim = bngsim.Simulator(self.tool, method="ode", codegen=True)
        result = sim.run(
            t_span=(float(start), float(stop + step)),
            n_points=n_points,
        )
        result_stack = np.column_stack((result.time, result.species))
        return pd.DataFrame(
            result_stack,
            columns=["time"] + result.species_names,
        )

    def modify(self, component: str, value: int | float) -> None:
        logger.debug("Setting %s = %s", component, value)
        try:
            if component in self.tool.param_names:
                self.tool.set_param(component, float(value))
            elif component in self.tool.species_names:
                self.tool.set_concentration(component, float(value))
            else:
                raise ValueError(f"Component '{component}' not found in model")
        except ValueError as e:
            raise ValueError(f"Error setting {component}: {e}") from e
