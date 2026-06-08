"""Tellurium (libRoadRunner) simulator wrapper."""

import logging
import os

import pandas as pd
import tellurium as te

from benchtop._abstract_simulator import AbstractSimulator

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TelluriumSimulator(AbstractSimulator):
    """Load SBML via Tellurium; integrate with CVODE or Gillespie."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self, *args, **kwargs):
        solver = "cvode"
        sbml_path = None

        for arg in args:
            logger.debug("Interpreting argument: %s", arg)
            if isinstance(arg, str) and os.path.exists(arg):
                _, extension = os.path.splitext(arg)
                if extension == ".xml":
                    sbml_path = arg
            if arg == "gillespie":
                solver = arg

        self.tool = te.loadSBMLModel(sbml_path)
        self.tool.setIntegrator(solver)
        integrator = self.tool.getIntegrator()
        integrator.absolute_tolerance = 1e-8
        integrator.relative_tolerance = 1e-6
        integrator.maximum_bisect = 10
        integrator.max_steps = 1e6

    def simulate(self, start, stop, step) -> pd.DataFrame:
        n_points = int(((stop + step) - start) / step)

        results_array = self.tool.simulate(
            start=float(start),
            end=float(stop + step),
            points=n_points,
        )

        column_headers = [col.strip("[]") for col in results_array.colnames]
        return pd.DataFrame(results_array, columns=column_headers)

    def modify(self, component: str, value: int | float) -> None:
        logger.debug("Setting %s = %s", component, value)
        try:
            self.tool[component] = float(value)
        except ValueError as e:
            raise ValueError(f"Error setting parameter {component}: {e}") from e
