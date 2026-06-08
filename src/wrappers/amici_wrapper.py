"""AMICI simulator wrapper for compiled SBML models."""

import logging
import os
from typing import Union

import amici
import numpy as np
import pandas as pd

from benchtop._abstract_simulator import AbstractSimulator

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AmiciSimulator(AbstractSimulator):
    """Step-wise AMICI integration with species/parameter modification."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self, *args, **kwargs):
        module_name = ""
        module_dir = ""

        for arg in args:
            if isinstance(arg, str) and os.path.exists(arg):
                _, extension = os.path.splitext(arg)
                if extension == ".xml":
                    continue
                module_name = arg
            if os.path.isdir(arg):
                module_dir = arg

        model_module = amici.import_model_module(module_name, module_dir)
        self.tool.model = model_module.getModel()

        species_initializations = np.array(self.tool.model.getInitialStates())
        species_initializations[np.argwhere(species_initializations <= 1e-6)] = 0.0
        self.tool.species_initializations = species_initializations

    def getStateIds(self, *args, **kwargs) -> list:
        return self.tool.model.getStateIds()

    def simulate(self, start, stop, step) -> pd.DataFrame:
        solver = self.tool.model.getSolver()
        solver.setMaxSteps = 1e10

        n_points = int((stop - start) / step)
        n_species = len(self.tool.species_initializations)
        results_matrix = np.zeros(shape=(n_points + 1, n_species))
        results_matrix[0, :] = self.tool.species_initializations

        for i in range(n_points):
            states_i = results_matrix[i, :]
            self.tool.model.setTimepoints(np.linspace(start, step, 2))
            self.tool.model.setInitialStates(states_i)
            rdata = amici.runAmiciSimulation(self.tool.model, solver)
            results_matrix[i + 1, :] = rdata._swigptr.x[-len(states_i):]

        columns = list(self.tool.model.getStateIds())
        return pd.DataFrame(results_matrix, columns=columns)

    def modify(self, component: str, value: Union[int, float]) -> None:
        species_ids = self.tool.model.getStateIds()
        parameter_ids = self.tool.model.getFixedParameterIds()

        if component in species_ids:
            comp_idx = species_ids.index(component)
            self.tool.species_initializations[comp_idx] = value
            logger.debug("Modified species '%s' to %s", component, value)
            return

        if component in parameter_ids:
            comp_idx = parameter_ids.index(component)
            self.tool.model.setFixedParameters(
                np.array([
                    value if i == comp_idx else self.tool.model.getFixedParameters()[i]
                    for i in range(len(parameter_ids))
                ])
            )
            logger.debug("Modified parameter '%s' to %s", component, value)
            return

        raise ValueError(
            f"Component '{component}' not found in model species or parameters."
        )
