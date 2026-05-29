#!/bin/env python3 
"""
script name: wrap_amici.py
Created on Thurs. 30-08-2025
Author: Jonah R. Huggins

Description: Wrapper method for running experiments with AMICI

Input: Simulation Settings

Output:
    Simulation Results

"""
import os
import pathlib
import logging
from typing import Union


import amici
import numpy as np
import pandas as pd

from AbstractSimulator import AbstractSimulator

logging.basicConfig(
    level=logging.DEBUG, # Overriden if Verbose Arg. True
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AmiciSimulator(AbstractSimulator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self, *args, **kwargs):
        """Constructor method for amici module"""
        module_name = ""
        module_dir = ""
        for arg in args:
            if type(arg) == str and os.path.exists(arg):
                _, extension = os.path.splitext(arg)

                if extension == ".xml":
                    continue
                module_name = arg
                print(module_name)
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
        """Primary simulation function using hybrid stochastic-deterministic method

        Parameters:

        Returns: 
            - results_dataframe (pd.DataFrame): finalized results of simulation. 
        """
        solver = self.tool.model.getSolver()
        solver.setMaxSteps = 1e10

        n_points = int((stop - start) / step)
        results_matrix = np.zeros(shape=(n_points+1,len(self.tool.species_initializations)))
        results_matrix[0,:] = self.tool.species_initializations

        for i in range(n_points):
            states_i = results_matrix[i, :]

            self.tool.model.setTimepoints(np.linspace(start,step,2))
            self.tool.model.setInitialStates(states_i)
            rdata_o4a = amici.runAmiciSimulation(
                 self.tool.model,
                 solver
            )
            results_matrix[i+1, :] = rdata_o4a._swigptr.x[-len(states_i):]

        columnsS = [ele for ele in self.tool.model.getStateIds()]
        results_df = pd.DataFrame(results_matrix, columns=columnsS)
        #dfT = pd.DataFrame({"time": rdata_o4a['t']})
        # Concatenate along columns, aligning by row index (timepoint)
        #results_df = pd.concat([dfT, dfS], axis=1)

        return results_df


    def modify(
            self, 
            component: str, 
            value: Union[int, float]
            ):
        """
        Modify the initial condition or parameter value in the AMICI model.

        Parameters
        ----------
        component : str
            Name of the species or parameter to modify.
        value : int | float
            New value to assign.
        """
        # Retrieve all identifiers
        species_ids = self.tool.model.getStateIds()
        parameter_ids = self.tool.model.getFixedParameterIds()

        # Modify species initializations
        if component in species_ids:
            comp_idx = species_ids.index(component)
            self.tool.species_initializations[comp_idx] = value
            logger.debug(f"Modified species '{component}' (index {comp_idx}) to {value}")
            return

        # Modify parameter values
        elif component in parameter_ids:
            comp_idx = parameter_ids.index(component)
            self.tool.model.setFixedParameters(
                np.array([
                    value if i == comp_idx else self.tool.model.getFixedParameters()[i]
                    for i in range(len(parameter_ids))
                ])
            )
            logger.debug(f"Modified parameter '{component}' (index {comp_idx}) to {value}")
            return

        else:
            raise ValueError(
                f"Component '{component}' not found in model species or parameters.\n"
                f"Available species: {species_ids[:5]}... ({len(species_ids)} total)\n"
                f"Available parameters: {parameter_ids[:5]}... ({len(parameter_ids)} total)"
            )
