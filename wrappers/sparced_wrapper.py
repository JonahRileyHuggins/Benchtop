#!/bin/env python3 
"""
script name: tellurium-wrapper.py
Created on Thurs. 30-08-2025
Author: Jonah R. Huggins

Description: Wrapper method for running experiments with tellurium-solver

Input: Simulation Settings

Output:
    Simulation Results

"""
import os
import pathlib
import logging
import importlib
import importlib.util

import numpy as np
import pandas as pd

import sys

sys.path.append('../')
sys.path.append('../bin/')
from src.benchtop.AbstractSimulator import AbstractSimulator
from bin.modules.RunSPARCED import RunSPARCED



logging.basicConfig(
    level=logging.INFO, # Overriden if Verbose Arg. True
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WrapSPARCED(AbstractSimulator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self, *args, **kwargs):
        self.tool.sbml_path = []
        self.tool.model = []
        self.tool.flagD = 1

        for arg in args:
            if isinstance(arg, str) and os.path.exists(arg):
                _, extension = os.path.splitext(arg)

                if extension == ".xml":
                    self.tool.sbml_path = str(pathlib.Path(arg).expanduser().resolve())

                elif os.path.isdir(arg):
                    model_path = pathlib.Path(arg).expanduser().resolve() / "RunSPARCED.py"
                    if not model_path.exists():
                        raise FileNotFoundError(f"No RunSPARCED.py found in {arg}")

                    spec = importlib.util.spec_from_file_location("RunSPARCED", model_path)
                    model_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(model_module)

                    self.tool.model = model_module.getModel()
                    self.tool.species_initializations = self.tool.model.getInitialStates()

            elif isinstance(arg, int):
                self.tool.flagD = arg

    def getStateIds(self, *args, **kwargs) -> list:
        return self.tool.model.getStateIds()

    def simulate(self, stop) -> pd.DataFrame:
        """Primary simulation function using hybrid stochastic-deterministic method

        Parameters:

        Returns: 
            - results_dataframe (pd.DataFrame): finalized results of simulation. 
        """

        solver = self.model.getSolver() # Create solver instance
        solver.setMaxSteps = 1e10

        xoutS_all, xoutG_all, tout_all = RunSPARCED(
            self.tool.flagD,
            stop,
            self.tool.species_initializations,
            [],
            self.tool.sbml_file,
            self.tool.model
            )
        
        columnsS = [ele for ele in self.toolmodel.getStateIds()]
        columnsG = [x for n, x in enumerate(columnsS) if x.startswith('m_')]
        resa = [sub.replace('m_', 'ag_') for sub in columnsG]
        resi = [sub.replace('m_', 'ig_') for sub in columnsG]
        columnsG2 = np.concatenate((resa, resi), axis=None)

        dfS = pd.DataFrame(xoutS_all, columns=columnsS)
        dfG = pd.DataFrame(xoutG_all, columns=columnsG2)
        dfT = pd.DataFrame({"time": tout_all})

        # Concatenate along columns, aligning by row index (timepoint)
        results_df = pd.concat([dfT, dfS, dfG], axis=1)

        return results_df
    def modify(
            self, 
            component: str, 
            value: int | float
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
        parameter_ids = self.tool.model.getParameterIds()

        # Modify species initializations
        if component in species_ids:
            comp_idx = species_ids.index(component)
            logger.info(f"Modifying species '{component}' (index {comp_idx}) to {value}")
            self.tool.species_initializations[comp_idx] = value
            return

        # Modify parameter values
        elif component in parameter_ids:
            comp_idx = parameter_ids.index(component)
            logger.info(f"Modifying parameter '{component}' (index {comp_idx}) to {value}")
            self.tool.model.setParameters(
                np.array([
                    value if i == comp_idx else self.tool.model.getParameters()[i]
                    for i in range(len(parameter_ids))
                ])
            )
            return

        else:
            raise ValueError(
                f"Component '{component}' not found in model species or parameters.\n"
                f"Available species: {species_ids[:5]}... ({len(species_ids)} total)\n"
                f"Available parameters: {parameter_ids[:5]}... ({len(parameter_ids)} total)"
            )
