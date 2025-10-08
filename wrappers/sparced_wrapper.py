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

import amici
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
        # default path for testing
        self.tool.sbml_file = []
        self.tool.model = []
        self.tool.flagD = 1
        # If a nested tuple is passed, unpack it
        for arg in args:
            if type(arg) == str and os.path.exists(arg):

                _, extension = os.path.splitext(arg)

                if extension == ".xml":

                    self.tool.sbml_file = str(pathlib.Path(arg).expanduser().resolve())

            if os.path.isdir(arg):
                model_module = amici.import_model_module("SPARCED", arg)
                #model_module = importlib.import_module(arg)
                self.tool.model = model_module.getModel()
                self.tool.species_initializations = [value for value in self.tool.model.getInitialStates()]

            if type(arg) == int:
                self.tool.flagD = arg

    def getStateIds(self, *args, **kwargs) -> list:
        return self.tool.model.getStateIds()

    def simulate(self, start: float, stop: float, step:float) -> pd.DataFrame:
        """Primary simulation function using hybrid stochastic-deterministic method

        Parameters:

        Returns: 
            - results_dataframe (pd.DataFrame): finalized results of simulation. 
        """

        solver = self.tool.model.getSolver() # Create solver instance
        solver.setMaxSteps = 1e10

        # Iterative time step
        self.tool.model.setTimepoints(np.linspace(0,step,2))

        xoutS_all, xoutG_all, tout_all = RunSPARCED(
            self.tool.flagD,
            stop / 3600,
            self.tool.species_initializations,
            [],
            self.tool.sbml_file,
            self.tool.model
            )
        
        columnsS = [ele for ele in self.tool.model.getStateIds()]
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
        parameter_ids = self.tool.model.getFixedParameterIds()

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
            self.tool.model.setFixedParameters(
                np.array([
                    value if i == comp_idx else self.tool.model.getFixedParameters()[i]
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
