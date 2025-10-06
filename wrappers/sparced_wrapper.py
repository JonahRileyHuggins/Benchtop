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

import numpy as np
import pandas as pd

from Experiment.AbstractSimulator import AbstractSimulator
from modules.RunSPARCED import RunSPARCED



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
        self.tool.sbml_path = []
        self.tool.model = []
        self.tool.flagD = 1
        # If a nested tuple is passed, unpack it
        for arg in args:
            if type(arg) == str and os.path.exists(arg):

                _, extension = os.path.splitext(arg)

                if extension == ".xml":

                    self.tool.sbml_path = str(pathlib.Path(arg).expanduser().resolve())

            if os.path.isdir(arg):

                model_module = importlib.import_module(arg)
                self.tool.model = model_module.getModel()
                self.tool.species_initializations = self.tool.model.getInitialStates()

            if type(arg) == int:
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
        Method for SingleCell simulator modify method
        """

        # Make a list of species and get component's index
        list_of_species = self.getStateIds()
        comp_idx = list_of_species.index(component)

        # get list of states and modify state[index] value
        self.tool.species_initializations[comp_idx] = value
