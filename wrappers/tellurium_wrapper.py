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

import pandas as pd
import tellurium as te

from AbstractSimulator import AbstractSimulator

logging.basicConfig(
    level=logging.DEBUG, # Overriden if Verbose Arg. True
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WrapTellurium(AbstractSimulator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self, *args, **kwargs):
        # default path for testing
        sbml_path = "LR-model.xml"
        solver = "rk45"
        # If a nested tuple is passed, unpack it
        for arg in args:
            logger.debug(f"Interpreting argument: {arg}")
            if type(arg) == str and os.path.exists(arg):

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
        """Primary simulation function using hybrid stochastic-deterministic method

        Parameters:

        Returns: 
            - results_dataframe (pd.DataFrame): finalized results of simulation. 
        """

        n_points = int(((stop+step) - start) / step)

        results_array = self.tool.simulate(
            start=float(start),
            end=float(stop+step), 
            points=n_points
            )

        # Clean column headers: remove surrounding square brackets
        column_headers = [col.strip("[]") for col in results_array.colnames]

        results_df = pd.DataFrame(results_array, columns=column_headers)

        return results_df

    def modify(
            self, 
            component: str, 
            value: int | float
            ):
        """
        Method for SingleCell simulator modify method
        """
        logger.debug(f"Assigning model state variable {component} to value {value}  ({type(value)})")
        try:
            self.tool[component] = float(value)
        except ValueError as e:
            raise ValueError(f"Error in setting parameter value: {e}")