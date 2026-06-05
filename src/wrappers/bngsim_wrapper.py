#!/bin/env python3 
"""
script name: bngsim_wrapper.py
Created on Thurs. 29-05-2026
Author: Jonah R. Huggins

Description: Wrapper method for running experiments with bngsim-simulator

Input: Simulation Settings

Output:
    Simulation Results

"""
import os
import pathlib
import logging

import bngsim
import numpy as np
import pandas as pd

from benchtop._abstract_simulator import AbstractSimulator

logging.basicConfig(
    level=logging.DEBUG, # Overriden if Verbose Arg. True
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BNGSimSimulator(AbstractSimulator):

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

    def load(self, args, **kwargs):
        """
        bngsim object class constructor
        """
        self.tool = bngsim.Model.from_sbml(args.model_paths[0])

    def simulate(self, start, stop, step) -> pd.DataFrame:
        """Primary simulation function using BNGSim

        Parameters:

        Returns: 
            - results_df (pd.DataFrame): results of simulation. 
        """

        n_points = int(((stop+step) - start) / step)
        sim = bngsim.Simulator(self.tool, method="ode", codegen=True)
        result = sim.run(t_span=(float(start), float(stop+step)), n_points=n_points)
        result_stack = np.column_stack((result.time, result.species))
        results_df = pd.DataFrame(result_stack, columns= [["time"] + result.species_names])
        #results_df = pd.DataFrame(result.species, columns=result.species_names)

        return results_df

    def modify(
            self, 
            component: str, 
            value: int | float
            ):
        """
        Method for simulator modify method
        """
        logger.debug(f"Assigning model state variable {component} to value {value}  ({type(value)})")
        
        try:
            if component in self.tool.param_names:
                self.tool.set_param(component, float(value))
            elif component in self.tool.species_names:
                self.tool.set_concentration(component, float(value))
            else:
                raise ValueError(f"component not found")
        except ValueError as e:
            raise ValueError(f"Error in setting component {component} value: {e}")


