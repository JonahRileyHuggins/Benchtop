#!/bin/env python3 
"""
script name: SingleCell.py
Created on Thurs. 2025-04-12 20:29:00
Author: Jonah R. Huggins

Description: [**PROTOTYPE**] entrypoint for model simulation. Parsing and wrapper \n
function that takes model attributes, performs basic data handling, and passes \n
initial model state array to C++ simulation engine  

Input: Simulation Settings

Output:
    Simulation Results

"""
# -----------------------Package Import & Defined Arguements-------------------#
import os
import sys
import importlib.util

import pandas as pd

sys.path.append(f'{os.path.dirname(os.path.dirname(__file__))}/src/benchtop/')
from AbstractSimulator import AbstractSimulator

# Absolute path to compiled extension (pySingleCell*.so file)
so_path = os.path.join(
    os.getenv("SINGLECELL_PATH"),
    "build",
    "pySingleCell.cpython-312-x86_64-linux-gnu.so"
)

if not os.path.isfile(so_path):
    raise FileNotFoundError(f"Could not find pySingleCell shared object at: {so_path}")

# Load the module from the given path
spec = importlib.util.spec_from_file_location("pySingleCell", so_path)
pySingleCell = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pySingleCell)

# Access class symbols, SC so python doesn't confuse with wrapper class
SC = pySingleCell.SingleCell

#-------------------Class Definition-----------------------------------------#
class SingleCell(AbstractSimulator):
    """Wrapper Class for C++ interface at pybind exposed methods"""

    def __init__(self, *args, **kwargs):
        """
        Populates self.tool with custom module, enables extensibility in 
        experiment framework
        """
        super().__init__(*args, **kwargs)

    def load(self,*args, **kwargs) -> SC:
        """Meets parent class loader method for SingleCell loader"""
        
        self.tool = SC(*args)

    def getStateIds(self, *args, **kwargs) -> list:
        return self.tool.getGlobalSpeciesIds()

    def simulate(self, start, stop, step) -> pd.DataFrame:
        """Primary simulation function using hybrid stochastic-deterministic method

        Parameters:

        Returns: 
            - results_dataframe (pd.DataFrame): finalized results of simulation. 
        """

        results_array = self.tool.simulate(
            start,
            stop, 
            step
            )

        speciesIds = self.tool.getGlobalSpeciesIds()

        results_df = pd.DataFrame(results_array, columns=speciesIds)

        return results_df

    def modify(
            self, 
            component: str, 
            value: int | float
            ):
        """
        Method for SingleCell simulator modify method
        """
def modify(
        self, 
        component: str, 
        value: int | float
    ):
    """
    Method for SingleCell simulator modify method with debug/error checks
    """

    # Check type of component
    if not isinstance(component, str):
        raise TypeError(f"Expected component to be str, got {type(component)}: {component}")

    # Check type of value
    if not isinstance(value, (int, float)):
        raise TypeError(f"Expected value to be int or float, got {type(value)}: {value}")

    # Optionally check for NaN or Inf
    if isinstance(value, float):
        if value != value:  # NaN check
            raise ValueError(f"Value for {component} is NaN")
        if value == float('inf') or value == float('-inf'):
            raise ValueError(f"Value for {component} is infinite: {value}")

    try:
        self.tool.modify(component, float(value))
    except Exception as e:
        raise RuntimeError(f"Failed to modify component '{component}' with value {value}") from e


