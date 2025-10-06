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
import importlib.util

import pandas as pd

from Experiment.AbstractSimulator import AbstractSimulator

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
        super().__init__(args, kwargs)

    def load(self,*args, **kwargs) -> SC:
        """Meets parent class loader method for SingleCell loader"""
        
        sbml_list = []

        for sbml_path in args:
            sbml_list.append(sbml_path)
        
        self.tool(*sbml_list)

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
        self.tool.modify(component, float(value))


if __name__ == '__main__':
    import sys
    import argparse
    from Experiment.file_loader import FileLoader

    # Arguement Parsing (Internal For Now)
    parser = argparse.ArgumentParser(description='Basic script for running single simulations with the SPARCED model')

    parser.add_argument('--sbmls', help="Path to sbml file(s)")

    parser.add_argument('--modify', '-m', metavar='KEY=VALUE', nargs='+',
                        help='Species to modify in key=value format', default=[])
    parser.add_argument('--start', help = 'start time in seconds for simulation', default = 0.0)
    parser.add_argument('--stop', help = 'stop time for simulation.', default = 86400.0)
    parser.add_argument('--step', help = 'step size of each iteration in the primary for-loop.', default = 30.0)
    parser.add_argument('--output', help = 'output path', default="singlecell_results.tsv")

    args = parser.parse_args()

    try: 
        
        single_cell = SingleCell(*args.sbmls)

    except FileNotFoundError:
        print("Invalid sbml path supplied")
        sys.exit(0)

    if 'modify' in args.__dict__.keys():
        for pair in args.modify:
            if '=' in pair:
                key, val = pair.split('=', 1)
                print("Setting %s to value %d", key, float(val))
                single_cell.modify(key, float(val))

    results_df = single_cell.simulate(start=args.start, stop=args.stop, step=args.step)
    
    results_df.to_csv(args.output, sep='\t', index=False)
