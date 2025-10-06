#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun. 17/11/2024 00:18AM - JRH

Description: This script is the main method for constructing an AMICI model
using SPARCED input files. The first step is to build the model. Note that
building the model may take several minutes. There are 3 main utilities described here:
1. Model Compilation
2. Experiment Simulation
3. Benchmarking

The script is executed by running the following command in the terminal:
    $ SingleCell Build
    $ SingleCell Simulate
    $ SingleCell Experiment
    $ SingleCell -h
"""

#-----------------------Package Import & Defined Arguements--------------------# 
import sys
sys.path.append('../')
from singlecell.shared_utils.arguments import parse_args

def main():
    """Main entry point."""
    args = parse_args()

    if args.command == "Build":
        """
        Handle the compile subcommand. This script is the method for constructing an \
        AMICI model using SPARCED input files.
        """
        from singlecell.ModelBuilding.launcher import Builder
        Builder(args)

    elif args.command == "Simulate":
        """ 
        Handle Simulate subcommand.
        Run a single simulation with a set of conditions.
        """
        from singlecell.Simulate.SingleCell import SingleCell
        SingleCell(args).simulate()

    elif args.command == "Experiment":
        """
        Handle Experiment subcommand. 
        Module to automate model-data comparisons and complex simulations. 
        """
        from singlecell.Experiment.launcher import Experimentalist
        Experimentalist(args)

    elif args.command == "Tool":
        """
        Handle misc. tools specified in tools subdirectory
        """
        from singlecell.tools.launcher import ToolBelt
        ToolBelt(args=args)

    else:
        print("No valid command provided. Use --help for guidance.")

if __name__ == "__main__":
    main()
