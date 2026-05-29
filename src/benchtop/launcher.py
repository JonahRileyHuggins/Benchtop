# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Script to automate model-data comparison and simulation complex experiments.

Provide a path to the model directory and the script will run all experiments. 

author: Jonah R. Huggins
"""

# -----------------------Package Import & Defined Arguements-------------------#
import os
import sys
import logging
from typing import List
from types import SimpleNamespace

from benchtop.Experiment import Experiment
from benchtop.arguments import parse_args

logging.basicConfig(
    level=logging.INFO, # Overriden if Verbose Arg. True
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------Function to Run All Experiments-------------------------#

def main():

    args = parse_args()

    if args.command == "Experiment":
        """
        Handle Experiment subcommand. 
        Module to automate model-data comparisons and complex simulations. 
        """
        launch_experiment(args)

    else:
        print("No valid command provided. Use --help for guidance.")


def launch_experiment(args: SimpleNamespace) -> None:
    """
    Launch Experiment submodule
    Input:
        - args.path: Path to PEtab problem configuration file
        - args.cores: the number of cores to use for the simulation
        - args.name: experiment name
        - args.run_all: a flag to run all Experiments

    Output:
        simulation results for all Experiments to a 'results' directory
        within the model directory
    """
    if args.run_all is not None:
        _run_all()
    else:
        assert args.path is not None, "Error: No experiment provided, \
            either provide a PEtab Problem or use the --run_all flag to run all Experiments."
        _run_experiment(args, args.path)


def _run_all(args: SimpleNamespace) -> None:
    """
    Run all Experiments in the provided directory.

    args:
        None

    Returns:
        None
    """
    experiment_list = _get_list_of_experiments(args.run_all)

    for yaml_path in experiment_list:

        assert os.path.exists(yaml_path), f"Error: Experiment {yaml_path} does not exist. verify and try again."

        # Run the Experiment
        _run_experiment(args, yaml_path)


def _run_experiment(args: SimpleNamespace, config_path: str) -> None:
    """
    Run an experiment

    args:
        Experiment (str): The path to the PEtab Problem (.yaml) file.

    Returns:
        None
    """
    assert os.path.exists(config_path), f"Error: Experiment {config_path} does not exist. check the Experiment\
                                    directory and try again."

    # Run the Experiment
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    experiment = Experiment(
        petab_yaml=args.path, 
        cores=args.cores, 
        cache_dir=args.cache_dir, 
        load_index=args.load_index,
        verbose=args.verbose
        )

    experiment.run(args.simulator, args)

    logger.debug("Closed simulation method successfully.")

    if args.No_Observables == True:
        logger.debug("Saved Results successfully.")

    else:
        experiment.observable_calculation(args)
        logger.debug("Ran observableCalc. methods successfully")


def _get_list_of_experiments(directory: str) -> List[str]:
    """
    Recursively searches for YAML files in the provided directory and its subdirectories.

    args:
        directory (str): The root directory to search.

    Returns:
        List[str]: A list of paths to the YAML files found.
    """
    yaml_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.yaml', '.yml')):
                yaml_files.append(os.path.join(root, file))

    return yaml_files

