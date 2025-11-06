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

sys.path.append(os.path.dirname(__file__))
from Experiment import Experiment

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from wrappers.SingleCell import SingleCell
logging.basicConfig(
    level=logging.INFO, # Overriden if Verbose Arg. True
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------Function to Run All Experiments-------------------------#

class Experimentalist:

    def __init__(self, args: SimpleNamespace):
        self.args = args
        self.launch_experiment()

    def launch_experiment(self) -> None:
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
        if self.args.run_all is not None:
            self.run_all()
        else:
            assert self.args.path is not None, "Error: No experiment provided, \
                either provide a Experiment or use the --run_all flag to run all Experiments."
            self.run_experiment(self.args.path)


    def run_all(self) -> None:
        """
        Run all Experiments in the provided directory.

        self.args:
            None

        Returns:
            None
        """
        experiment_list = self._get_list_of_experiments(self.args.run_all)

        for yaml_path in experiment_list:

            assert os.path.exists(yaml_path), f"Error: Experiment {yaml_path} does not exist. verify and try again."

            # Run the Experiment
            self.run_experiment(yaml_path)


    def run_experiment(self, config_path: str) -> None:
        """
        Run an experiment

        self.args:
            Experiment (str): The path to the Experiment YAML file.

        Returns:
            None
        """
        assert os.path.exists(config_path), f"Error: Experiment {config_path} does not exist. check the Experiment\
                                        directory and try again."

        # Run the Experiment
        if self.args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        experiment = Experiment(
            petab_yaml=self.args.path, 
            cores=self.args.cores, 
            cache_dir=self.args.cache_dir, 
            load_index=self.args.load_index,
            verbose=self.args.verbose
            )

        experiment.run(SingleCell, *self.args.sbml)

        logger.debug("Closed simulation method successfully.")

        if self.args.No_Observables == True:
            experiment.save_results(self.args)
            logger.debug("Saved Results successfully.")

        else:
            experiment.observable_calculation(self.args)
            logger.debug("Ran observableCalc. methods successfully")


    def _get_list_of_experiments(self, directory: str) -> List[str]:
        """
        Recursively searches for YAML files in the provided directory and its subdirectories.

        self.args:
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

