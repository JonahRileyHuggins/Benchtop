#!/bin/env python3 
"""
Primary class object of an experiment

author: Jonah R. Huggins
"""
# =========================================
# ============ Package Import ============
# =========================================
import os
import logging
import argparse
import pickle as pkl
from datetime import date
import multiprocessing as mp

from src.benchtop.Worker import Worker
from src.benchtop.Manager import Manager
from src.benchtop.Organizer import Organizer
import src.benchtop.ObservableCalculator as obs
from src.benchtop.ResultsCacher import ResultCache
from src.benchtop.file_loader import FileLoader
from src.benchtop.AbstractSimulator import AbstractSimulator


logging.basicConfig(
    level=logging.INFO, # Overriden if Verbose Arg. True
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Experiment:

    def __init__(self, 
                 petab_yaml: os.PathLike | str, 
                 cores: int = os.cpu_count(), 
                 verbose = False
                 ) -> None:
        """
        Class object describing a single experiment. 

        Parameters
        ----------
        petab_yaml : str, required
            path to PEtab formated experiment
        
        cores : int, optional
            number of cores to allocate to benchmarking for parallel performance

        """

        self.cache = ResultCache()
        self.org = Organizer(cores)
        self.size = cores
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        self.petab_yaml = os.path.abspath(petab_yaml)

        if not os.path.exists(self.petab_yaml):
            raise FileNotFoundError(f"{self.petab_yaml} is not a valid benchmark")

        # Load the details of the experiment
        # !DotDict Notation! Loader contains configuration file and PEtab files.
        self.loader = FileLoader(petab_yaml)
        self.loader._petab_files()

        self.details = self.loader.config

        self.name = self.details.problems[0].name or None

        self.cell_count = getattr(self.details.problems[0], "cell_count", 1)

        logger.info("Loading Experiment %s details from %s", self.name, self.petab_yaml)

        # SingleCell() constructor is variadic, supply multiple SBML files!
        self.sbml_list = self.__sbml_getter()

        # Loads jobs directory with results_dict class member
        self.manager = Manager(self.loader.problems[0])


    def run(self,
            simulator: AbstractSimulator,
            *args,
            ) -> None:
        """
        Parameters
        ----------
        simulator : AbstractSimulator
            child class of abstract AbstractSimulator Class, defined as a
            wrapper for a particular simulator

        args : tuple, optional
            Extra arguments to pass to function.
        """

        logger.debug(f"Starting in-silico experiment across {self.size} cores.")

        num_rounds, job_directory = self.org.task_organization(
            self.loader.problems[0].measurement_files[0],
            self.cell_count
        )

        for round_i in range(num_rounds):

            # Get list of tasks for current round:
            tasks = self.org.task_assignment(
                rank_jobs_directory=job_directory,
                round_i=round_i
            )

            worker_args = [
                (
                    task, 
                    self.manager,
                    simulator, 
                    *args
                ) 
                for task in tasks]
            
            # split workload across processes:
            with mp.Pool(processes=os.cpu_count()) as pool:
                pool.starmap(Worker, worker_args)
                        
        # Have root store final results of all sims and cleanup cache
        self.__store_final_results()

    def __sbml_getter(self) -> list:
        """Retrieves all sbml files defined in PEtab configuration file"""
        sbml_file_list = [
            fp
            for problem in self.loader.problems
            if hasattr(problem, "sbml_files")
            for fp in problem.sbml_files
        ]

        return sbml_file_list

    def __store_final_results(self) -> None:
        """Stores all simulation results stored in cache into Rank 0 self.results_dict object"""

        for key in self.manager.results_dict.keys(): 
            
            condition_id = self.manager.results_dict[key]['conditionId']
            cell = self.manager.results_dict[key]['cell']

            df = self.manager.results_lookup(condition_id, cell)

            for column in df.columns:

                self.manager.results_dict[key][column] = df[column]
                
        return # saves results to results_dict class member

    def save_results(self, args) -> None:
        """Save the results of the simulation to a file
        input:
            None
        output:
            returns the saved results as a nested dictionary within
            a pickle file
        """

        # Benchmark results are stored within the specified model directory

        results_directory = os.path.join(os.path.dirname(self.petab_yaml), "results")

        if args is not None and 'output' in args:

            results_directory = args.output

        if not os.path.exists(results_directory):
            os.makedirs(results_directory)

        # Final output is saved in pickle format
        results_path = os.path.join(results_directory, f"{date.today()}.pkl")

        if self.name is not None:
            results_path = os.path.join(results_directory, f"{self.name}.pkl")

        with open(results_path, "wb") as f:
            pkl.dump(self.manager.results_dict, f)

        self.cache.delete_cache()


    def observable_calculation(self, *args) -> None:
        """Calculate the observables and compare to the experimental data.
        input:
            results: dict - results of the SPARCED model unit test simulation
        output:
            returns the results of the SPARCED model unit test simulation
        """
        self.manager.results_dict = obs.ObservableCalculator(self).run()

        self.save_results(args)

        return # Proceeds to next command provided in launchers.py



if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='Experiment')
    parser.add_argument('--path', '-p', default = None, help = 'path to configuration file detailing \
                                                                            which files to inspect for name changes.')
    parser.add_argument('--name', '-n', default = 'Deterministic', help = "String-type name of model")
    parser.add_argument('--cores', '-c', default=os.cpu_count(), help = "Number of processes to divide tasks across")
    parser.add_argument('--catchall', metavar='KEY=VALUE', nargs='*',
                        help="Catch-all arguments passed as key=value pairs")
    parser.add_argument('-v', '--verbose', help="Be verbose", action="store_true", dest="verbose")
    parser.add_argument('--output', '-o', default = ".", help  = "path to which you want output files stored")
    parser.add_argument(
        '--observables',
        action='store_false',
        help='Enable downsampling of data'
    )

    args = parser.parse_args()
   
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    experiment = Experiment(args.path, args.cores)

    experiment.run()

    logger.debug("Closed simulation method successfully.")

    if args.No_Observables == False:
        experiment.save_results(args)
        logger.debug("Saved Results successfully.")

    else:
        experiment.observable_calculation(args)
        logger.debug("Ran observableCalc. methods successfully")

    
