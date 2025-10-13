#!/bin/env python3 
"""
Primary class object of an experiment. Runs via embarassingly parallel
simulation, where each process recieves an individual task in round-robin 
agorithm. Results are dumped into hidden cache directory and serialized
as pickle files. Job ordering is organized via Kahn's algorithm. 

author: Jonah R. Huggins
"""
# =========================================
# ============ Package Import ============
# =========================================
import os
import sys
import logging
import argparse
import pickle as pkl
from datetime import date
from typing import Union
import multiprocessing as mp

sys.path.append(os.path.dirname(__file__))
from Worker import worker_method
from Record import Record
from Organizer import Organizer
import ObservableCalculator as obs
from file_loader import FileLoader
from AbstractSimulator import AbstractSimulator


logging.basicConfig(
    level=logging.INFO, # Overriden if Verbose Arg. True
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Experiment:

    def __init__(self, 
                 petab_yaml: Union[os.PathLike, str], 
                 cores: int = os.cpu_count(),
                 cache_dir: str = './.cache',
                 load_index: bool = False,
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

        # recieves one or more SBML files
        self.sbml_list = self.__sbml_getter()

        # Loads jobs directory with results_dict class member
        self.record = Record(
            problem=self.loader.problems[0],
            cache_dir=cache_dir,
            load_index=load_index
            )


    def run(self,
            simulator: AbstractSimulator,
            *args, 
            start: float = 0.0,
            step: float = 30.0,
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

        num_rounds, job_index = self.org.task_organization(
            self.loader.problems[0].measurement_files[0],
            self.cell_count
        )

        for round_i in range(num_rounds):

            # Get list of tasks for current round:
            tasks = self.org.task_assignment(
                rank_jobs_directory=job_index,
                round_i=round_i
            )

            logger.debug(f"Tasks for round: {tasks}")

            worker_args = [
                (
                    task, 
                    self.record,
                    simulator,
                    *args,
                    start,
                    step
                ) 
                for task in tasks]
            
            # split workload across processes:
            with mp.Pool(processes=self.size) as pool:
                pool.starmap(worker_method, worker_args)
                        
        # Have root store final results of all sims and cleanup cache
        self._store_final_results()

    def __sbml_getter(self) -> list:
        """Retrieves all sbml files defined in PEtab configuration file"""
        sbml_file_list = [
            fp
            for problem in self.loader.problems
            if hasattr(problem, "sbml_files")
            for fp in problem.sbml_files
        ]

        return sbml_file_list

    def _store_final_results(self) -> None:
        """Stores all simulation results stored in cache into Rank 0 self.results_dict object"""

        for key in self.record.results_dict.keys(): 
            
            condition_id = self.record.results_dict[key]['conditionId']
            cell = self.record.results_dict[key]['cell']

            df = self.record.results_lookup(condition_id, cell)

            for column in df.columns:

                self.record.results_dict[key][column] = df[column]
                
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
            pkl.dump(self.record.results_dict, f)

        self.record.cache.delete_cache()


    def observable_calculation(self, *args) -> None:
        """Calculate the observables and compare to the experimental data.
        input:
            results: dict - results of the SPARCED model unit test simulation
        output:
            returns the results of the SPARCED model unit test simulation
        """
        self.record.results_dict = obs.ObservableCalculator(self).run()

        self.save_results(args)

        return # Proceeds to next command provided in launchers.py

    def resume(
        self,
        simulator: AbstractSimulator,
        *args, 
        start: float = 0.0,
        step: float = 30.0,
    ) -> None:
        """Starts Experiment from last completed simulation setting"""
        
        cache_index = self.record.cache.read_cache_index()

        # --- 1. Identify incomplete jobs ---
        incomplete = [
            f"{self.record.results_dict[key]['conditionId']}+{self.record.results_dict[key]['cell']}"
            for key in cache_index.keys()
            if not cache_index[key]['complete']
        ]

        # --- 2. Find all pre-simulations (unique topo order) ---
        topo_sorted = self.org.topologic_sort(
            measurements_df=self.loader.problems[0].measurement_files[0]
        )

        # --- 3. Retrieve all simulations + cell replicates ---
        total_tasks = self.org.total_tasks(
            tasks=topo_sorted,
            cell_count=self.cell_count
        )

        # --- 4. Filter total_tasks to only include incomplete ones ---
        incomplete_set = set(incomplete)
        total_tasks = [task for task in total_tasks if task in incomplete_set]

        # --- 5. Guard clause: if nothing to resume ---
        if not total_tasks:
            logger.info(f"No incomplete jobs found for experiment '{self.name}'. Nothing to resume.")
            return

        # --- 6. Handle zero-order dependency logic ---
        delayed_tasks = self.org.delay_secondary_conditions(
            measurements_df=self.loader.problems[0].measurement_files[0],
            task_list=total_tasks,
            cell_count=self.cell_count
        )

        logger.info(f"Resuming {len(total_tasks)} jobs for experiment '{self.name}'...")

        # --- 7. Rebuild task index for parallel scheduling ---
        num_rounds = -(-len(delayed_tasks) // self.size)  # Ceiling division

        for round_idx in range(num_rounds):
            tasks = []
            for _ in range(self.size):
                if delayed_tasks:
                    tasks.append(delayed_tasks.pop(0))
                else:
                    tasks.append(None)

            logger.debug(f"Tasks for round {round_idx + 1}/{num_rounds}: {tasks}")

            worker_args = [
                (
                    task, 
                    self.record,
                    simulator,
                    *args,
                    start,
                    step
                ) 
                for task in tasks
            ]

            # --- 8. Parallel execution ---
            with mp.Pool(processes=self.size) as pool:
                pool.starmap(worker_method, worker_args)

            logger.debug(f"Completed round {round_idx + 1}/{num_rounds}")

        # --- 9. Store final results and cleanup ---
        self._store_final_results()

                    

        

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

    
