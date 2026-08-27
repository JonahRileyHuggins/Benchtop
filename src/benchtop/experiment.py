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
import pickle as pkl
from datetime import date
from typing import Union
import multiprocessing as mp

from benchtop._worker import worker_method
from benchtop._record import Record
from benchtop._results_cacher import DEFAULT_CACHE
from benchtop.registry import load_simulator, SIMULATOR_REGISTRY
from benchtop._organizer import Organizer
from benchtop._observable_calculator import ObservableCalculator
from benchtop.file_loader import FileLoader
from benchtop._abstract_simulator import AbstractSimulator

logging.basicConfig(
    level=logging.INFO, # Overriden if Verbose Arg. True
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)

class Experiment:

    def __init__(self, 
                 petab_yaml: Union[os.PathLike, str], 
                 cores: int = os.cpu_count(),
                 cache_dir: Union[os.PathLike, str] = DEFAULT_CACHE,
                 load_index: bool = False,
                 verbose = False,
                 no_confirm: bool = False
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

        self.details = self.loader.config ### slate to remove self-storage

        # add one or more SBML files
        self.sbml_list = self.__sbml_getter()
        
        # Loads jobs directory with results_dict class member
        self.record = Record(
            problems=self.loader.problems,
            cache_dir=cache_dir,
            load_index=load_index,
            no_confirm=no_confirm
            )

        self.load_index = load_index
        self.name = os.path.splitext(os.path.basename(self.petab_yaml))[0]
        self.cell_count = self.loader.problems[0].cell_count
        self.default_simulator = "tellurium"

    def run(self,
            simulator: str | AbstractSimulator = None,
            *args, 
            start: float = 0.0,
            step: float = 30.0,
            ) -> None:
        """
        Parameters
        ----------
        simulator : AbstractSimulator
            child class of abstract AbstractSimulator Class, defined as a
            wrapper for a particular simulator. When provided, overrides
            per-problem YAML ``simulator`` for all problems.

        args : tuple, optional
            Extra arguments to pass to function.
        """

        if self.load_index:
            return self.resume(
                simulator,
                *args,
                start=start,
                step=step,
            )

        for problem_index, config_problem in enumerate(self.details.problems):
            problem = self.loader.problems[problem_index]
            problem_name = problem.name

            if self.record.cache.is_problem_complete(problem_name):
                logger.info("Skipping completed problem '%s'", problem_name)
                continue

            self.name = problem_name
            self.cell_count = problem.cell_count
            self.record.set_current_problem(problem, problem_name)
            resolved_simulator = self._simulator_for_problem(problem, simulator)
            worker_args_base = self.__add_sbml_to_args(args, problem)

            logger.info(
                "Running problem '%s' (%d/%d) from %s",
                problem_name,
                problem_index + 1,
                len(self.loader.problems),
                self.petab_yaml,
            )
            logger.debug(
                "Starting in-silico experiment %s across %d cores.",
                self.name,
                self.size,
            )

            num_rounds, job_index = self.org.task_organization(
                problem.measurement_files[0],
                self.cell_count
            )

            for round_i in range(num_rounds):
                tasks = self.org.task_assignment(
                    rank_jobs_directory=job_index,
                    round_i=round_i
                )

                logger.debug("Tasks for round: %s", tasks)

                worker_args = [
                    (
                        task, 
                        self.record,
                        resolved_simulator,
                        worker_args_base,
                        start,
                        step, 
                    ) 
                    for task in tasks]
                
                with mp.Pool(processes=self.size) as pool:
                    pool.starmap(worker_method, worker_args)
                            
                self.__update_cache_for_round(tasks)

            self._finalize_problem(problem_name)

    def _simulator_for_problem(
        self,
        problem,
        override: str | AbstractSimulator | None,
    ) -> AbstractSimulator:
        """Resolve simulator: explicit override > problem YAML > default."""
        if override is not None:
            chosen = override
            resolved = self._resolve_simulator(override)
        else:
            chosen = getattr(problem, "simulator", None) or self.default_simulator
            resolved = self._resolve_simulator(chosen)
        
        return resolved

    def _resolve_simulator(
        self, simulator: str | AbstractSimulator
    ) -> AbstractSimulator:
        if isinstance(simulator, str):
            try:
                return load_simulator(simulator)
            except KeyError:
                raise ValueError (
                    f"Unknown simulator '{simulator}'. "
                    f"Avaliable: {list(SIMULATOR_REGISTRY)}"
                )

        if callable(simulator):
            return simulator

        raise TypeError(
            "simulator must be a registered name or a callable simulator wrapper"
        )

    def __update_cache_for_round(self, task_list: list) -> None:
        """Receives task list for current round,
        splits task into conditionID and cell number,
        updates results_dict[complete] with True."""
        
        remaining = []

        for task in task_list:
            if task is None:
                continue

            condition_id, cell = task.split("+")
            key = self.record.find_job_key(condition_id, cell)

            if key is None:
                remaining.append(task)
                continue

            self.record.cache.update_cache_index(key=key, status=True)

        assert remaining == [], f"Error in simulation task updates: {remaining}"

    def __add_sbml_to_args(self, args: tuple, problem=None) -> tuple:
        """Adds this problem's SBML paths to args (or all paths if no problem)."""
        if not args:
            from types import SimpleNamespace
            args_nsp = SimpleNamespace()
        else:
            args_nsp = args[0]

        if problem is not None and hasattr(problem, "sbml_files"):
            args_nsp.model_paths = list(problem.sbml_files)
        else:
            args_nsp.model_paths = self.sbml_list
        return args_nsp

    def __sbml_getter(self) -> list:
        """Retrieves all sbml files defined in PEtab configuration file"""
        sbml_file_list = [
            fp
            for problem in self.loader.problems
            if hasattr(problem, "sbml_files")
            for fp in problem.sbml_files
        ]
        
        return sbml_file_list

    def save_results(self, args) -> None:
        """Save the results of the simulation to a file
        input:
            None
        output:
            returns the saved results as a nested dictionary within
            a pickle file
        """

        results_directory = os.path.join(os.path.dirname(self.petab_yaml), "results")

        if args is not None and hasattr(args, "output") and args.output:
            results_directory = args.output

        if not os.path.exists(results_directory):
            os.makedirs(results_directory)

        benchmark_name = os.path.splitext(os.path.basename(self.petab_yaml))[0]
        results_path = os.path.join(results_directory, f"{benchmark_name}.pkl")

        job_results = {
            key: self.record.cache.results_dict[key]
            for key in self.record.cache.job_keys()
        }

        with open(results_path, "wb") as f:
            pkl.dump(job_results, f)

        self.record.cache.delete_cache()

    def calculate_observables(self, *args) -> None:
        """Calculate observables and compare to experimental data for all problems."""
        combined_results = {}

        for problem_index, problem in enumerate(self.loader.problems):
            problem_name = problem.name
            self.record.set_current_problem(problem, problem_name)

            problem_results = ObservableCalculator(self).run()
            combined_results[problem_name] = problem_results

        self.record.cache.results_dict = combined_results
        self.save_results(args[0] if args else None)

        return

    def resume(
        self,
        simulator: str | AbstractSimulator = None,
        *args, 
        start: float = 0.0,
        step: float = 30.0,
    ) -> None:
        """Starts Experiment from last completed simulation setting"""
        resumed_any = False

        for problem_index, problem in enumerate(self.loader.problems):
            problem_name = problem.name

            if self.record.cache.is_problem_complete(problem_name):
                logger.info("Skipping completed problem '%s'", problem_name)
                continue

            self.name = problem_name
            self.cell_count = problem.cell_count
            self.record.set_current_problem(problem, problem_name)
            resolved_simulator = self._simulator_for_problem(problem, simulator)
            worker_args_base = self.__add_sbml_to_args(args, problem)

            incomplete = self.record.incomplete_tasks_for_problem(problem_name)

            topo_sorted = self.org.topologic_sort(
                measurements_df=problem.measurement_files[0]
            )

            total_tasks = self.org.total_tasks(
                tasks=topo_sorted,
                cell_count=self.cell_count
            )

            incomplete_set = set(incomplete)
            total_tasks = [task for task in total_tasks if task in incomplete_set]

            if not total_tasks:
                self.record.cache.update_problem_status(problem_name, True)
                logger.info(
                    "No incomplete jobs for problem '%s'; marking complete.",
                    problem_name,
                )
                continue

            resumed_any = True
            delayed_tasks = self.org.delay_secondary_conditions(
                measurements_df=problem.measurement_files[0],
                task_list=total_tasks,
                cell_count=self.cell_count
            )

            logger.info(
                "Resuming %d jobs for problem '%s'...",
                len(total_tasks),
                problem_name,
            )

            num_rounds = -(-len(delayed_tasks) // self.size)

            for round_idx in range(num_rounds):
                tasks = []
                for _ in range(self.size):
                    if delayed_tasks:
                        tasks.append(delayed_tasks.pop(0))
                    else:
                        tasks.append(None)

                logger.debug(
                    "Tasks for round %d/%d: %s",
                    round_idx + 1,
                    num_rounds,
                    tasks,
                )

                worker_args = [
                    (
                        task, 
                        self.record,
                        resolved_simulator,
                        worker_args_base,
                        start,
                        step
                    ) 
                    for task in tasks
                ]

                with mp.Pool(processes=self.size) as pool:
                    pool.starmap(worker_method, worker_args)

                self.__update_cache_for_round(tasks)
                logger.debug("Completed round %d/%d", round_idx + 1, num_rounds)

            self._finalize_problem(problem_name)

        if not resumed_any:
            logger.info("No incomplete jobs found. Nothing to resume.")

    def _finalize_problem(self, problem_name: str) -> None:
        """Mark a problem complete once all of its jobs are finished."""
        remaining = self.record.incomplete_tasks_for_problem(problem_name)
        if remaining:
            raise RuntimeError(
                f"Problem '{problem_name}' still has incomplete jobs: {remaining}"
            )

        self.record.cache.update_problem_status(problem_name, True)
        logger.info("Problem '%s' complete.", problem_name)
