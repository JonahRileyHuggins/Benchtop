"""Experiment orchestrator for parallel PEtab-style in-silico benchmarks.

Loads benchmark configuration, schedules conditions via topological sort,
dispatches simulations across a process pool, caches trajectories, and
computes observables.
"""

import logging
import multiprocessing as mp
import os
import pickle as pkl
from datetime import date
from typing import Union

import benchtop._observable_calculator as obs
from benchtop._abstract_simulator import AbstractSimulator
from benchtop._organizer import Organizer
from benchtop._record import Record
from benchtop._worker import worker_method
from benchtop.file_loader import FileLoader
from benchtop.registry import SIMULATOR_REGISTRY, load_simulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Experiment:
    """Run a single benchmark: simulate conditions, cache results, compute observables."""

    def __init__(
        self,
        petab_yaml: Union[os.PathLike, str],
        cores: int = os.cpu_count(),
        cache_dir: str = "./.cache",
        load_index: bool = False,
        verbose: bool = False,
    ) -> None:
        """Load benchmark YAML and PEtab companion files.

        Parameters
        ----------
        petab_yaml : path-like
            Benchmark configuration YAML.
        cores : int
            Worker processes for parallel simulation.
        cache_dir : str
            Directory for per-simulation pickle cache.
        load_index : bool
            Resume from an existing cache index.
        verbose : bool
            Enable debug logging.
        """
        self.org = Organizer(cores)
        self.size = cores

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        self.petab_yaml = os.path.abspath(petab_yaml)
        if not os.path.exists(self.petab_yaml):
            raise FileNotFoundError(f"{self.petab_yaml} is not a valid benchmark")

        self.loader = FileLoader(petab_yaml)
        self.loader._petab_files()

        self.details = self.loader.config
        self.name = self.details.problems[0].name or None
        self.cell_count = getattr(self.details.problems[0], "cell_count", 1)

        logger.info("Loading Experiment %s details from %s", self.name, self.petab_yaml)

        self.sbml_list = self._sbml_getter()

        self.record = Record(
            problem=self.loader.problems[0],
            cache_dir=cache_dir,
            load_index=load_index,
        )

    def run(
        self,
        simulator: str | AbstractSimulator,
        *args,
        start: float = 0.0,
        step: float = 30.0,
    ) -> None:
        """Simulate all conditions across the worker pool.

        Parameters
        ----------
        simulator : str or AbstractSimulator
            Registry name or simulator instance.
        args
            Passed to the simulator constructor (e.g. CLI namespace).
        start, step : float
            Simulation start time and output step size.
        """
        logger.debug("Starting in-silico experiment across %s cores.", self.size)

        if isinstance(simulator, str):
            try:
                simulator = load_simulator(simulator)
            except KeyError:
                raise ValueError(
                    f"Unknown simulator '{simulator}'. "
                    f"Available: {list(SIMULATOR_REGISTRY)}"
                )
        elif not isinstance(simulator, AbstractSimulator):
            raise TypeError(
                "simulator must be a registry name or AbstractSimulator subclass"
            )

        args = self._add_sbml_to_args(args)

        num_rounds, job_index = self.org.task_organization(
            self.loader.problems[0].measurement_files[0],
            self.cell_count,
        )

        for round_i in range(num_rounds):
            tasks = self.org.task_assignment(
                rank_jobs_directory=job_index,
                round_i=round_i,
            )
            logger.debug("Tasks for round: %s", tasks)

            worker_args = [
                (task, self.record, simulator, args, start, step)
                for task in tasks
            ]

            with mp.Pool(processes=self.size) as pool:
                pool.starmap(worker_method, worker_args)

            self._update_cache_for_round(tasks)

    def _update_cache_for_round(self, task_list: list) -> None:
        """Mark completed simulations in the cache index after each round."""
        remaining = []

        for task in task_list:
            if task is None:
                continue

            condition_id, cell = task.split("+")
            matched = False

            for key, record in self.record.cache.results_dict.items():
                if (
                    str(record["conditionId"]) == str(condition_id)
                    and str(record["cell"]) == str(cell)
                ):
                    self.record.cache.update_cache_index(key=key, status=True)
                    matched = True
                    break

            if not matched:
                remaining.append(task)

        assert remaining == [], f"Error in simulation task updates: {remaining}"

    def _add_sbml_to_args(self, args: tuple):
        """Attach SBML paths from config onto the simulator args namespace."""
        args_nsp = args[0]
        args_nsp.model_paths = self.sbml_list
        return args_nsp

    def _sbml_getter(self) -> list:
        """Collect SBML file paths declared in the benchmark YAML."""
        return [
            fp
            for problem in self.loader.problems
            if hasattr(problem, "sbml_files")
            for fp in problem.sbml_files
        ]

    def save_results(self, args) -> None:
        """Write final results pickle and remove the temporary cache."""
        results_directory = os.path.join(os.path.dirname(self.petab_yaml), "results")

        if args is not None and hasattr(args, "output"):
            results_directory = args.output

        os.makedirs(results_directory, exist_ok=True)

        results_path = os.path.join(results_directory, f"{date.today()}.pkl")
        if self.name is not None:
            results_path = os.path.join(results_directory, f"{self.name}.pkl")

        with open(results_path, "wb") as f:
            pkl.dump(self.record.cache.results_dict, f)

        self.record.cache.delete_cache()

    def observable_calculation(self, *args) -> None:
        """Evaluate observable formulas and persist comparison results."""
        self.record.cache.results_dict = obs.ObservableCalculator(self).run()
        self.save_results(args)

    def resume(
        self,
        simulator: AbstractSimulator,
        *args,
        start: float = 0.0,
        step: float = 30.0,
    ) -> None:
        """Continue an interrupted run from the cache index."""
        cache_index = self.record.cache.read_cache_index()

        incomplete = [
            f"{self.record.cache.results_dict[key]['conditionId']}"
            f"+{self.record.cache.results_dict[key]['cell']}"
            for key in cache_index
            if not cache_index[key]["complete"]
        ]

        topo_sorted = self.org.topologic_sort(
            measurements_df=self.loader.problems[0].measurement_files[0],
        )

        total_tasks = self.org.total_tasks(
            tasks=topo_sorted,
            cell_count=self.cell_count,
        )

        incomplete_set = set(incomplete)
        total_tasks = [task for task in total_tasks if task in incomplete_set]

        if not total_tasks:
            logger.info(
                "No incomplete jobs found for experiment '%s'. Nothing to resume.",
                self.name,
            )
            return

        delayed_tasks = self.org.delay_secondary_conditions(
            measurements_df=self.loader.problems[0].measurement_files[0],
            task_list=total_tasks,
            cell_count=self.cell_count,
        )

        logger.info("Resuming %s jobs for experiment '%s'...", len(total_tasks), self.name)

        num_rounds = -(-len(delayed_tasks) // self.size)  # ceiling division

        for round_idx in range(num_rounds):
            tasks = []
            for _ in range(self.size):
                tasks.append(delayed_tasks.pop(0) if delayed_tasks else None)

            logger.debug(
                "Tasks for round %s/%s: %s", round_idx + 1, num_rounds, tasks
            )

            worker_args = [
                (task, self.record, simulator, *args, start, step)
                for task in tasks
            ]

            with mp.Pool(processes=self.size) as pool:
                pool.starmap(worker_method, worker_args)

            logger.debug("Completed round %s/%s", round_idx + 1, num_rounds)

        self._store_final_results()
