"""Per-process simulation worker: load model, apply conditions, cache trajectories."""

import gc
import logging
import multiprocessing as mp

import pandas as pd

from benchtop._abstract_simulator import AbstractSimulator
from benchtop._record import Record

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def worker_method(
    task: str,
    record: Record,
    simulator: AbstractSimulator,
    args: tuple = (),
    start: float = 0.0,
    step: float = 30.0,
) -> None:
    """Entry point for multiprocessing.Pool; constructs Worker in the child process."""
    Worker(task, record, simulator, args, start, step)


class Worker:
    """Run one simulation task: preequilibrate, set parameters, simulate, cache."""

    def __init__(
        self,
        task: str,
        record: Record,
        simulator: AbstractSimulator,
        args: tuple = (),
        start: float = 0.0,
        step: float = 30.0,
    ):
        self.record = record
        self.simulator = simulator(args)
        self._run_task(task, start, step)
        self.simulator = None
        gc.collect()

    def _run_task(self, task: str, start: float = 0.0, step: float = 30.0) -> None:
        rank = mp.current_process().name

        if task is None:
            logger.debug("Rank %s has no tasks to complete", rank)
            return

        condition, cell, condition_id = self.record.condition_cell_id(
            rank_task=task,
            conditions_df=self.record.problem.condition_files[0],
        )

        logger.info("%s running %s for replicate %s", rank, condition_id, cell)
        logger.debug(
            "Conditions for %s: %s",
            condition_id,
            [f"{i}: {j}" for i, j in zip(condition.index, condition.values)],
        )

        # Apply preequilibration final state before condition overrides
        precondition_results = self._extract_preequilibration_results(condition_id, cell)
        if precondition_results:
            self._set_model_state(
                list(precondition_results.keys()),
                list(precondition_results.values()),
            )

        self._set_model_state(condition.keys(), condition.values.tolist())

        stop_time = self._get_simulation_time(condition)
        results_array = self.simulator.simulate(start, stop_time, step)
        results = pd.DataFrame(results_array)

        parcel = self._package_results(results, condition_id, cell)
        logger.info("%s finished %s for cell %s", rank, condition_id, cell)

        self._cache_results(parcel)

    def _extract_preequilibration_results(
        self, condition_id: str, cell: int
    ) -> dict:
        """Return final-state species dict from a preequilibration run, if any."""
        measurement_df = self.record.problem.measurement_files[0]
        precondition_dict = {}

        if "preequilibrationConditionId" not in measurement_df.columns:
            return precondition_dict

        precondition_matches = measurement_df[
            measurement_df["simulationConditionId"] == condition_id
        ]
        if precondition_matches.empty:
            return precondition_dict

        precondition_id = precondition_matches["preequilibrationConditionId"].iloc[0]
        if pd.isna(precondition_id) or str(precondition_id).strip().lower() == "nan":
            return precondition_dict

        logger.debug(
            "Searching preequilibration for condition_id=%s, cell=%s",
            condition_id,
            cell,
        )

        precondition_df = self.record.results_lookup(precondition_id, cell)
        if precondition_df is None:
            return precondition_dict

        logger.info(
            "Extracting preequilibration %s for condition %s",
            precondition_id,
            condition_id,
        )

        if "time" in precondition_df.columns:
            precondition_df = precondition_df.drop("time", axis=1)

        return dict(
            zip(
                list(precondition_df.columns),
                precondition_df.iloc[-1].to_list(),
            )
        )

    def _set_model_state(self, names: list, states: list) -> None:
        """Apply parameter/species overrides to the simulator."""
        if len(names) != len(states):
            raise ValueError(
                f"Length mismatch: {len(names)} names vs {len(states)} states"
            )

        skip = {"conditionId", "conditionName"}

        for name, state in zip(names, states):
            if name in skip:
                continue
            if not isinstance(name, str):
                raise TypeError(f"Invalid component name type: {name} ({type(name)})")
            logger.debug("Modifying %s = %s", name, state)
            self.simulator.modify(name, state)

    def _get_simulation_time(self, condition: pd.Series) -> float:
        """Max measurement timepoint for this condition."""
        measurement_df = self.record.problem.measurement_files[0]
        matching_times = measurement_df.loc[
            measurement_df["simulationConditionId"].isin(condition), "time"
        ]

        if matching_times.empty:
            raise ValueError(
                f"No simulation time defined for condition {condition['conditionId']}"
            )

        return matching_times.max()

    def _cache_results(self, parcel: dict) -> None:
        """Persist simulation DataFrame to the pickle cache."""
        condition_id = parcel["conditionId"]
        cell = parcel["cell"]
        results = parcel["results"]

        for key in self.record.cache.results_dict:
            entry = self.record.cache.results_dict[key]
            if (
                str(entry["conditionId"]) == str(condition_id)
                and str(entry["cell"]) == str(cell)
            ):
                self.record.cache.save(key=key, df=results)

    def _package_results(
        self, results: pd.DataFrame, condition_id: str, cell: str
    ) -> dict:
        return {
            "conditionId": condition_id,
            "cell": int(cell),
            "results": results,
        }
