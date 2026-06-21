"""Results index: maps condition/cell pairs to cache keys and lookup helpers."""

import logging
import uuid
from types import SimpleNamespace
from typing import List, Union

import pandas as pd

from benchtop._results_cacher import ResultCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Record:
    """Shared results dictionary accessed by workers across processes."""

    def __init__(
        self,
        problems: Union[List[SimpleNamespace], SimpleNamespace],
        cache_dir: str = "./.cache",
        load_index: bool = False,
    ) -> None:
        if isinstance(problems, list):
            self.problems = problems
        else:
            self.problems = [problems]

        self.problem = self.problems[0]
        self.current_problem_name = self.problem.name

        if not load_index:
            results_dict = self._results_dictionary()
        else:
            results_dict = None

        self.cache = ResultCache(
            results_dict=results_dict,
            cache_dir=cache_dir,
            load_index=load_index,
            problem_names=[p.name for p in self.problems],
        )

    def set_current_problem(
        self, problem: SimpleNamespace, problem_name: str | None = None
    ) -> None:
        """Point workers and lookups at the problem currently being simulated."""
        self.problem = problem
        self.current_problem_name = problem_name or problem.name

    def _results_dictionary(self) -> dict:
        """Build initial index: one entry per problem × condition × cell replicate."""
        results = {}

        for problem in self.problems:
            conditions_df = problem.condition_files[0]
            measurement_df = problem.measurement_files[0]

            for _, condition in conditions_df.iterrows():
                condition_id = condition["conditionId"]

                for cell in range(1, problem.cell_count + 1):
                    if "datasetId" in measurement_df.columns:
                        identifier = measurement_df["datasetId"][
                            measurement_df["simulationConditionId"] == condition_id
                        ].values[0]
                    else:
                        identifier = self._identifier_generator()

                    results[identifier] = {
                        "problem": problem.name,
                        "conditionId": condition_id,
                        "cell": cell,
                        "complete": False,
                    }

        return results

    def find_job_key(
        self, condition_id: str, cell: int, problem_name: str | None = None
    ) -> str | None:
        """Return cache key for a condition/cell pair within a problem."""
        problem_name = problem_name or self.current_problem_name

        for key in self.cache.job_keys():
            entry = self.cache.results_dict[key]
            entry_problem = entry.get("problem")
            if entry_problem is not None and entry_problem != problem_name:
                continue
            if (
                str(entry["conditionId"]) == str(condition_id)
                and str(entry["cell"]) == str(cell)
            ):
                return key

        return None

    def results_lookup(
        self,
        condition_id: str,
        cell: int,
        problem_name: str | None = None,
    ) -> pd.DataFrame | None:
        """Load cached trajectory for a condition/cell pair."""
        key = self.find_job_key(condition_id, cell, problem_name)
        if key is None:
            logger.error(
                "No prior results found for %s at cell %s (problem=%s)",
                condition_id,
                cell,
                problem_name or self.current_problem_name,
            )
            return None

        logger.debug("Results found for %s, cell %s", condition_id, cell)
        return self.cache.load(key)

    def condition_cell_id(self, rank_task: str, conditions_df) -> tuple:
        """Parse ``conditionId+cell`` task string into condition row and IDs."""
        condition_id, cell = rank_task.split("+")

        matches = conditions_df.loc[conditions_df["conditionId"] == condition_id]
        if matches.empty:
            raise ValueError(f"Condition ID '{condition_id}' not found in conditions_df")

        return matches.iloc[0], cell, condition_id

    def _identifier_generator(self) -> str:
        return str(uuid.uuid4())
