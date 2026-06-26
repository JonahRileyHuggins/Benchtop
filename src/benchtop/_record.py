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

        if load_index:
            merged = self._merge_loaded_index(self.cache.results_dict)
            self.cache.results_dict = merged
            self.cache._write_cache_index()

    def set_current_problem(
        self, problem: SimpleNamespace, problem_name: str | None = None
    ) -> None:
        """Point workers and lookups at the problem currently being simulated."""
        self.problem = problem
        self.current_problem_name = problem_name or problem.name

    def _results_dictionary(self) -> dict:
        """Build initial index: one entry per problem × condition × cell replicate."""
        jobs = self._expected_jobs()
        return {key: entry for key, entry in jobs.items()}

    def _expected_jobs(self) -> dict:
        """Return job entries keyed by stable cache identifiers."""
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

    def _job_lookup_key(self, entry: dict) -> tuple:
        return (
            entry.get("problem"),
            str(entry["conditionId"]),
            str(entry["cell"]),
        )

    def _merge_loaded_index(self, loaded: dict) -> dict:
        """Align a loaded cache index with the current benchmark configuration."""
        expected_jobs = self._expected_jobs()
        problem_names = [p.name for p in self.problems]

        meta = loaded.get(ResultCache.PROBLEMS_META_KEY, {})
        for name in problem_names:
            meta.setdefault(name, {"complete": False})

        loaded_by_job = {}
        for key in loaded:
            if key == ResultCache.PROBLEMS_META_KEY:
                continue
            entry = loaded[key]
            loaded_by_job[self._job_lookup_key(entry)] = (key, entry)

        merged = {ResultCache.PROBLEMS_META_KEY: meta}
        used_keys = set()

        for _, expected_entry in expected_jobs.items():
            job_key = self._job_lookup_key(expected_entry)
            if job_key in loaded_by_job:
                cache_key, loaded_entry = loaded_by_job[job_key]
                merged[cache_key] = loaded_entry
                used_keys.add(cache_key)
            else:
                identifier = self._identifier_generator()
                merged[identifier] = expected_entry

        for job_key, (cache_key, loaded_entry) in loaded_by_job.items():
            if cache_key not in used_keys:
                merged[cache_key] = loaded_entry

        return merged

    def incomplete_tasks_for_problem(self, problem_name: str) -> list[str]:
        """Return ``conditionId+cell`` task strings not yet marked complete."""
        incomplete = []
        for key in self.cache.job_keys():
            entry = self.cache.results_dict[key]
            if not self._job_belongs_to_problem(entry, problem_name):
                continue
            if not entry["complete"]:
                incomplete.append(f"{entry['conditionId']}+{entry['cell']}")
        return incomplete

    @staticmethod
    def _job_belongs_to_problem(entry: dict, problem_name: str) -> bool:
        entry_problem = entry.get("problem")
        if entry_problem is None:
            return True
        return entry_problem == problem_name

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
