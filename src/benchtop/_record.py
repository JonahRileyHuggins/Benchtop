"""Results index: maps condition/cell pairs to cache keys and lookup helpers."""

import logging
import uuid

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
        problem: dict,
        cache_dir: str = "./.cache",
        load_index: bool = False,
    ) -> None:
        self.problem = problem
        results_dict = self._results_dictionary()

        self.cache = ResultCache(
            results_dict=results_dict,
            cache_dir=cache_dir,
            load_index=load_index,
        )

    def _results_dictionary(self) -> dict:
        """Build initial index: one entry per condition × cell replicate."""
        conditions_df = self.problem.condition_files[0]
        measurement_df = self.problem.measurement_files[0]
        results = {}

        for _, condition in conditions_df.iterrows():
            condition_id = condition["conditionId"]

            for cell in range(1, self.problem.cell_count + 1):
                if "datasetId" in measurement_df.columns:
                    identifier = measurement_df["datasetId"][
                        measurement_df["simulationConditionId"] == condition_id
                    ].values[0]
                else:
                    identifier = self._identifier_generator()

                results[identifier] = {
                    "conditionId": condition_id,
                    "cell": cell,
                    "complete": False,
                }

        return results

    def results_lookup(self, condition_id: str, cell: int) -> pd.DataFrame | None:
        """Load cached trajectory for a condition/cell pair."""
        for key in self.cache.results_dict:
            entry = self.cache.results_dict[key]
            if (
                str(entry["conditionId"]) == str(condition_id)
                and str(entry["cell"]) == str(cell)
            ):
                logger.debug("Results found for %s, cell %s", condition_id, cell)
                return self.cache.load(key)

        logger.error("No prior results found for %s at cell %s", condition_id, cell)
        return None

    def condition_cell_id(self, rank_task: str, conditions_df) -> tuple:
        """Parse ``conditionId+cell`` task string into condition row and IDs."""
        condition_id, cell = rank_task.split("+")

        matches = conditions_df.loc[conditions_df["conditionId"] == condition_id]
        if matches.empty:
            raise ValueError(f"Condition ID '{condition_id}' not found in conditions_df")

        return matches.iloc[0], cell, condition_id

    def _identifier_generator(self) -> str:
        return str(uuid.uuid4())
