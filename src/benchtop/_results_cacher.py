"""Pickle cache for simulation trajectories with a JSON completion index."""

import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

DEFAULT_CACHE = Path(Path.home() / ".cache" / "benchtop")

class ResultCache:
    """Store per-simulation DataFrames on disk; track completion in cache_index.json."""

    PROBLEMS_META_KEY = "__problems__"

    def __init__(
        self,
        results_dict: Optional[Dict[str, Any]] = None,
        cache_dir: Union[os.PathLike, str] = DEFAULT_CACHE,
        load_index: bool = False,
        problem_names: Optional[List[str]] = None,
    ) -> None:
        self.cache_dir = os.path.abspath(cache_dir)
        self.cache_index_path = os.path.join(self.cache_dir, "cache_index.json")

        if not load_index:
            if results_dict is None:
                raise ValueError("results_dict must be provided when load_index=False")

            self.results_dict = results_dict
            self._ensure_problem_status(problem_names or self._infer_problem_names())

            try:
                os.makedirs(self.cache_dir, exist_ok=False)
            except OSError:
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=False)

            self._write_cache_index()
        else:
            if not os.path.exists(self.cache_index_path):
                raise FileNotFoundError(
                    f"No cache index found at {self.cache_index_path}. "
                    "Run once with load_index=False to create it."
                )

            with open(self.cache_index_path) as f:
                self.results_dict = json.load(f)

            inferred = self._infer_problem_names()
            self._ensure_problem_status(problem_names or inferred)

    def _infer_problem_names(self) -> List[str]:
        names = []
        for key in self.job_keys():
            problem = self.results_dict[key].get("problem")
            if problem and problem not in names:
                names.append(problem)
        return names

    def _ensure_problem_status(self, problem_names: List[str]) -> None:
        if self.PROBLEMS_META_KEY not in self.results_dict:
            self.results_dict[self.PROBLEMS_META_KEY] = {
                name: {"complete": False} for name in problem_names
            }
            return

        status = self.results_dict[self.PROBLEMS_META_KEY]
        for name in problem_names:
            status.setdefault(name, {"complete": False})

    def job_keys(self) -> List[str]:
        return [key for key in self.results_dict if key != self.PROBLEMS_META_KEY]

    def _write_cache_index(self) -> None:
        with open(self.cache_index_path, "w") as f:
            json.dump(self.results_dict, f, indent=2)

    def _key_to_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.pkl")

    def update_cache_index(self, key: str, status: bool) -> None:
        with open(self.cache_index_path) as f:
            cache_data = json.load(f)

        cache_data[key]["complete"] = status

        with open(self.cache_index_path, "w") as f:
            json.dump(cache_data, f, indent=2)

        self.results_dict[key]["complete"] = status

    def update_problem_status(self, problem_name: str, complete: bool) -> None:
        with open(self.cache_index_path) as f:
            cache_data = json.load(f)

        self._ensure_problem_status([problem_name])
        cache_data[self.PROBLEMS_META_KEY][problem_name]["complete"] = complete

        with open(self.cache_index_path, "w") as f:
            json.dump(cache_data, f, indent=2)

        self.results_dict[self.PROBLEMS_META_KEY][problem_name]["complete"] = complete

    def is_problem_complete(self, problem_name: str) -> bool:
        status = self.results_dict.get(self.PROBLEMS_META_KEY, {})
        return status.get(problem_name, {}).get("complete", False)

    def save(self, key: str, df: pd.DataFrame) -> None:
        with open(self._key_to_path(key), "wb") as f:
            pickle.dump(df, f)

    def load(self, key: str) -> pd.DataFrame:
        with open(self._key_to_path(key), "rb") as f:
            return pickle.load(f)

    def load_columns(self, key: str, columns: list) -> pd.DataFrame:
        """Load a trajectory and return only the requested columns."""
        trajectory = self.load(key)
        missing = [column for column in columns if column not in trajectory.columns]
        if missing:
            raise KeyError(
                f"Columns {missing} not found in cached trajectory '{key}'."
            )

        subset = trajectory.loc[:, columns].copy()
        return subset

    def delete_cache(self) -> None:
        shutil.rmtree(self.cache_dir, ignore_errors=False)

    def read_cache_index(self) -> Dict[str, Any]:
        with open(self.cache_index_path) as f:
            return json.load(f)
