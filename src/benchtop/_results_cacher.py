"""Pickle cache for simulation trajectories with a JSON completion index."""

import json
import os
import pickle
import shutil
from typing import Any, Dict, Optional

import pandas as pd


class ResultCache:
    """Store per-simulation DataFrames on disk; track completion in cache_index.json."""

    def __init__(
        self,
        results_dict: Optional[Dict[str, Any]] = None,
        cache_dir: str = "./.cache",
        load_index: bool = False,
    ) -> None:
        self.cache_dir = os.path.abspath(cache_dir)
        self.cache_index_path = os.path.join(self.cache_dir, "cache_index.json")

        if not load_index:
            if results_dict is None:
                raise ValueError("results_dict must be provided when load_index=False")

            self.results_dict = results_dict

            try:
                os.makedirs(self.cache_dir, exist_ok=False)
            except OSError:
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=False)

            with open(self.cache_index_path, "w") as f:
                json.dump(self.results_dict, f, indent=2)
        else:
            if not os.path.exists(self.cache_index_path):
                raise FileNotFoundError(
                    f"No cache index found at {self.cache_index_path}. "
                    "Run once with load_index=False to create it."
                )

            with open(self.cache_index_path) as f:
                self.results_dict = json.load(f)

    def _key_to_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.pkl")

    def update_cache_index(self, key: str, status: bool) -> None:
        with open(self.cache_index_path) as f:
            cache_data = json.load(f)

        cache_data[key]["complete"] = status

        with open(self.cache_index_path, "w") as f:
            json.dump(cache_data, f, indent=2)

    def save(self, key: str, df: pd.DataFrame) -> None:
        with open(self._key_to_path(key), "wb") as f:
            pickle.dump(df, f)

    def load(self, key: str) -> pd.DataFrame:
        with open(self._key_to_path(key), "rb") as f:
            return pickle.load(f)

    def delete_cache(self) -> None:
        shutil.rmtree(self.cache_dir, ignore_errors=False)

    def read_cache_index(self) -> Dict[str, Any]:
        with open(self.cache_index_path) as f:
            return json.load(f)
