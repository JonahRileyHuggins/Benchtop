
import os
import json
import pickle
import shutil
from typing import Any, Dict, Optional

import pandas as pd


class ResultCache:

    def __init__(
            self, 
            results_dict: Optional[Dict[str, Any]] = None, 
            cache_dir: str = './.cache', 
            load_index: bool = False
        ) -> None:
        self.cache_dir = os.path.abspath(cache_dir)
        try: 
            os.makedirs(self.cache_dir, exist_ok=False)
        except OSError as e:
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=False)

        self.cache_index_path = os.path.join(self.cache_dir, "cache_index.json")
        
        if not load_index:
            if results_dict is None:
                raise ValueError("results_dict must be provided when load_index=False")

            # Initialize new cache index with 'complete' flags
            for key in results_dict.keys():
                results_dict[key]['complete'] = False

            self.results_dict = results_dict

            # Write new cache index
            with open(self.cache_index_path, 'w') as f:
                json.dump(self.results_dict, f, indent=2)

        else:
            # Load existing cache index
            if not os.path.exists(self.cache_index_path):
                raise FileNotFoundError(
                    f"No cache index found at {self.cache_index_path}. "
                    "Run once with load_index=False to create it."
                )

            with open(self.cache_index_path, 'r') as f:
                self.results_dict = json.load(f)


    def _key_to_path(self, key: str) -> str:
        """Convert a dictionary key to a safe file path"""

        return os.path.join(self.cache_dir, f"{key}.pkl")

    def update_cache_index(self, key: str, status: bool) -> None:
        # Read the current cache
        with open(self.cache_index_path, 'r') as f:
            cache_data = json.load(f)

        # Update the entry
        cache_data[key]['complete'] = status

        # Write it back
        with open(self.cache_index_path, 'w') as f:
            json.dump(cache_data, f, indent=2)

    def save(self, key: str, df: pd.DataFrame) -> None:
        """Save a single DataFrame under a key"""
        path = self._key_to_path(key)

        with open(path, 'wb') as f:
            pickle.dump(df, f)

    def load(self, key: str) -> pd.DataFrame:
        """Load a single DataFrame by key"""

        path = self._key_to_path(key)

        with open(path, 'rb') as f:
            return pickle.load(f)

    def delete_cache(self) -> None:
        """Removes cache directory after results have been saved."""
        shutil.rmtree(self.cache_dir, ignore_errors=False)

    def read_cache_index(self) -> Dict[str, Any]:
        """Read cache_index.json as dictionary"""
        with open(self.cache_index_path, 'r') as file:
            cache_index = json.load(file)

        return cache_index
