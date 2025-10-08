
import os
import json
import pickle
import shutil
from typing import Any
import pandas as pd


class ResultCache:

    def __init__(
            self, 
            results_dict: dict[str, Any], 
            cache_dir: str = './.cache', 
            load_index: bool = False
            ) -> None:
        self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_index_path = os.path.join(self.cache_dir, "cache_index.json")
        
        if not load_index:
            
            # Convert job identifiers (keys) into dict entries with default value 0
            for key in results_dict.keys():
                results_dict[key]['complete'] = False

            # Write to cache file (overwrite if it already exists)
            with open(self.cache_index_path, 'w') as file:
                json.dump(results_dict, file, indent=2)

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

    def read_cache_index(self) -> dict[str, Any]:
        """Read cache_index.json as dictionary"""
        with open(self.cache_index_path, 'r') as file:
            cache_index = json.load(file)

        return cache_index