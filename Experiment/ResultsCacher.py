
import os
import pickle
import shutil

import pandas as pd


class ResultCache:

    def __init__(self, cach_dir: str = './.cache') -> None:
        self.cache_dir = cach_dir

        os.makedirs(self.cache_dir, exist_ok=True)

    def _key_to_path(self, key: str) -> str:
        """Convert a dictionary key to a safe file path"""

        return os.path.join(self.cache_dir, f"{key}.pkl")

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
