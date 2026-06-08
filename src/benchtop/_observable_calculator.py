"""Evaluate PEtab observable formulas and align simulation with measurements."""

import math
import re
from typing import List

import numpy as np
import pandas as pd


class ObservableCalculator:
    """Downsample trajectories, evaluate formulas, pair with experimental data."""

    def __init__(self, parent):
        self.results_dict = parent.record.cache.results_dict
        self.cache = parent.record.cache
        self.observable_df = parent.loader.problems[0].observable_files[0]
        self.measurement_df = parent.loader.problems[0].measurement_files[0]
        self.data_groups = self._group_conditions_and_observables()
        self.observable_results = self._build_observable_results_dict()

    def _group_conditions_and_observables(self) -> pd.core.groupby.generic.DataFrameGroupBy:
        if self.measurement_df.empty or self.observable_df.empty:
            raise ValueError("PEtab DataFrame is empty; cannot group.")

        return self.measurement_df.groupby(["simulationConditionId", "observableId"])

    def _build_observable_results_dict(self) -> dict:
        return {
            entry: {
                "conditionId": self.results_dict[entry]["conditionId"],
                "cell": self.results_dict[entry]["cell"],
            }
            for entry in self.results_dict
        }

    def run(self) -> dict:
        """Compute observables for every cached simulation entry."""
        for entry in self.results_dict:
            condition_id = self.results_dict[entry]["conditionId"]
            matched_formulas = self._get_entry_formulas(condition_id)
            dataset = self.cache.load(entry)

            for observable_key, formula in matched_formulas.items():
                self.observable_results[entry][observable_key] = {}
                group = self.data_groups.get_group((condition_id, observable_key))

                self.observable_results[entry][observable_key]["experiment"] = (
                    self._get_experimental_data(group)
                )
                self.observable_results[entry][observable_key]["simulation"] = (
                    self._calculate_formula(dataset, formula, group)
                )
                self.observable_results[entry][observable_key]["time"] = (
                    self._downsample_timepoints(dataset, group)
                )

        return self.observable_results

    def _get_entry_formulas(self, condition_id: str) -> dict:
        matched_obs_ids = self._get_condition_observables(condition_id)
        return {
            obs_id: self.observable_df["observableFormula"][
                self.observable_df["observableId"] == obs_id
            ].iloc[0]
            for obs_id in matched_obs_ids
        }

    def _get_condition_observables(self, condition_id: str) -> list:
        return [
            obs
            for (cond, obs) in self.data_groups.groups
            if cond == condition_id
        ]

    def _get_experimental_data(self, group) -> np.ndarray:
        return np.array(group["measurement"])

    def _calculate_formula(self, dataset: pd.DataFrame, formula: str, group) -> np.ndarray:
        """Substitute species arrays into formula and eval; downsample to exp times."""
        null_like = {"", None, 0, "0", float("nan"), np.nan, "nan"}
        if formula in null_like or (isinstance(formula, float) and math.isnan(formula)):
            return None

        species_names = self._get_valid_species(formula)
        namespace = self._formula_namespace(species_names, dataset)

        try:
            formula_answer = eval(formula, {"np": np}, namespace)
        except Exception:
            raise RuntimeError(f"Failed to evaluate observable formula: {formula}")

        return self._downsample_results(formula_answer, dataset, group)

    @staticmethod
    def _get_valid_species(formula: str) -> List[str]:
        """Extract PEtab-compliant species identifiers from a formula string."""
        if not isinstance(formula, str):
            raise TypeError("Observable formula must be a string.")

        valid_species = re.findall(
            r"(?:@[A-Za-z_]+::[A-Za-z_]\w*|[A-Za-z_]\w*)(?:\(\))?",
            formula,
        )
        if not valid_species:
            raise ValueError("No valid species found in the observable formula.")

        return valid_species

    def _formula_namespace(self, species_names: list, dataset: pd.DataFrame) -> dict:
        """Map species names to numpy arrays for safe eval."""
        return {
            name: np.asarray(self._safe_retrieve_array(dataset, name))
            for name in species_names
        }

    @staticmethod
    def _safe_retrieve_array(dataset, species_name) -> np.ndarray:
        species_arr = dataset[species_name]
        if species_arr is None:
            raise KeyError(f"Species '{species_name}' not found in dataset.")

        if isinstance(species_arr, (pd.Series, pd.DataFrame)):
            species_arr = species_arr.to_numpy().ravel()
        if not isinstance(species_arr, np.ndarray):
            raise ValueError(
                f"Replacement value for species '{species_name}' is not a valid array."
            )

        return species_arr

    def _downsample_results(
        self, observable_answer: np.ndarray, dataset: pd.DataFrame, group
    ) -> np.ndarray:
        """Slice simulation values to experimental measurement timepoints."""
        valid_rows = group.dropna(subset=["measurement"])
        if valid_rows.empty:
            return observable_answer

        exp_time = np.sort(
            valid_rows["time"].dropna().astype(float).unique()
        )
        sim_indices = self._get_exp_time_indices(exp_time, dataset["time"])
        return observable_answer[np.sort(sim_indices)]

    @staticmethod
    def _get_exp_time_indices(exp_time: np.ndarray, sim_time: np.ndarray) -> list:
        """Index of closest simulation time for each experimental timepoint."""
        return [
            int(np.argmin(np.abs(sim_time - t)))
            for t in np.sort(exp_time)
        ]

    def _downsample_timepoints(self, dataset: pd.DataFrame, group) -> np.ndarray:
        valid_rows = group.dropna(subset=["measurement"])
        if valid_rows.empty:
            return dataset["time"]

        exp_time = np.sort(
            valid_rows["time"].dropna().astype(float).unique()
        )
        indices = self._get_exp_time_indices(exp_time, dataset["time"])
        return np.unique(np.array(dataset["time"][indices]))
