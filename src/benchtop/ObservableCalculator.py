#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculates and downsamples observable formulas from simulation results.

Author: Jonah R. Huggins

Description: 
"""
# -----------------------Package Import & Defined Arguements-------------------#
import re
import math
from typing import List, Hashable

import numpy as np
import pandas as pd

#-------------------------Initialization & Variables---------------------------#

class ObservableCalculator:
    """Class object for calculating provided observable in PEtab Observables file.
    Uses composition and encapsulation properties of Experiment object to extend
    functionality."""

    def __init__(self, parent):

        self.results_dict = parent.record.results_dict

        self.cache = parent.record.cache

        self.observable_df = parent.loader.problems[0].observable_files[0]
        
        self.measurement_df = parent.loader.problems[0].measurement_files[0]

        self.data_groups = self._group_conditions_and_observables()

        self.observable_results = self._build_observable_results_dict()

    def _group_conditions_and_observables(self) -> pd.core.groupby.generic.DataFrameGroupBy:
        """
        Group conditions and observables in the measurement dataframe.

        Returns:
        - pd.DataFrame: The grouped measurement dataframe.
        """
        try:
            if self.measurement_df.empty or self.observable_df.empty:
                raise ValueError("PEtab DataFrame is empty. Cannot group empty dataframes")

            # Group conditions and observables in the measurement dataframe
            grouped_df = self.measurement_df.groupby(["simulationConditionId", "observableId"])

            return grouped_df

        except Exception as e:
            print(f"Error in group_conditions_and_observables: {e}")
            return pd.DataFrame()

    def _build_observable_results_dict(self):
        """Constructs the results dictionary for data reduced, calculated observables 
        to be stored in.
        """
        # Stores the calculated observable values for each observable and condition
        observable_dict = {}

        for entry in self.results_dict:

            observable_dict[entry] = {
                "conditionId": self.results_dict[entry]["conditionId"],
                "cell": self.results_dict[entry]["cell"]
            }

        return observable_dict

    def run(self):
        """Runtime function for executing the observable calculator and reducing results to bare minimum"""

        for entry in self.results_dict:

            conditionId = self.results_dict[entry]['conditionId']

            matched_formulas = self._get_entry_formulas(conditionId)

            # --- reduce I/O operations by loading per entry ---
            dataset = self.cache.load(entry)

            # -- iterative process for downsampling to observable-only data ---
            for observable_key, formula in matched_formulas.items():

                self.observable_results[entry][observable_key] = {}

                group = self.data_groups.get_group((conditionId, observable_key))

                # --- PEtab measurements file defines experimental data ---
                self.observable_results[entry][observable_key]['experiment'] = self._get_experimental_data(group)

                self.observable_results[entry][observable_key]['simulation'] = self._calculate_formula(
                    dataset, 
                    formula, 
                    group
                    )

                # --- Timepoints are reduced to bare minimum if applicable ---
                self.observable_results[entry][observable_key]['time'] = self._downsample_timepoints(
                    dataset, 
                    group
                    )

        return self.observable_results

    def _get_entry_formulas(self, conditionId: str):
        """Takes an entry from a results dictionary and finds the corresponding observable formulas
        for an entry"""

        matched_obsIds = self._get_condition_observables(conditionId)

        formulas = {}

        for obsId in matched_obsIds:

            # Extraction Logic Statement
            formula_i = self.observable_df['observableFormula']\
                [self.observable_df['observableId'] == obsId].iloc[0] # This should be the first and only value in the series

            formulas[obsId] = formula_i

        return formulas

    def _get_condition_observables(self, conditionId):
        """Get observableIds associated with conditionId"""

        matched_observableIds = []

        for (cond, obs), index in self.data_groups.groups.items():

            if cond == conditionId:
                
                matched_observableIds.append(obs)

        return matched_observableIds

    def _get_experimental_data(self, group):
        """Gets experimental data from PEtab measurement file"""
        
        return np.array(group['measurement'])

    def _calculate_formula(self, dataset: pd.DataFrame, formula: str, 
                           group: pd.core.groupby.generic.DataFrameGroupBy):
        """Takes a formula string and returns the results of the intended mathematical
        expression."""

        # List of values considered to mean "empty" or "skip"
        acceptable_nulls = ['', None, 0, '0', float('nan'), np.nan]

        # Check if formula is in the null-like set
        if formula in acceptable_nulls or (
            isinstance(formula, float) and math.isnan(formula)
            ):
            return None
        
        species = self._get_valid_species(formula)
        
        for variable in species:
            # At each iteration, the formula updates with each species array
            formula = self.swap_species_for_array(
                dataset, variable, formula
            )

        formula_answer = eval(formula)

        formula_answer = self._downsample_results(formula_answer, dataset, group)

        return formula_answer

    @staticmethod
    def _get_valid_species(formula: str) -> List[str]:
        """
        Extract valid species identifiers from an observable formula based on PEtab naming conventions.

        Parameters:
        - observable_formula (str): The formula containing species and mathematical expressions.

        Returns:
        - List[str]: A list of valid species identifiers.

        Raises:
        - ValueError: If no valid species are found in the observable formula.
        - TypeError: If the input is not a string.
        """
        if not isinstance(formula, str):
            raise TypeError("Input observable_formula must be a string.")
        
        # Regex for PEtab-compliant species identifiers
        valid_species = re.findall(r"[A-Za-z_]\w*", formula)

        if not valid_species:
            raise ValueError("No valid species found in the observable formula.")

        return valid_species

    def swap_species_for_array(
            self, 
            dataset: pd.DataFrame, 
            species_i: str, 
            observable_formula: str
            ) -> str:
        """
        Takes a species identifier and returns the corresponding array from the results_dict.

        Parameters:
        - dataset (pd.DataFrame): current experiment simulation loaded
        - species_i (str): The species identifier.
        - observable_formula (str): The formula containing species and mathematical expressions.

        Returns:
        - observable_formula (str): The observable formula with the species replaced by the array.

        Raises:
        - KeyError: If the species identifier is not found in the results dictionary.
        - ValueError: If the replacement value cannot be converted to a NumPy array.
        """
        try:
            # Validate inputs
            if not isinstance(species_i, str):
                raise TypeError("The species_i must be a string.")
            if not isinstance(observable_formula, str):
                raise TypeError("The observable_formula must be a string.")

            # Retrieve species-specific results from current entry
            replacement_value = dataset[species_i]

            if replacement_value is None:
                raise KeyError(f"Species '{species_i}' not found in dataset.")

            # Convert to NumPy array
            if isinstance(replacement_value, pd.Series):
                replacement_value = replacement_value.to_numpy()
            elif not isinstance(replacement_value, np.ndarray):
                raise ValueError(f"Replacement value for species '{species_i}' is not a valid array or Series.")

            # Prepare replacement string
            replacement_value_str = f"np.array({replacement_value.tolist()})"

            # Replace only exact matches of the species name in the formula
            observable_formula = re.sub(fr"\b{re.escape(species_i)}\b",
                                        replacement_value_str,
                                        observable_formula)

            return observable_formula

        except Exception as e:
            print(f"Error in swap_species_for_array: {e}")
            raise

    def _downsample_results(self, observable_answer: np.array, 
                            dataset: pd.DataFrame, 
                            group: pd.core.groupby.generic.DataFrameGroupBy
                            ) -> np.array:
        """Reduce the data to only the timepoints in the experimental data.

        Parameters:
        - observable_answer (np.array): The observable values from the simulation.
        - dataset (pd.DataFrame): current experiment simulation loaded
        - group (pd.DataFrameGroupBy): Grouped measurement dataframe.

        Returns:
        - observable_answer (np.array): The reduced observable values.
        """
        # Ensure first that there is no experimental values in the group's measurement
        # before reducing the timepoints, if none are found, return the original
        if group["measurement"].isna().all():
            return observable_answer

        exp_time = self.measurement_df["time"].unique()

        time = dataset['time']

        sim_equivalent_indicies = self._get_exp_time_indicies(exp_time, time)

        # Reduce the observable_answer to only the timepoints in the experimental data
        observable_answer = observable_answer[sim_equivalent_indicies]

        return observable_answer

    @staticmethod
    def _get_exp_time_indicies(exp_time:np.array, sim_time:np.array):
        """Returns indicies of simulation time trajectories closest to experimental 
        equivalent timepoint recordings"""

        sim_timepoint_indicies = []

        for t in exp_time:
            closest_idx = np.argmin(np.abs(sim_time - t))
            sim_timepoint_indicies.append(closest_idx)

        return sim_timepoint_indicies

    def _downsample_timepoints(
            self, 
            dataset: pd.DataFrame,
            group: pd.core.groupby.generic.DataFrameGroupBy
            ) -> np.array:
        """Reduce the number of timepoints in the simulation results. to match
            the number of timepoints in the experimental data.

        Parameters:
        - dataset (pd.DataFrame): current experiment simulation loaded

        Returns:
        - time (np.array): The reduced timepoints.
        """

        time = dataset['time']

        if group["measurement"].isna().all():
            return time

        exp_time = self.measurement_df["time"].unique()

        sim_equivalent_indicies = self._get_exp_time_indicies(exp_time, time)

        return np.unique(np.array(time[sim_equivalent_indicies]))
