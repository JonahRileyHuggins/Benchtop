#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class object for handling results. 

Author: Jonah R. Huggins

"""
import pandas as pd

import src.benchtop.utils as utils
from src.benchtop.ResultsCacher import ResultCache


class Record:
    """Records results dictionary access across processes."""
    def __init__(
            self, 
            problem: dict,
            cache_dir: str = './.cache', 
            load_index: bool = False
            ) -> None:

        self.problem = problem

        self.results_dict = self.__results_dictionary()

        self.cache = ResultCache(
            results_dict = self.results_dict,
            cache_dir=cache_dir, 
            load_index=load_index
            )
    
    def __results_dictionary(self) -> dict:
        """Create an empty dictionary for storing results
        input:
            filtered_conditions: pd.DataFrame - filtered conditions dataframe

        output:
            returns the empty results dictionary, ready to be filled
        """

        #for now, only supporting one problem per file
        conditions_df = self.problem.condition_files[0]
        measurement_df = self.problem.measurement_files[0]

        results = {}

        for idx, condition in conditions_df.iterrows():

            condition_id = condition["conditionId"]

            for cell in range(1, self.problem.cell_count+1):
                if "datasetId" in measurement_df.columns:
                    identifier = measurement_df["datasetId"]\
                        [measurement_df["simulationConditionId"] == condition_id].values[0]
                else:
                    identifier = utils.identifier_generator()

                results[identifier] = {
                    "conditionId": condition_id,
                    "cell": cell
                }

        return results
    
    def results_lookup(
            self, 
            condition_id: str, 
            cell: int
            ) -> pd.DataFrame:
        """Indexes results dictionary on condition id, returns results"""
        # results keys should all be species names paired with single numpy arrays. 
        for key in self.results_dict.keys():
            if self.results_dict[key]['conditionId'] == condition_id\
                and self.results_dict[key]['cell'] == cell:

                return self.cache.load(key)
            
    def condition_cell_id(
        self,
        rank_task: str, 
        conditions_df
        ) -> str:
        """
        Extract the condition for the task from the filtered_conditions
        output:
            returns the condition for the task
        """

        cell = rank_task.split("+")[1]

        condition_id = rank_task.split("+")[0]

        matches = conditions_df.loc[conditions_df["conditionId"] == condition_id]

        if matches.empty:
            raise ValueError(f"Condition ID '{condition_id}' not found in conditions_df")

        condition = matches.iloc[0]

        return condition, cell, condition_id

