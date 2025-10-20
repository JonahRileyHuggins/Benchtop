#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class object for handling results. 

Author: Jonah R. Huggins

"""

import uuid
import logging

import pandas as pd

from ResultsCacher import ResultCache

logging.basicConfig(
    level=logging.INFO, # Overriden if Verbose Arg. True
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Record:
    """Records results dictionary access across processes."""
    def __init__(
            self, 
            problem: dict,
            cache_dir: str = './.cache', 
            load_index: bool = False
            ) -> None:

        self.problem = problem

        # --- initial dictionary, replaced if cached index present ---
        results_dict = self.__results_dictionary()

        self.cache = ResultCache(
            results_dict = results_dict,
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
                    identifier = self.__identifier_generator()

                results[identifier] = {
                    "conditionId": condition_id,
                    "cell": cell,
                    "complete": False
                }

        return results
    
    def results_lookup(
            self, 
            condition_id: str, 
            cell: int
            ) -> pd.DataFrame:
        """Indexes results dictionary on condition id, returns results"""
        # results keys should all be species names paired with single numpy arrays. 
        for key in self.cache.results_dict.keys():
            if str(self.cache.results_dict[key]['conditionId']) == str(condition_id)\
                and str(self.cache.results_dict[key]['cell']) == str(cell):
                logger.debug(f"results found for {condition_id} and cell {cell}")
                return self.cache.load(key)
            
        logger.error(f"No prior results found for {condition_id} at cell {cell}")
                    
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

    def __identifier_generator(self):
        """Generates unique identifier for each simulation in experiment
        output:
            returns the unique identifier
        """

        identifier = str(uuid.uuid4())

        return identifier