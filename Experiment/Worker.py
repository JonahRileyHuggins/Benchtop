#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script name: Worker.py
Created on Fri Jan. 10th 12:00:00 2024
Author: Jonah R. Huggins

Description: Individual worker method for simulated experiments.

"""
# -----------------------Package Import & Defined Arguements-------------------#
import gc
import logging
import multiprocessing as mp

import numpy as np
import pandas as pd

from Experiment.Manager import Manager
from Experiment.ResultsCacher import ResultCache
from Experiment.AbstractSimulator import AbstractSimulator

logging.basicConfig(
    level=logging.INFO, # Overriden if Verbose Arg. True
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class Worker:

    def __init__(
            self,
            task: str, 
            manager: Manager,
            simulator: AbstractSimulator,
            args: tuple = (), 
        ):
        """
        simulator : AbstractSimulator
            child class of abstract AbstractSimulator Class, defined as a
            wrapper for a particular simulator

        args : tuple, optional
            Extra arguments to pass to function.
        """

        self.manager = manager

        # Store an instance of the simulator in worker class
        self.simulator = simulator(*args)

        # Run individual simulation
        self.__run_task(task)

    def __run_task(self, task: str) -> dict:
            """organized simulation method, executed by each process"""
            rank = mp.current_process().name
            if task is None:
                logger.debug(f"Rank {rank} has no tasks to complete")

                return # No need to save anything if no simulation task

            condition, cell, condition_id = self.manager.condition_cell_id(
                rank_task=task,
                conditions_df=self.manager.problem.condition_files[0]
            )

            logger.info(f"{rank} running {condition_id} for replicate {cell}")

            state_ids = self.simulator.getStateIds()

            precondition_results = self.__extract_preequilibration_results(condition_id, cell)
            if precondition_results:
                self.__setModelState(state_ids, precondition_results)

            self.__setModelState(condition.keys(), condition.values.tolist())

            stop_time = self.__get_simulation_time(condition)
            results_array = self.simulator.simulate(0.0, stop_time, 30.0)

            results = pd.DataFrame(results_array)
            results['time'] = np.arange(0, stop_time, 30)

            parcel = self.__package_results(results, condition_id, cell)

            logger.info(f"{rank} finished {condition_id} for cell {cell}")

            self.__cache_results(parcel)

            # Reset rank internal model after simulation and take out the garbage
            self.simulator = None
            gc.collect()

            logger.info(f"Rank {rank} has completed {condition_id} for process {cell}")

    def __extract_preequilibration_results(
            self, 
            condition_id: str, 
            cell: int
            ) -> list:
        """
        Find if a given condition has a preequilibration. Pulls from results dictionary
        final timepoint array.
        """
    
        # For now, only supporting one problem per file
        measurement_df = self.manager.problem.measurement_files[0]
        precondition_results = []

        if 'preequilibrationConditionId' in measurement_df.columns:
            # Filter matching simulationConditionId
            precondition_matches = measurement_df[
                measurement_df['simulationConditionId'] == condition_id
            ]

            if not precondition_matches.empty:
                # Use iloc[0] to safely get the first preequilibrationConditionId
                precondition_id = precondition_matches['preequilibrationConditionId'].iloc[0]
                
                if pd.notna(precondition_id) and str(precondition_id).strip().lower() != 'nan':

                    logger.debug((
                        f"Extracting preequilibration condition {precondition_id}",
                        f"for condition {condition_id}"
                    ))

                    precondition_df = self.manager.results_lookup(precondition_id, cell)
                    
                    if precondition_df is not None:
                        if "time" in precondition_df.columns: 
                            precondition_df = precondition_df.drop("time", axis = 1)

                        precondition_results = precondition_df.iloc[:, -1]

        return precondition_results


    def __setModelState(self, names: list, states: list) -> None:
        """Set model state with list of floats"""

        for name, state in zip(names, states):
            try:

                self.simulator.modify(name, state)

            except:
                ValueError
                
        logger.debug("Updated model state")

    def __get_simulation_time(
            self, 
            condition: pd.Series
            ) -> float:
        """
        Returns the simulation time for a condition. Raises an error if time is undefined.
        """
        #Only supporting one problem per config file 
        measurement_df = self.manager.problem.measurement_files[0]
        matching_times = measurement_df.loc[
            measurement_df["simulationConditionId"].isin(condition), "time"
        ]

        if matching_times.empty:
            raise ValueError(
                f"No simulation time defined for condition {condition['conditionId']}"
            )

        return matching_times.max()

    def __cache_results(
            self, 
            parcel: dict
            ) -> None:
        """Saves simulation results to cache directory"""

        condition_id = parcel['conditionId']
        cell = parcel["cell"]
        results = parcel['results']

        for key in self.manager.results_dict.keys():

            if self.manager.results_dict[key]['conditionId'] == condition_id \
                and self.manager.results_dict[key]['cell'] == cell:

                # Save results
                cache = ResultCache()
                cache.save(key=key, df=results)

        return # Saves individual simulation data in cache directory

    def __package_results(
            self,
            results: pd.DataFrame,
            condition_id: str,
            cell: str,
        ) -> dict:
        """
        Combines results, condition identifier, and cell number into dict for storage, 
        """

        # make a dict entry in rank_results for every column in results
        rank_results = {
            "conditionId": condition_id,
            "cell": int(cell),
            "results": results,
        }

        return rank_results
