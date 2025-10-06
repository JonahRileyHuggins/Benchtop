#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Curates and handles the organization of tasks for the experiment processes

Author: Jonah R. Huggins
"""
# -----------------------Package Import & Defined Arguements-------------------#
import os
from collections import defaultdict, deque

import pandas as pd


class Organizer:

    def __init__(self, workers = os.cpu_count()):
        self.workers = workers

        
    def task_organization(
        self, 
        measurement_df: pd.DataFrame, 
        cell_count: int
    ) -> dict:
        """Assigns tasks to each rank
        Input:
            rank: int - the rank of the MPI process
            size: int - the number of MPI processes
            details (dict): petab yaml file containing paths to PEtab files.
        Output:
            task_list: list - the list of tasks assigned to the rank
        """

        size = self.workers

        list_of_jobs = self.total_tasks(measurement_df, cell_count)

        rank_jobs_directory = {}

        for i in range(size):
            
            rank_ids = self.assign_tasks(i, len(list_of_jobs))

            jobs_list = [list_of_jobs[job] for job in rank_ids]

            rank_jobs_directory[i] = jobs_list

        # Assign each rank it's task for the round
        rounds_to_complete = -(-len(list_of_jobs) // size)

        return rounds_to_complete, rank_jobs_directory

    def __topo_sort_conditions(
            self,
            measurements_df: pd.DataFrame
            ) -> list:
        """
        Given a DataFrame with columns
        - 'simulationConditionId'
        - 'preequilibrationConditionId'
        return a list of unique simulationConditionIds in dependency order
        (i.e. all pre‐equilibration conditions come before their dependents).
        """
        # 1) Collect all nodes
        nodes = set(measurements_df['simulationConditionId'].dropna()) \
            | set(measurements_df['preequilibrationConditionId'].dropna())
        
        # 2) Build adjacency list and in‐degree map
        succs = defaultdict(list)   # prerequisite → [dependents…]
        indegree = {n: 0 for n in nodes}
        
        for _, row in measurements_df.dropna(subset=['preequilibrationConditionId']).iterrows():
            pre = row['preequilibrationConditionId']
            sim = row['simulationConditionId']
            succs[pre].append(sim)
            indegree[sim] += 1
        
        # 3) Kahn’s algorithm for topological sort
        queue = deque(n for n, d in indegree.items() if d == 0)
        ordered = []
        
        while queue:
            n = queue.popleft()
            ordered.append(n)
            for m in succs[n]:
                indegree[m] -= 1
                if indegree[m] == 0:
                    queue.append(m)
        
        if len(ordered) != len(nodes):
            raise RuntimeError("Circular dependency detected among conditions!")
        
        return ordered

    def __delay_post_conditions(
            self,
            measurement_df: pd.DataFrame, 
            task_list: list, 
            cell_count: int, 
        ) -> list:
        
        """Inserts into topologically sorted task list (with cells) `None` values 
        to delay conditions with pre-conditional dependency simulations"""
        
        size = self.workers

        pre_conds = measurement_df['preequilibrationConditionId'].drop_duplicates().dropna().to_list()
        # Since this is only called after topological sorting via Khan's alg., all 0-order conditions 
        # are first; the task_list is already ordered!

        for idx, job in enumerate(task_list):

            if job == None:
                continue

            cond_id = job.split("+")[0]

            if cond_id in pre_conds:

                pause_ranks = max((size - cell_count), 0)

                while pause_ranks:

                    task_list.insert(idx+cell_count, None)

                    pause_ranks -= 1

                pre_conds.pop(pre_conds.index(cond_id))

        return task_list


    def total_tasks(
            self,
            measurements_df: pd.DataFrame, 
            cell_count: int
            ) -> list:
        """Calculate the total number of tasks from the measurement dataframe"""

        if 'preequilibrationConditionId' in measurements_df.columns:
            # Reorder conditions with 0-order dependencies first:
            ordered_conditions = self.__topo_sort_conditions(measurements_df)

        else:
            ordered_conditions = measurements_df['simulationConditionId'].unique().tolist()

        list_of_jobs = []
        for cond in ordered_conditions:
            for cell in range(1, cell_count + 1):
                list_of_jobs.append(f"{cond}+{cell}")

        if 'preequilibrationConditionId' in measurements_df.columns: 
            # Add delays for dependent conditions & cells; requires cell number in job ID
            list_of_jobs = self.__delay_post_conditions(
                measurements_df,
                list_of_jobs, 
                cell_count
                )

        return list_of_jobs

    def assign_tasks(
            self, 
            rank: int, 
            total_jobs: int, 
            ) -> list:
        """
        Assign tasks to a given rank in round-robin fashion, including those that are part
        of sequential chains. Ensures layering of job dependencies, avoids
        oversubscription what the longest sequential dependency chain requires.
        """
        size = self.workers
        num_rounds = -(-total_jobs // size)  # Equivalent to math.ceil(total_jobs / size)
        
        rank_jobs = []
        
        for round_index in range(num_rounds):
            job_id = rank + round_index * size
            if job_id < total_jobs:
                rank_jobs.append(job_id)

        return rank_jobs

    def task_assignment(
        self,
        rank_jobs_directory: dict,
        round_i: int
    ) -> list:
        size = self.workers

        round_i_tasks = []

        # Assign each rank it's task for the round
        for i in range(size):

            rank_jobs = rank_jobs_directory[i]

            if round_i < len(rank_jobs):
                rank_job_for_round = rank_jobs[round_i]

            else:
                rank_job_for_round = None

            round_i_tasks.append(rank_job_for_round)


        return round_i_tasks
