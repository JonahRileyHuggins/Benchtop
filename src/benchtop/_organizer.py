"""Task scheduling: topological ordering, cell replication, and round-robin assignment."""

import os
from collections import defaultdict, deque

import pandas as pd


class Organizer:
    """Build and assign simulation tasks respecting preequilibration dependencies."""

    def __init__(self, workers: int = os.cpu_count()):
        self.workers = workers

    def task_organization(
        self,
        measurement_df: pd.DataFrame,
        cell_count: int,
    ) -> tuple[int, dict]:
        """Return (num_rounds, rank_jobs_directory) for the full experiment."""
        size = self.workers

        topological_list = self.topologic_sort(measurements_df=measurement_df)
        total_tasks = self.total_tasks(tasks=topological_list, cell_count=cell_count)
        delayed_list = self.delay_secondary_conditions(
            measurements_df=measurement_df,
            task_list=total_tasks,
            cell_count=cell_count,
        )

        rank_jobs_directory = {}
        for i in range(size):
            rank_ids = self.assign_tasks(i, len(delayed_list))
            rank_jobs_directory[i] = [delayed_list[job] for job in rank_ids]

        rounds_to_complete = -(-len(delayed_list) // size)  # ceiling division
        return rounds_to_complete, rank_jobs_directory

    def topologic_sort(self, measurements_df: pd.DataFrame) -> list:
        """Kahn topological sort of conditions; preequilibration edges first."""
        if "preequilibrationConditionId" not in measurements_df.columns:
            return measurements_df["simulationConditionId"].dropna().unique().tolist()

        sim_nodes = measurements_df["simulationConditionId"].dropna().unique().tolist()
        pre_nodes = measurements_df["preequilibrationConditionId"].dropna().unique().tolist()
        nodes = sorted(set(sim_nodes) | set(pre_nodes))

        succs = defaultdict(list)
        indegree = {n: 0 for n in nodes}

        for _, row in measurements_df.dropna(
            subset=["preequilibrationConditionId"]
        ).iterrows():
            pre = row["preequilibrationConditionId"]
            sim = row["simulationConditionId"]
            succs[pre].append(sim)
            indegree[sim] += 1

        for k in succs:
            succs[k].sort()

        queue = deque(sorted(n for n, d in indegree.items() if d == 0))
        ordered = []

        while queue:
            n = queue.popleft()
            ordered.append(n)

            for m in succs[n]:
                indegree[m] -= 1
                if indegree[m] == 0:
                    queue.append(m)

            queue = deque(sorted(queue))

        if len(ordered) != len(nodes):
            raise RuntimeError("Circular dependency detected among conditions!")

        return ordered

    def delay_secondary_conditions(
        self,
        measurements_df: pd.DataFrame,
        task_list: list,
        cell_count: int,
    ) -> list:
        """Insert None padding so preequilibration completes before dependents."""
        if "preequilibrationConditionId" not in measurements_df.columns:
            return task_list

        pre_conds = (
            measurements_df["preequilibrationConditionId"]
            .drop_duplicates()
            .dropna()
            .to_list()
        )

        for idx, job in enumerate(task_list):
            if job is None:
                continue

            cond_id = job.split("+")[0]
            if cond_id not in pre_conds:
                continue

            pause_ranks = max(self.workers - cell_count, 0)
            while pause_ranks:
                task_list.insert(idx + cell_count, None)
                pause_ranks -= 1

            pre_conds.pop(pre_conds.index(cond_id))

        return task_list

    def total_tasks(self, tasks: list, cell_count: int) -> list:
        """Expand conditions into ``conditionId+cell`` task strings."""
        return [
            f"{cond}+{cell}"
            for cond in tasks
            for cell in range(1, cell_count + 1)
        ]

    def assign_tasks(self, rank: int, total_jobs: int) -> list:
        """Round-robin job indices assigned to a single worker rank."""
        size = self.workers
        num_rounds = -(-total_jobs // size)

        rank_jobs = []
        for round_index in range(num_rounds):
            job_id = rank + round_index * size
            if job_id < total_jobs:
                rank_jobs.append(job_id)

        return rank_jobs

    def task_assignment(self, rank_jobs_directory: dict, round_i: int) -> list:
        """Collect one task per worker for the given round."""
        round_i_tasks = []

        for i in range(self.workers):
            rank_jobs = rank_jobs_directory[i]
            if round_i < len(rank_jobs):
                round_i_tasks.append(rank_jobs[round_i])
            else:
                round_i_tasks.append(None)

        return round_i_tasks
