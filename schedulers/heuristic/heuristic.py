from typing import NamedTuple

import numpy as np

from ..scheduler import Scheduler


class HeuristicObs(NamedTuple):
    job_ptr: np.ndarray
    frontier_stages: set
    schedulable_stages: dict
    exec_supplies: np.ndarray
    num_committable_execs: int
    source_job_idx: int
    DRA_exec_cap : dict
    stage_mask : np.ndarray


class HeuristicScheduler(Scheduler):
    """Base class for all heuristic schedulers"""

    @classmethod
    def preprocess_obs(cls, obs):
        frontier_mask = np.ones(obs["dag_batch"].nodes.shape[0], dtype=bool)
        dst_nodes = obs["dag_batch"].edge_links[:, 1] # destination nodes: children nodes of uncompleted nodes in the queue.
        frontier_mask[dst_nodes] = False # children nodes cannot be scheduled, yet.
        # frontier_stages identify the frontier node number
        # ex) frontier_stages = {1, 3, 4, 7} if frontier_mask = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        frontier_stages = set(frontier_mask.nonzero()[0])

        job_ptr = np.array(obs["dag_ptr"])
        try: # For hyperheuristics
            stage_mask = obs["stage_mask"]
        except KeyError:
            stage_mask = obs["dag_batch"].nodes[:, 2].astype(bool)
        schedulable_stages = dict(
            zip(stage_mask.nonzero()[0], np.arange(stage_mask.sum()))
        )
        exec_supplies = np.array(obs["exec_supplies"])
        num_committable_execs = obs["num_committable_execs"]
        source_job_idx = obs["source_job_idx"]
        DRA_exec_cap = obs["DRA_exec_cap"]
        stage_mask = obs["stage_mask"]

        return HeuristicObs(
            job_ptr,
            frontier_stages,
            schedulable_stages,
            exec_supplies,
            num_committable_execs,
            source_job_idx,
            DRA_exec_cap,
            stage_mask
        )

    @classmethod
    def find_stage(cls, obs, job_idx):
        """searches for a schedulable stage in a given job, prioritizing
        frontier stages
        """
        stage_idx_start = obs.job_ptr[job_idx]
        stage_idx_end = obs.job_ptr[job_idx + 1]

        selected_stage_idx = -1
        for node in range(stage_idx_start, stage_idx_end):
            try:
                i = obs.schedulable_stages[node]
            except KeyError:
                continue

            # froniter_stages are stages that doesn't have any children at all (whether it's already processed or not)
            if node in obs.frontier_stages:
                return i

            if selected_stage_idx == -1:
                selected_stage_idx = i

        return selected_stage_idx
