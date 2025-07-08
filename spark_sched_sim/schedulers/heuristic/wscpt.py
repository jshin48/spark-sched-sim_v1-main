import numpy as np
from .heuristic import HeuristicScheduler
from .heuristic_utils import parse_jobs_from_obs
from spark_sched_sim.wrappers import DAGNNObsWrapper

class WscptScheduler(HeuristicScheduler):
    def __init__(self, num_executors, resource_allocation):
        name = "WSCPT"
        super().__init__(name)
        self.num_executors = num_executors
        self.resource_allocation = resource_allocation
        self.obs_wrapper_cls = DAGNNObsWrapper
        self.heuristic_idx = 0
    def schedule(self, obs):
        jobs = parse_jobs_from_obs(obs)
        num_active_jobs = len(jobs)
        stage_mask = obs["stage_mask"]

        schedulable_stages = dict(zip(stage_mask.nonzero()[0], np.arange(stage_mask.sum())))
        exec_supplies = np.array(obs["exec_supplies"])
        num_committable_execs = obs["num_committable_execs"]
        source_job_idx = obs["source_job_idx"]

        DRA_exec_cap = obs["DRA_exec_cap"]
        dag_ptr = obs["dag_ptr"]

        def compute_num_exec(job_idx):
            if self.resource_allocation == 'DRA':
                dra_cap = DRA_exec_cap.get(job_idx, 0)
                if job_idx < num_active_jobs:
                    return max(min(dra_cap, num_committable_execs), 0)
                else:
                    return max(min(dra_cap - exec_supplies[job_idx], num_committable_execs), 0)
            else:
                return num_committable_execs

        def get_stage_idx(job_idx, stage):
            global_idx = dag_ptr[job_idx] + stage.id_
            return schedulable_stages[global_idx]

        # 1. Local Scheduling: within same job
        if source_job_idx < num_active_jobs:
            local_job = jobs[source_job_idx]
            if local_job.frontier_stages:
                num_exec = compute_num_exec(source_job_idx)
                if num_exec > 0:
                    selected_stage = max(local_job.frontier_stages, key=lambda s: s.cpt)
                    #selected_stage = min(local_job.frontier_stages, key=lambda s: s.remaining_work)
                    return {
                        "heuristic_idx": self.heuristic_idx,
                        "stage_idx": get_stage_idx(source_job_idx, selected_stage),
                        "num_exec": num_exec - 1,
                    }

        # 2. Global SJF: Across all jobs
        candidate_jobs = [
            job for job in jobs if job.frontier_stages and compute_num_exec(job.id_) > 0
        ]

        if not candidate_jobs:
            print("No candidate jobs found for WSCPT scheduling.")
            return {"heuristic_idx" : self.heuristic_idx, "stage_idx": -1, "num_exec": num_committable_execs - 1}

        selected_job = min(candidate_jobs, key=lambda job: job.cpt)
        selected_stage = max(selected_job.frontier_stages, key=lambda s: s.cpt)
        #selected_job = min(candidate_jobs, key=lambda job: job.total_remaining_work)
        #selected_stage = min(selected_job.frontier_stages, key=lambda s: s.remaining_work)

        num_exec = compute_num_exec(selected_job.id_)
        return {
            "heuristic_idx" : self.heuristic_idx,
            "stage_idx": get_stage_idx(selected_job.id_, selected_stage),
            "num_exec": num_exec - 1,
        }
