import numpy as np
from .heuristic import HeuristicScheduler

class WscptScheduler(HeuristicScheduler):
    def __init__(self, num_executors, resource_allocation):
        super().__init__("WSCPT")
        self.num_executors = num_executors
        self.resource_allocation = resource_allocation

    def schedule(self, obs):
        job_ptr = np.array(obs["dag_ptr"])
        stage_mask = obs["stage_mask"]
        stage_cpt = obs["dag_batch"].nodes[:,5]
        masked_stages_cpt = np.multiply(stage_cpt,stage_mask)
        num_active_jobs = len(obs["exec_supplies"])
        num_committable_execs = obs["num_committable_execs"]
        source_job_idx = obs["source_job_idx"]

        def get_job_cpt():
            return {job_idx: max(masked_stages_cpt[job_ptr[job_idx]:job_ptr[job_idx + 1]])
                    for job_idx in range(num_active_jobs)
                    if masked_stages_cpt[job_ptr[job_idx]:job_ptr[job_idx + 1]].sum() > 0}

        if source_job_idx < num_active_jobs and stage_mask[job_ptr[source_job_idx]:job_ptr[source_job_idx + 1]].sum() > 0:
            selected_job_idx = source_job_idx
            num_exec = min(obs["DRA_exec_cap"][selected_job_idx], num_committable_execs) - 1 if self.resource_allocation == 'DRA' else num_committable_execs - 1
        else:
            job_cpt = get_job_cpt()
            selected_job_idx = min(job_cpt, key=job_cpt.get)
            if self.resource_allocation == 'DRA':
                while obs["exec_supplies"][selected_job_idx] >= obs["DRA_exec_cap"][selected_job_idx]:
                    job_cpt[selected_job_idx] = np.inf
                    selected_job_idx = min(job_cpt, key=job_cpt.get)
                if job_cpt[selected_job_idx] == np.inf:
                    obs["DRA_exec_cap"][selected_job_idx] = self.num_executors
                num_exec = min(obs["DRA_exec_cap"][selected_job_idx] - obs["exec_supplies"][selected_job_idx], num_committable_execs) - 1
            else:
                num_exec = num_committable_execs - 1

        stage_idx_start, stage_idx_end = job_ptr[selected_job_idx], job_ptr[selected_job_idx + 1]
        selected_stage_idx = np.argmax(masked_stages_cpt[stage_idx_start:stage_idx_end]) + stage_idx_start

        return {"stage_idx": selected_stage_idx, "num_exec": num_exec} if masked_stages_cpt[stage_idx_start:stage_idx_end].sum() > 0 else {"stage_idx": -1, "num_exec": num_committable_execs}