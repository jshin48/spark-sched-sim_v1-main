import numpy as np

from .heuristic import HeuristicScheduler


class WscptScheduler(HeuristicScheduler):
    def __init__(self, num_executors, dynamic_partition=True, seed=42):
        name = "WSCPT"
        super().__init__(name)
        self.num_executors = num_executors
        self.dynamic_partition = dynamic_partition
        self.set_seed(seed)

    def set_seed(self, seed):
        self.np_random = np.random.RandomState(seed)

    def schedule(self, obs):
        job_ptr = np.array(obs["dag_ptr"])
        stage_mask = obs["stage_mask"]
        stage_cpt = obs["dag_batch"].nodes[:,5]
        schedulable_stages = dict(zip(stage_mask.nonzero()[0], np.arange(stage_mask.sum())))
        masked_stages_cpt = np.multiply(stage_cpt,stage_mask)

        exec_supplies = np.array(obs["exec_supplies"])
        num_committable_execs = obs["num_committable_execs"]
        source_job_idx = obs["source_job_idx"]

        num_active_jobs = len(exec_supplies)

        if self.dynamic_partition:
            executor_cap = self.num_executors / max(1, num_active_jobs)
            executor_cap = int(np.ceil(executor_cap))
        else:
            executor_cap = self.num_executors

        selected_job_idx = -1
        if schedulable_stages:
            # first, try to find a stage in the same job that is releasing executers
            if source_job_idx < num_active_jobs:
                stage_idx_start = job_ptr[source_job_idx]
                stage_idx_end = job_ptr[source_job_idx + 1]
                if stage_mask[stage_idx_start:stage_idx_end].sum() > 0 :
                    selected_job_idx = source_job_idx

            if selected_job_idx == -1:
                # find job cpt
                job_cpt = {}
                for job_idx in range(num_active_jobs):
                    #check if the job has any schedulable stages
                    job_stages_cpt = masked_stages_cpt[np.arange(job_ptr[job_idx], job_ptr[job_idx + 1])]
                    if len(job_stages_cpt.nonzero()[0]) > 0 :
                        job_cpt[job_idx] = max(job_stages_cpt)
                selected_job_idx = min(job_cpt,key = job_cpt.get)

            """searches for a schedulable stage in a given job, prioritizing a node with the longest cpt"""
            stage_idx_start = job_ptr[selected_job_idx]
            stage_idx_end = job_ptr[selected_job_idx + 1]
            if max(masked_stages_cpt[stage_idx_start:stage_idx_end]) == 0:
                print("masked_stages_children:",masked_stages_cpt)
                print("job_ptr:", job_ptr)
                print("selected_job_idx",selected_job_idx)
                print(stage_idx_start,stage_idx_end)

            assert max(masked_stages_cpt[stage_idx_start:stage_idx_end]) > 0
            node_selected = np.argmax(masked_stages_cpt[stage_idx_start:stage_idx_end]) + stage_idx_start
            selected_stage_idx = schedulable_stages[node_selected]
            num_exec = self.np_random.randint(0, num_committable_execs)
            # print("-- WSCPT : num_committable_execs",num_committable_execs, "num_exec", num_exec)
            return {"stage_idx": selected_stage_idx, "num_exec": num_exec}

        else:
            # didn't find any stages to schedule
            return {"stage_idx": -1, "num_exec": num_committable_execs}
