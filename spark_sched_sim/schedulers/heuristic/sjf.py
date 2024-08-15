import numpy as np

from .heuristic import HeuristicScheduler

class SjfScheduler(HeuristicScheduler):
    def __init__(self, num_executors, resource_allocation, seed=42):
        name = "SJF"
        super().__init__(name)
        self.num_executors = num_executors
        self.set_seed(seed)
        self.resource_allocation = resource_allocation

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
        selected_job_idx = -1
        num_exec = -1
        if schedulable_stages:
            # first, try to find a stage in the same job that is releasing executers
            if source_job_idx < num_active_jobs:
                stage_idx_start = job_ptr[source_job_idx]
                stage_idx_end = job_ptr[source_job_idx + 1]
                if stage_mask[stage_idx_start:stage_idx_end].sum() > 0:
                    selected_job_idx = source_job_idx
                    num_exec = num_committable_execs-1

            if selected_job_idx == -1:
                # find job cpt
                job_cpt = {}
                for job_idx in range(num_active_jobs):
                    #check if the job has any schedulable stages
                    job_stages_cpt = masked_stages_cpt[np.arange(job_ptr[job_idx], job_ptr[job_idx + 1])]
                    if len(job_stages_cpt.nonzero()[0]) > 0 :
                        job_cpt[job_idx] = max(job_stages_cpt)

                if self.resource_allocation == 'DRA':
                    for job in range(num_active_jobs):
                        selected_job_idx = min(job_cpt, key=job_cpt.get)
                        # if obs["DRA_exec_cap"][selected_job_idx] < self.num_executors:
                        #     # increase DRA exec_cap manually for the next job if the job that is releasing executors have the last stage to be processed.
                        #     if job_ptr[1] == 1 and stage_mask[0] == False:
                        #         obs["DRA_exec_cap"][selected_job_idx] = self.num_executors

                        if obs["exec_supplies"][selected_job_idx] >= obs["DRA_exec_cap"][selected_job_idx]:
                            job_cpt[selected_job_idx] = np.inf
                            continue

                    if job_cpt[min(job_cpt)] == np.inf:
                        selected_job_idx = min(job_cpt, key=job_cpt.get)
                        obs["DRA_exec_cap"][selected_job_idx] = self.num_executors

                    num_exec = min(obs["DRA_exec_cap"][selected_job_idx] - obs["exec_supplies"][selected_job_idx],
                                   obs["num_committable_execs"]) - 1
                else:
                    selected_job_idx = min(job_cpt, key=job_cpt.get)

                # for job in range(num_active_jobs):
                #     selected_job_idx = min(job_cpt,key = job_cpt.get)
                #     if self.resource_allocation == 'DRA':
                #         if obs["DRA_exec_cap"][selected_job_idx] < self.num_executors:
                #             # increase DRA exec_cap manually for the next job if the job that is releasing executors have the last stage to be processed.
                #             if job_ptr[1] == 1 and stage_mask[0] == False:
                #                 obs["DRA_exec_cap"][selected_job_idx] = self.num_executors
                #         if obs["exec_supplies"][selected_job_idx] >= obs["DRA_exec_cap"][selected_job_idx]:
                #             job_cpt[selected_job_idx] = np.inf
                #             continue
                #     else:
                #         break
                # if job_cpt[min(job_cpt)] == np.inf:
                #     #print("IN WSCPT: All job exec_cap are reached")
                #     selected_job_idx = min(job_cpt,key = job_cpt.get)
                #     obs["DRA_exec_cap"][selected_job_idx] = self.num_executors
                #
                # if self.resource_allocation == 'DRA':
                #     num_exec = min(obs["DRA_exec_cap"][selected_job_idx] - obs["exec_supplies"][selected_job_idx],
                #                    obs["num_committable_execs"]) - 1

            """searches for a schedulable stage in a given job, prioritizing a node with the shortest cpt"""
            stage_idx_start = job_ptr[selected_job_idx]
            stage_idx_end = job_ptr[selected_job_idx + 1]

            assert max(masked_stages_cpt[stage_idx_start:stage_idx_end]) > 0
            # since the masked stages cpt is shown as"0", the value should be replaced to avoid to be chosen as minimum cpt.
            criteria = masked_stages_cpt[stage_idx_start:stage_idx_end]
            node_selected = np.argmin(np.where(criteria>0,criteria,np.inf)) + stage_idx_start
            selected_stage_idx = schedulable_stages[node_selected]

            return {"stage_idx": selected_stage_idx, "num_exec": num_exec}

        else:
            # didn't find any stages to schedule
            return {"stage_idx": -1, "num_exec": num_committable_execs}
