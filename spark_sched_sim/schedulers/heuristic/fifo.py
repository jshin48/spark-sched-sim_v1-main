import numpy as np

from .heuristic import HeuristicScheduler


class FifoScheduler(HeuristicScheduler):
    def __init__(self, num_executors, resource_allocation):
        name = "FIFO"
        super().__init__(name)
        self.num_executors = num_executors
        self.resource_allocation = resource_allocation

    def schedule(self, obs):
        obs = self.preprocess_obs(obs)
        num_active_jobs = len(obs.exec_supplies)
        num_exec = -1
        if obs.schedulable_stages:
            # first, try to find a stage in the same job that is releasing executers
            if obs.source_job_idx < num_active_jobs:
                selected_stage_idx = self.find_stage(obs, obs.source_job_idx)

                if selected_stage_idx != -1:
                    return {
                        "stage_idx": selected_stage_idx,
                        "num_exec": obs.num_committable_execs-1,
                    }

            # search through jobs by order of arrival.
            if self.resource_allocation == 'DRA':
                for selected_job_idx in range(num_active_jobs):
                    if obs.exec_supplies[selected_job_idx] >= obs.DRA_exec_cap[selected_job_idx]:
                        continue

                    selected_stage_idx = self.find_stage(obs, selected_job_idx)
                    if selected_stage_idx == -1:
                        print("job", selected_job_idx,"do not have schedulable stages")
                        continue
                    else:
                        num_exec = min(obs.num_committable_execs, obs.DRA_exec_cap[selected_job_idx] - obs.exec_supplies[selected_job_idx])-1
                        return {"stage_idx": selected_stage_idx, "num_exec": num_exec}

                #print({ "selected job idx":selected_job_idx,"stage_idx": selected_stage_idx, "num_exec": num_exec})
            else:
                for selected_job_idx in range(num_active_jobs):
                    selected_stage_idx = self.find_stage(obs, selected_job_idx)
                    if selected_stage_idx == -1:
                        continue
                    else:
                        return {"stage_idx": selected_stage_idx, "num_exec": -1}

        # didn't find any stages to schedule
        return {"stage_idx": -1, "num_exec": obs.num_committable_execs}
