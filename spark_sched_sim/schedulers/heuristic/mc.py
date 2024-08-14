import numpy as np

from .heuristic import HeuristicScheduler


class McScheduler(HeuristicScheduler):
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
        stage_num_children = obs["dag_batch"].nodes[:,6]
        schedulable_stages = dict(zip(stage_mask.nonzero()[0], np.arange(stage_mask.sum())))
        schedulable_stages_children = dict(zip(stage_mask.nonzero()[0], stage_num_children[stage_mask.nonzero()[0]]))

        exec_supplies = np.array(obs["exec_supplies"])
        num_committable_execs = obs["num_committable_execs"]
        source_job_idx = obs["source_job_idx"]

        num_active_jobs = len(exec_supplies)

        if self.dynamic_partition:
            executor_cap = self.num_executors / max(1, num_active_jobs)
            executor_cap = int(np.ceil(executor_cap))
        else:
            executor_cap = self.num_executors

        selected_stage_idx = -1
        if schedulable_stages:
            # first, try to find a stage in the same job that is releasing executers
            if source_job_idx < num_active_jobs:
                stage_idx_start = job_ptr[source_job_idx]
                stage_idx_end = job_ptr[source_job_idx + 1]
                if stage_mask[stage_idx_start:stage_idx_end].sum() > 0:
                    source_job_schedulable_stages_children = {}
                    for key, value in schedulable_stages_children.items():
                        if stage_idx_start <= key <= stage_idx_end:
                            source_job_schedulable_stages_children[key] = value
                    node_selected = max(source_job_schedulable_stages_children, key=source_job_schedulable_stages_children.get)
                    selected_stage_idx = schedulable_stages[int(node_selected)]

            if selected_stage_idx == -1:
                """searches for a schedulable stage with the maximum children"""
                node_selected = max(schedulable_stages_children, key = schedulable_stages_children.get)
                selected_stage_idx = schedulable_stages[int(node_selected)]
            num_exec = self.np_random.randint(0, num_committable_execs)  #adjustment before taking wrappers
            # print("-- MC : num_committable_execs",num_committable_execs, "num_exec", num_exec)
            return {"stage_idx": selected_stage_idx, "num_exec": num_exec}

        else:
            # didn't find any stages to schedule
            return {"stage_idx": -1, "num_exec": num_committable_execs}
