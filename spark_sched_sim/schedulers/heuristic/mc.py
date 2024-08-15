import numpy as np

from .heuristic import HeuristicScheduler

class McScheduler(HeuristicScheduler):
    def __init__(self, num_executors, resource_allocation, seed=42):
        name = "MC"
        super().__init__(name)
        self.num_executors = num_executors
        self.set_seed(seed)
        self.resource_allocation = resource_allocation

    def set_seed(self, seed):
        self.np_random = np.random.RandomState(seed)

    def schedule(self, obs):
        job_ptr = np.array(obs["dag_ptr"])
        stage_mask = obs["stage_mask"]  # List of True or False if a corresponding node is schedulable. see "spark_sched_sim.py 439"
        stage_num_children = obs["dag_batch"].nodes[:, 6]
        stage_indices = stage_mask.nonzero()[0]
        schedulable_stages = {stage_idx: idx for idx, stage_idx in enumerate(stage_indices)}
        schedulable_stages_children = {stage_idx: stage_num_children[stage_idx] for stage_idx in stage_indices}

        # schedulable_stages = dict(zip(stage_mask.nonzero()[0], np.arange(stage_mask.sum())))
        # schedulable_stages_children = dict(zip(stage_mask.nonzero()[0], stage_num_children[stage_mask.nonzero()[0]]))

        exec_supplies = obs["exec_supplies"]
        num_committable_execs = obs["num_committable_execs"]
        source_job_idx = obs["source_job_idx"]
        exec_cap = obs["DRA_exec_cap"]

        num_active_jobs = len(np.array(exec_supplies))

        if schedulable_stages:
            # first, try to find a stage in the same job that is releasing executors
            if source_job_idx < num_active_jobs:
                stage_idx_start = job_ptr[source_job_idx]
                stage_idx_end = job_ptr[source_job_idx + 1]

                # If source job has at least one schedulable stage, assign a stage with the maximum child
                if stage_mask[stage_idx_start:stage_idx_end].sum() > 0:
                    max_value = -float('inf')
                    node_selected = None

                    for key in schedulable_stages_children:
                        if stage_idx_start <= key < stage_idx_end and schedulable_stages_children[key] > max_value:
                            max_value = schedulable_stages_children[key]
                            node_selected = key

                    selected_stage_idx = schedulable_stages[int(node_selected)]

                    return {"stage_idx": selected_stage_idx, "num_exec": num_committable_execs-1}

                # searches for a schedulable stage with the maximum children
            for _ in range(len(schedulable_stages)):
                node_selected = max(schedulable_stages_children, key = schedulable_stages_children.get)
                selected_stage_idx = schedulable_stages[int(node_selected)]

                if self.resource_allocation == 'DRA':
                    # Find the job id of the selected stage
                    for i in range(len(job_ptr)):
                        if node_selected < job_ptr[i+1]:
                            selected_job_idx = i
                            break
                        else:
                            continue

                    if exec_supplies[selected_job_idx] >= exec_cap[selected_job_idx]:
                        schedulable_stages_children[node_selected] = -1
                        continue
                    else:
                        num_exec = min(exec_cap[selected_job_idx] - exec_supplies[selected_job_idx], num_committable_execs)-1
                    return {"stage_idx": selected_stage_idx, "num_exec": num_exec}
                else:
                    # resource allocation will be determined in "neural.py"
                    return {"stage_idx": selected_stage_idx, "num_exec": -1}
            return {"stage_idx": -1, "num_exec": num_committable_execs}

        else:
            # didn't find any stages to schedule
            return {"stage_idx": -1, "num_exec": num_committable_execs}
