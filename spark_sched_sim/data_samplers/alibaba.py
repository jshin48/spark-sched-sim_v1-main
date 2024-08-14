import os.path as osp
import sys
import pathlib
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

import numpy as np
import networkx as nx

from .base_data_sampler import BaseDataSampler
from ..components import Job, Stage

QUERY_SIZES = ["j_10_test"]
NUM_QUERIES = 28963


class AlibabaDataSampler(BaseDataSampler):
    def __init__(
        self,
        job_arrival_rate: float,
        job_arrival_cap: int,
        num_executors: int,
        warmup_delay: int,
        splitting_rule,
        **kwargs,
    ):
        """
        job_arrival_rate (float): non-negative number that controls how
            quickly new jobs arrive into the system. This is the parameter
            of an exponential distributions, and so its inverse is the
            mean job inter-arrival time in ms.
        job_arrival_cap: (optional int): limit on the number of jobs that
            arrive throughout the simulation. If set to `None`, then the
            episode ends when a time limit is reached.
        num_executors (int): number of simulated executors. More executors
            means a higher possible level of parallelism.
        warmup_delay (int): an executor is slower on its first task from
            a stage if it was previously idle or moving jobs, which is
            caputred by adding a warmup delay (ms) to the task duration
        """
        self.job_arrival_cap = job_arrival_cap
        self.mean_interarrival_time = 1 / job_arrival_rate
        self.warmup_delay = warmup_delay
        self.splitting_rule = splitting_rule
        self.np_random = None

        if not osp.isdir("data/alibaba"):
            print("Alibaba data is unavailable")

    def reset(self, np_random: np.random.RandomState):
        self.np_random = np_random

    def job_sequence(self, max_time):
        """generates a sequence of job arrivals over time, which follow a
        Poisson process parameterized by `self.job_arrival_rate`
        """
        assert self.np_random
        job_sequence = []

        t = 0
        job_idx = 0
        while t < max_time and (
            not self.job_arrival_cap or job_idx < self.job_arrival_cap
        ):
            job = self._sample_job(job_idx, t)
            job_sequence.append((t, job))

            # sample time in ms until next arrival
            t += self.np_random.exponential(self.mean_interarrival_time)
            job_idx += 1

        return job_sequence

    def task_duration(self, job, stage, task, executor): #Called in spark_sched_sim
        num_local_executors = len(job.local_executors)

        assert num_local_executors > 0
        assert self.np_random

        if executor.is_idle:
            # the executor was just sitting idly or moving between jobs, so it needs time to warm up
            return stage.most_recent_duration + self.warmup_delay

        if executor.task.stage_id == task.stage_id:
            # the executor is continuing work on the same stage, which is relatively fast
            return stage.most_recent_duration

            # the executor is new to this stage (or 'rest_wave' data was not available)
        return stage.most_recent_duration + self.warmup_delay

    @classmethod
    def _load_query(cls, query_num, query_size):
        query_path = osp.join("data/alibaba", str(query_size))

        adj_matrix = np.load(
            osp.join(query_path, f"adj_mat_{query_num}.npy"), allow_pickle=True
        )

        task_duration_data = np.load(
            osp.join(query_path, f"ins_dur{query_num}.npy"), allow_pickle=True
        )

        assert adj_matrix.shape[0] == adj_matrix.shape[1]
        assert adj_matrix.shape[0] == len(task_duration_data)

        return adj_matrix, task_duration_data

    def _sample_job(self, job_id, t_arrival):
        adj_mat = np.array([0])
        while len(np.stack(adj_mat.nonzero(),axis=-1)) == 0:  #Need to sample a DAG with at least two stages
            query_num = 1 + self.np_random.integers(NUM_QUERIES)
            query_size = self.np_random.choice(QUERY_SIZES)
            adj_mat, task_duration_data = self._load_query(query_num, query_size)

        num_stages = adj_mat.shape[0]
        stages = []

        #calculate cpt
        if self.splitting_rule is None:
            task_duration_list = [task_duration_data[stage_id][1] for stage_id in range(num_stages)]
            num_tasks_list = [int(task_duration_data[stage_id][0]) for stage_id in range(num_stages)]
        elif type(self.splitting_rule) == int or self.splitting_rule == "DTS":
            task_duration_list = [task_duration_data[stage_id][1] * int(task_duration_data[stage_id][0]) for stage_id in range(num_stages)]
            num_tasks_list = [1 for stage_id in range(num_stages)]
        else:
            sys.exit("splitting_rule is invalid @alibaba line 127")

        cpt, num_children = self._get_node_feature(task_duration_list, adj_mat)

        for stage_id in range(num_stages):
            num_tasks = num_tasks_list[stage_id]
            task_duration = max(task_duration_list[stage_id], 1) #Some tasks have duration less than 1, which recorded as 0 in alibaba data.
            rough_duration = task_duration

            stage = Stage(stage_id, job_id, num_tasks, rough_duration,cpt[stage_id], num_children[stage_id])
            stages += [stage]


        # generate DAG
        dag = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)
        for _, _, d in dag.edges(data=True):
            d.clear()

        #print("adj_mat:",adj_mat, "/dag.edges:",dag.edges)

        job = Job(job_id, stages, dag, t_arrival,np.max(cpt))
        job.query_num = query_num
        job.query_size = query_size

        summary = {"job_idx":job_id, "adj_mat": adj_mat, "task_duration": task_duration_data, "cpt": [stage.cpt for stage in stages]}
        #print("job_idx:",job_id, "adj_mat:",adj_mat, "num_tasks:",num_tasks_list, "task_duration_data:",task_duration_data, "task_duration:",task_duration_list)

        return job

    # Return cpt of each node in a DAG
    def _get_node_feature(self, task_duration_list, adj_mat):
        num_node = adj_mat.shape[0]
        num_children = [0] * num_node
        children_idx = [0] * num_node

        #Find direct children to each stage
        for stage_id in range(num_node):
            children_idx[stage_id] = adj_mat[stage_id].nonzero()[0]
            num_children[stage_id] =  children_idx[stage_id].size

        # Find cpt of each stage
        each_task_duration = [max(task_duration_list[stage_id], 1) for stage_id in range(num_node)]
        cpt = np.zeros(num_node)

        cpt_updated_count = [0] * num_node
        for stage_id in range(num_node):
            if children_idx[stage_id].size == 0:
                cpt[stage_id] = each_task_duration[stage_id]
                cpt_updated_count[stage_id] = 1

        while sum(cpt_updated_count) < num_node:
            for stage_id in range(num_node):
                if np.all(cpt[children_idx[stage_id]] > 0) and cpt_updated_count[stage_id] == 0:
                        cpt[stage_id] = np.max(cpt[children_idx[stage_id]]) + each_task_duration[stage_id]
                        cpt_updated_count[stage_id] = 1

        return cpt, num_children

    # def _init_executor_intervals(self, exec_cap):
    #     exec_levels = [5, 10, 20, 40, 50, 60, 80, 100]
    #
    #     intervals = np.zeros((exec_cap + 1, 2))
    #
    #     # get the left most map
    #     intervals[: exec_levels[0] + 1] = exec_levels[0]
    #
    #     # get the center map
    #     for i in range(len(exec_levels) - 1):
    #         intervals[exec_levels[i] + 1 : exec_levels[i + 1]] = (
    #             exec_levels[i],
    #             exec_levels[i + 1],
    #         )
    #
    #         if exec_levels[i + 1] > exec_cap:
    #             break
    #
    #         # at the data point
    #         intervals[exec_levels[i + 1]] = exec_levels[i + 1]
    #
    #     # get the residual map
    #     if exec_cap > exec_levels[-1]:
    #         intervals[exec_levels[-1] + 1 : exec_cap] = exec_levels[-1]
    #
    #     self.executor_intervals = intervals

#
# class MultiSet:
#     """
#     allow duplication in set
#     """
#
#     def __init__(self):
#         self.set = {}
#
#     def __contains__(self, item):
#         return item in self.set
#
#     def add(self, item):
#         if item in self.set:
#             self.set[item] += 1
#         else:
#             self.set[item] = 1
#
#     def clear(self):
#         self.set.clear()
#
#     def remove(self, item):
#         self.set[item] -= 1
#         if self.set[item] == 0:
#             del self.set[item]
