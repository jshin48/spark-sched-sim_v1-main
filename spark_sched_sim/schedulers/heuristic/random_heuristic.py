import random as rand

from .heuristic import HeuristicScheduler
import torch_geometric.utils as pyg_utils

from ..scheduler import Scheduler
from spark_sched_sim import graph_utils
from ..heuristic.heuristic import HeuristicScheduler
from ..heuristic.fifo import FifoScheduler
from ..heuristic.wscpt import WscptScheduler
from ..heuristic.mc import McScheduler
from ..heuristic.sjf import SjfScheduler
from ..heuristic.ljf import LjfScheduler
from spark_sched_sim.wrappers import DAGNNObsWrapper


class RandomHeuristic(HeuristicScheduler):
    def __init__(self, num_executors, resource_allocation, num_heuristics=5):
        super().__init__("Randomheuristic")
        self.num_executors = num_executors
        self.resource_allocation = resource_allocation
        self.num_heuristics = num_heuristics
        self.obs_wrapper_cls = DAGNNObsWrapper

        self.mc_scheduler = McScheduler(self.num_executors, self.resource_allocation)
        self.wscpt_scheduler = WscptScheduler(self.num_executors, self.resource_allocation)
        self.sjf_scheduler = SjfScheduler(self.num_executors, self.resource_allocation)
        self.ljf_scheduler = LjfScheduler(self.num_executors, self.resource_allocation)
        self.fifo_scheduler = FifoScheduler(self.num_executors, self.resource_allocation)
    def schedule(self, obs):
        heuristic_idx = rand.randint(0, self.num_heuristics - 1)

        if heuristic_idx == 0:
            return self.mc_scheduler(obs)
        elif heuristic_idx == 1:
            return self.wscpt_scheduler(obs)
        elif heuristic_idx == 2:
            return self.sjf_scheduler(obs)
        elif heuristic_idx == 3:
            return self.ljf_scheduler(obs)
        elif heuristic_idx == 4:
            return self.fifo_scheduler(obs)
        else:
            raise ValueError(f"Invalid heuristic index {heuristic_idx}")


