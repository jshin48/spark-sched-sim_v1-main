import numpy as np

from .heuristic import HeuristicScheduler
from .mc import McScheduler
from .wscpt import WscptScheduler
from spark_sched_sim.wrappers import DAGNNObsWrapper



class HybridHeuristicScheduler(HeuristicScheduler):
    def __init__(self, num_executors, resource_allocation, rule_switch_threshold=4, seed=42):
        super().__init__("Hybridheuristic")
        self.num_executors = num_executors
        self.resource_allocation = resource_allocation
        self.set_seed(seed)

        self.obs_wrapper_cls = DAGNNObsWrapper
        self.mc_scheduler = McScheduler(self.num_executors, self.resource_allocation)
        self.wscpt_scheduler = WscptScheduler(self.num_executors, self.resource_allocation)
        self.rule_switch_threshold = rule_switch_threshold

    def set_seed(self, seed):
        self.np_random = np.random.RandomState(seed)

    def schedule(self, obs):
        #obs = self.preprocess_obs(obs)
        num_queue = len(obs["stage_mask"].nonzero()[0])
        if num_queue < self.rule_switch_threshold:
            # print("mc is chosen with obs: \n",obs )
            return self.mc_scheduler(obs)

        else:
            # print("wscpt is chosen with obs: \n",obs )
            return self.wscpt_scheduler(obs)