__all__ = [
    "Scheduler",
    "TrainableScheduler",
    "DecimaScheduler",
    "RandomScheduler",
    "RoundRobinScheduler",
    "make_scheduler",
    "NeuralScheduler",
    "DecimaScheduler",
    "DAGformerScheduler",
    "DAGNNScheduler",
    "HeuristicScheduler",
    "RandomScheduler",
    "RoundRobinScheduler",
    "make_scheduler",
    "HybridHeuristicScheduler",
    "HyperHeuristicScheduler"
    "WscptScheduler",
    "McScheduler",
    "SjfScheduler",
    "LjfScheduler",
    "FifoScheduler"
]

from copy import deepcopy

from schedulers.Hyperheuristics.scheduler import NeuralScheduler
from .neural.hyperheuristic import HyperHeuristicScheduler

from .neural.dagformer import DAGformerScheduler
from .neural.dagnn import DAGNNScheduler

from .heuristic.heuristic import HeuristicScheduler
from .heuristic.random_scheduler import RandomScheduler
from .heuristic.hybridheuristic import HybridHeuristicScheduler
from .heuristic.round_robin import RoundRobinScheduler
from .heuristic.wscpt import WscptScheduler
from .heuristic.mc import McScheduler
from .heuristic.sjf import SjfScheduler
from .heuristic.ljf import LjfScheduler
from .heuristic.fifo import FifoScheduler

def make_scheduler(agent_cfg):
    glob = globals()
    agent_cls = agent_cfg["agent_cls"]
    assert agent_cls in glob, f"'{agent_cls}' is not a valid scheduler."
    return glob[agent_cls](**deepcopy(agent_cfg))
