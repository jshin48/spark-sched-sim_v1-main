__all__ = [
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

]

from copy import deepcopy

from .neural.neural import NeuralScheduler
from .neural.decima import DecimaScheduler
from .neural.hyperheuristic import HyperHeuristicScheduler

from .neural.dagformer import DAGformerScheduler
from .neural.dagnn import DAGNNScheduler

from .heuristic.heuristic import HeuristicScheduler
from .heuristic.random_scheduler import RandomScheduler
from .heuristic.hybridheuristic import HybridHeuristicScheduler



def make_scheduler(agent_cfg):
    glob = globals()
    agent_cls = agent_cfg["agent_cls"]
    assert agent_cls in glob, f"'{agent_cls}' is not a valid scheduler."
    return glob[agent_cls](**deepcopy(agent_cfg))
