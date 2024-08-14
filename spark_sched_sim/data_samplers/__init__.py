__all__ = ["TPCHDataSampler", "make_data_sampler"]

from copy import deepcopy

from .tpch import TPCHDataSampler
from .alibaba import AlibabaDataSampler


def make_data_sampler(data_sampler_cfg):
    glob = globals()
    data_sampler_cls = data_sampler_cfg["data_sampler_cls"]
    assert (
        data_sampler_cls in glob
    ), f"'{data_sampler_cls}' is not a valid data sampler."
    return glob[data_sampler_cls](**deepcopy(data_sampler_cfg))
