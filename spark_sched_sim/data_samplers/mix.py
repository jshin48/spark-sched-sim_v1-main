import numpy as np

from .base_data_sampler import BaseDataSampler
from . import alibaba
from . import tpch

class MixDataSampler(BaseDataSampler):
    def __init__(
        self,
        job_arrival_rate: float,
        job_arrival_cap: int,
        num_executors: int,
        warmup_delay: int,
        splitting_rule,
        **kwargs,
    ):
        self.job_arrival_cap = job_arrival_cap
        self.mean_interarrival_time = 1 / job_arrival_rate
        self.np_random = None

        self.tpch_sampler = tpch.TPCHDataSampler(job_arrival_rate,job_arrival_cap,num_executors,warmup_delay,splitting_rule)
        self.alibaba_sampler = alibaba.AlibabaDataSampler(job_arrival_rate,job_arrival_cap,num_executors,warmup_delay,splitting_rule)
        self.alibaba_ratio = 0

    def reset(self, np_random: np.random.RandomState):
        self.np_random = np_random
        self.tpch_sampler.reset(self.np_random)
        self.alibaba_sampler.reset(self.np_random)

    def job_sequence(self, max_time):
        """generates a sequence of job arrivals over time, which follow a
        Poisson process parameterized by `self.job_arrival_rate`
        """
        assert self.np_random
        job_sequence = []

        t = 0
        job_idx = 0
        alibaba_ratio = min(1, self.alibaba_ratio)
        tpch_ratio = 1-alibaba_ratio
        self.mean_interarrival_time = alibaba_ratio * 8 + tpch_ratio * 48

        while t < max_time and (
            not self.job_arrival_cap or job_idx < self.job_arrival_cap
        ):

            job_sampler = np.random.choice([self.tpch_sampler,self.alibaba_sampler], 1, p=[tpch_ratio, alibaba_ratio])[0]
            job = job_sampler._sample_job(job_idx, t)
            job_sequence.append((t, job))

            # sample time in ms until next arrival
            t += self.np_random.exponential(self.mean_interarrival_time)
            job_idx += 1

        self.alibaba_ratio += 0.02

        return job_sequence

    def task_duration(self, job, stage, task, executor): #Called in spark_sched_sim

        if job.sample_type == 'tpch':
            return self.tpch_sampler.task_duration(job, stage, task, executor)
        elif job.sample_type == 'alibaba':
            return self.alibaba_sampler.task_duration(job, stage, task, executor)