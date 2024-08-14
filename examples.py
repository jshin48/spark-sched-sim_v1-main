"""Examples of how to run job scheduling simulations with different schedulers
"""
import os.path as osp
from pathlib import Path
from pprint import pprint

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import gymnasium as gym
import pathlib, sys

from cfg_loader import load
from spark_sched_sim.schedulers import (
    RoundRobinScheduler,
    NeuralScheduler,
    make_scheduler,
    HybridheuristicScheduler,
)
from spark_sched_sim.wrappers import NeuralActWrapper
from spark_sched_sim import metrics

# ENV_CFG = {
#     "num_executors": 20,
#     "job_arrival_cap": 200,
#     "job_arrival_rate": 4.e-5,
#     "moving_delay": 2000.0,
#     "warmup_delay": 1000.0,
#     "test_data" : "tpch", #alibaba or tpch
#     "render_mode": "human"
# }

ENV_CFG = {
    "num_executors": 40,
    "job_arrival_cap": 100,
    "job_arrival_rate": 8.e-5, #for alibaba 8.e-6, #for tpch 4.e-5
    "moving_delay": 10,
    "warmup_delay": 10,
    "pod_creation_time": 1.,
    "train_data": "alibaba",
    "test_data" : "alibaba",
    "render_mode": "human",
    "rule_switch_threshold" : 4,
    "hyper_model_name": "model_100_DNN", #The name of the model should match with resource allocation param
    "decima_model_name": "model_40_rand_exec",
    "resource_allocation": "DNN",
    "splitting_rule": None #DTS, int, None
}

def main():
    # save final rendering to artifacts dir
    pathlib.Path("artifacts").mkdir(parents=True, exist_ok=True)

    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)

    # parser.add_argument(
    #     "--sched",
    #     choices=["fair", "decima"],
    #     dest="sched",
    #     help="which scheduler to run",
    #     required=True,
    # )
    #
    # args = parser.parse_args()
    #
    # sched_map = {"fair": fair_example, "decima": decima_example}
    #
    # sched_map[args.sched]()

    #fair_example()
    #decima_example()
    hyperheuristic_example()
    #hybridheuristic_example()

def fair_example():
    # Fair scheduler
    scheduler = RoundRobinScheduler(ENV_CFG["num_executors"], dynamic_partition=True)

    print("Example: Fair Scheduler")
    print("Env settings:")
    pprint(ENV_CFG)

    print("Running episode...")
    avg_job_duration = run_episode(ENV_CFG, scheduler)

    print(f"Done! Average job duration: {avg_job_duration:.1f}s", flush=True)
    print()

def decima_example():
    cfg = load(filename=osp.join("config", "decima_tpch.yaml"))
    agent_cfg = cfg["agent"] | {
        "num_executors": ENV_CFG["num_executors"],
        "state_dict_path": osp.join("models", "decima", ENV_CFG["train_data"], ENV_CFG["decima_model_name"]+".pt"),
        "resource_allocation": cfg["env"]["resource_allocation"],
    }
    scheduler = make_scheduler(agent_cfg)

    print("Example: Decima")
    print("Env settings:")
    ENV_CFG["plot_title"] = Path("./test_results/"+ENV_CFG["train_data"]+"/decima/decima_" +ENV_CFG["train_data"]+"_"+ENV_CFG["test_data"] +
                                 "_" + ENV_CFG["decima_model_name"] + "_" + str(ENV_CFG["splitting_rule"])+".png")
    pprint(ENV_CFG)

    print("Running episode...")
    avg_job_duration = run_episode(ENV_CFG, scheduler)

    print(f"Done! Average job duration: {avg_job_duration:.1f}s", flush=True)

def hyperheuristic_example():
    cfg = load(filename=osp.join("config", "hyperheuristic_"+ ENV_CFG["test_data"]+".yaml"))
    agent_cfg = cfg["agent"] | {
        "num_executors": ENV_CFG["num_executors"],
        "state_dict_path": osp.join("models", "hyperheuristic", ENV_CFG["train_data"], ENV_CFG["hyper_model_name"]+".pt"),
        "num_heuristics": cfg["env"]["num_heuristics"],
        "list_heuristics": cfg["env"]["list_heuristics"],
        "resource_allocation": cfg["env"]["resource_allocation"],
    }
    scheduler = make_scheduler(agent_cfg)

    print("Example: Hyper-Heuristic (DRL)")
    print("Env settings:")
    env_cfg = dict(ENV_CFG)
    env_cfg["num_heuristics"] = cfg["env"]["num_heuristics"]
    env_cfg["list_heuristics"] = cfg["env"]["list_heuristics"]
    env_cfg["plot_title"] = Path("./test_results/"+ENV_CFG["train_data"]+"/hyper/hyper_" +ENV_CFG["train_data"]+"_"+ENV_CFG["test_data"] +
                                 "_" + ENV_CFG["hyper_model_name"] + "_" + str(ENV_CFG["splitting_rule"])+".png")
    pprint(env_cfg)

    print("Running episode...")
    avg_job_duration = run_episode(env_cfg, scheduler)

    print(f"Done! Average job duration: {avg_job_duration:.1f}s", flush=True)

def hybridheuristic_example():
    cfg = load(filename=osp.join("config", "hyperheuristic_"+ENV_CFG["test_data"]+".yaml"))

    env_cfg = dict(ENV_CFG)
    env_cfg["num_heuristics"] = cfg["env"]["num_heuristics"]
    env_cfg["list_heuristics"] = cfg["env"]["list_heuristics"]
    env_cfg["plot_title"] = "Hybrid_"+ENV_CFG["test_data"]+"_k"+str(ENV_CFG["rule_switch_threshold"])+".png"

    scheduler = HybridheuristicScheduler(ENV_CFG["num_executors"],ENV_CFG["rule_switch_threshold"])

    print("Example: Hybrid-Heuristic")
    print("Env settings:")
    pprint(env_cfg)

    print("Running episode...")
    avg_job_duration = run_episode(env_cfg, scheduler)

    print(f"Done! Average job duration: {avg_job_duration:.1f}s", flush=True)

def run_episode(env_cfg, scheduler, seed=1234):
    if env_cfg["test_data"] == "alibaba":
         env_cfg["data_sampler_cls"]= "AlibabaDataSampler"
    elif env_cfg["test_data"] == "tpch":
        env_cfg["data_sampler_cls"] = "TPCHDataSampler"
    else:
        sys.exit("Check the test data")
    env = gym.make("spark_sched_sim:SparkSchedSimEnv-v0", env_cfg=env_cfg)

    if isinstance(scheduler, NeuralScheduler) or isinstance(scheduler, HybridheuristicScheduler):
        env = NeuralActWrapper(env)
        env = scheduler.obs_wrapper_cls(env)

    obs, _ = env.reset(seed=seed, options=None)
    terminated = truncated = False

    while not (terminated or truncated):
        if isinstance(scheduler, NeuralScheduler):
            action, *_ = scheduler(obs)
        else:
            action = scheduler(obs)
        obs, _, terminated, truncated, _ = env.step(action)

    avg_job_duration = metrics.avg_job_duration(env) * 1e-3
    #metrics.print_task_job_time(env)
    # cleanup rendering
    env.close()

    return avg_job_duration

if __name__ == "__main__":
    main()
