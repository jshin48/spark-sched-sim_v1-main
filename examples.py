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

ENV_CFG = {
    "num_executors": 2,
    "job_arrival_cap": 5,
    "job_arrival_rate": 8.e-5, #for alibaba 8.e-6, #for tpch 4.e-5
    "moving_delay": 1000,#2000,
    "warmup_delay": 0,#1000,
    "pod_creation_time": 10,#100.,
    "train_data": "alibaba",
    "test_data" : "alibaba",
    "render_mode": "human", #human
    "hybrid_rule_threshold" : 4,
    "decima_model_name": "model_100_DNN",
    "hyper_model_name": "model_5_DNN_None/100/model", #The name of the model should match with resource allocation param
    # "num_heuristics" : 5,
    # "list_heuristics": ['MC','WSCPT'],
    "resource_allocation": "DNN", #HyperHeuristic, Random, DNN, DRA
    "num_resource_heuristics" : 3,
    "list_resource_heuristics" : ['Maximum','DRA','fair'],
    "splitting_rule": "None" #DTS, int, "None"
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
    decima_example()
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
    cfg = load(filename=osp.join("config", "decima_alibaba.yaml"))
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
    avg_job_duration = run_episode(ENV_CFG|{"agent_cls": agent_cfg["agent_cls"]}, scheduler)

    print(f"Done! Average job duration: {avg_job_duration:.1f}s", flush=True)

def hyperheuristic_example():
    cfg = load(filename=osp.join("config", "hyperheuristic_"+ ENV_CFG["test_data"]+".yaml"))
    agent_cfg = cfg["agent"] | {
        "num_executors": ENV_CFG["num_executors"],
        "state_dict_path": Path("models/hyperheuristic/" +ENV_CFG["train_data"]+ "/"+ ENV_CFG["hyper_model_name"] +".pt"),
        #"state_dict_path": osp.join("models", "hyperheuristic", ENV_CFG["train_data"], ENV_CFG["hyper_model_name"]+".pt"),
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
    env_cfg["plot_title"]= Path("./test_results/"+"graph_hyper.png")
    #env_cfg["plot_title"] = Path("./test_results/"+ENV_CFG["train_data"]+"/hyper/hyper_" +ENV_CFG["train_data"]+"_"+ENV_CFG["test_data"] +
    # "_" + ENV_CFG["hyper_model_name"] + "_" + str(ENV_CFG["splitting_rule"])+".png")
    pprint(env_cfg)

    print("Running episode...")
    avg_job_duration = run_episode(env_cfg|{"agent_cls": agent_cfg["agent_cls"]}, scheduler)

    print(f"Done! Average job duration: {avg_job_duration:.1f}s", flush=True)

def hybridheuristic_example():
    cfg = load(filename=osp.join("config", "hyperheuristic_"+ENV_CFG["test_data"]+".yaml"))

    env_cfg = dict(ENV_CFG)
    env_cfg["num_heuristics"] = cfg["env"]["num_heuristics"]
    env_cfg["list_heuristics"] = cfg["env"]["list_heuristics"]
    env_cfg["plot_title"] = "Hybrid_"+ENV_CFG["test_data"]+"_k"+str(ENV_CFG["hybrid_rule_threshold"])+".png"

    scheduler = HybridheuristicScheduler(ENV_CFG["num_executors"],ENV_CFG["hybrid_rule_threshold"])

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
