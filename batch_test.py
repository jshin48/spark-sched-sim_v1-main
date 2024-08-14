from examples import run_episode

import os.path as osp
from pathlib import Path
from pprint import pprint

from cfg_loader import load
from spark_sched_sim.schedulers import (
    RoundRobinScheduler,
    NeuralScheduler,
    make_scheduler,
    HybridheuristicScheduler,
)

ENV_CFG = {
    "num_executors": 20,
    "job_arrival_cap": 300,
    "job_arrival_rate": 8.e-6, #for alibaba 8.e-6, #for tpch 4.e-5
    "moving_delay": 2000,
    "warmup_delay": 1000,
    "pod_creation_time": 1.,
    "train_data": "alibaba",
    "test_data" : "alibaba",
    "render_mode": None,
    "rule_switch_threshold" : 4,
    "hyper_model_name": "model_100_DNN", #The name of the model should match with resource allocation param
    "decima_model_name": "model_40_rand_exec",
    "resource_allocation": "DNN",
    "splitting_rule": None #DTS, int, None
}

num_ex = 50
if __name__ == "__main__":
    cfg = load(filename=osp.join("config", "hyperheuristic_" + ENV_CFG["test_data"] + ".yaml"))
    list_model = ["model_100_DNN","model_100_DRA","model_100_hyper"]
    list_resource_allocation = ['DNN','DRA','HyperHeuristic']
    list_result = []
    for i in range(len(list_model)):
        model_name = list_model[i]
        resource_allocation = list_resource_allocation[i]
        print("-", model_name, ",", resource_allocation)
        count = 0
        result = []
        while count < num_ex:
            agent_cfg = cfg["agent"] | {
                "num_executors": ENV_CFG["num_executors"],
                "state_dict_path": osp.join("models", "hyperheuristic", ENV_CFG["train_data"], str(model_name)+".pt"),
                "num_heuristics": cfg["env"]["num_heuristics"],
                "list_heuristics": cfg["env"]["list_heuristics"],
                "resource_allocation": resource_allocation#cfg["env"]["resource_allocation"],
            }
            scheduler = make_scheduler(agent_cfg)
            env_cfg = dict(ENV_CFG)
            env_cfg["num_heuristics"] = cfg["env"]["num_heuristics"]
            env_cfg["list_heuristics"] = cfg["env"]["list_heuristics"]
            env_cfg["num_resource_heuristics"] = cfg["env"]["num_resource_heuristics"]
            env_cfg["plot_title"] = Path("./test_results/"+ENV_CFG["train_data"]+"/hyper/hyper_" +ENV_CFG["train_data"]+"_"+ENV_CFG["test_data"] +
                                         "_" + ENV_CFG["hyper_model_name"] + "_" + str(ENV_CFG["splitting_rule"])+".png")
            avg_job_duration = run_episode(env_cfg, scheduler,seed = count+10)
            result.append(avg_job_duration)
            count += 1
            print(f"  {count}) Average job duration: {avg_job_duration:.1f}s", flush=True)
        list_result.append(result)
    print("Result:")
    print(list_result)