import os
import csv
from pathlib import Path
from pprint import pprint

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import gymnasium as gym
import pathlib, sys
import pandas as pd

from cfg_loader import load
from spark_sched_sim.schedulers import (
    RoundRobinScheduler,
    NeuralScheduler,
    make_scheduler,
    HybridHeuristicScheduler,
    WscptScheduler,
    McScheduler,
    SjfScheduler,
    LjfScheduler,
    FifoScheduler
)
from spark_sched_sim.wrappers import NeuralActWrapper
from spark_sched_sim import metrics
from param import *

args.input_file = './results/2025/0707/ex_list.csv'
args.result_folder = './results/2025/0707/'
args.output_file = 'result1.csv'
args.moving_delay = 1000
args.warmup_delay = 1000
args.pod_creation_time = 100

def main():
    with open(args.input_file) as f:
        df = pd.read_csv(f)

    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)

    f = open(args.result_folder + args.output_file, 'w', encoding='UTF8', newline='')
    writer = csv.writer(f)
    writer.writerow(list(df)+["avg_job_duration"])
    df['num_executors'] = df['num_executors'].astype(int)
    df['num_heuristics'] = df['num_heuristics'].astype(int)
    #print(df.dtypes)

    for i in range(len(df)):
        print(f"Running example scheduler: {df['scheduler_name'].iloc[i]}")
        result_set = []
        param_update(list(df),list(df.iloc[i]))
        #pprint(vars(args))
        for ex_num in range(int(args.num_experiments)):
            if df["scheduler_name"].iloc[i] == "DecimaScheduler":
                agent_cfg_file = "decima_tpch.yaml"
            else:
                agent_cfg_file = "hyperheuristic_tpch.yaml"
            result = example(i,ex_num,agent_cfg_file)
            result_set.append(result)
        writer.writerow(list(df.iloc[i])+[result_set])

def example(ex_id,ex_num,cfg_file):
    CFG = load(filename=os.path.join("config", cfg_file))
    # Update agent configuration
    agent_cfg = CFG["agent"] | {
        "num_executors": args.num_executors,
        "num_heuristics": args.num_heuristics,
        "list_heuristics": args.list_heuristics,
        "resource_allocation": args.resource_allocation,
        "num_resource_heuristics": args.num_resource_heuristics,
        "list_resource_heuristics": args.list_resource_heuristics,
        "input_feature": args.input_feature,
    }
    env_cfg = vars(args)
    env_cfg["plot_title"] = Path(args.result_folder+str(ex_id)+"_"+str(ex_num)+".png")
    #agent_cfg["agent_cls"] =args.scheduler_name
    #env_cfg["agent_cls"] = agent_cfg["agent_cls"]
    # if agent_cfg["agent_cls"] == "HybridHeuristicScheduler":
    #     scheduler = HybridHeuristicScheduler(env_cfg["num_executors"],agent_cfg["resource_allocation"],rule_switch_threshold=4)
    # elif agent_cfg["agent_cls"] == "RoundRobinScheduler":
    #     scheduler = RoundRobinScheduler(env_cfg["num_executors"], env_cfg["resource_allocation"])
    # elif agent_cfg["agent_cls"] == "SjfScheduler":
    #     scheduler = SjfScheduler(env_cfg["num_executors"], env_cfg["resource_allocation"])
    # elif agent_cfg["agent_cls"] == "LjfScheduler":
    #     scheduler = LjfScheduler(env_cfg["num_executors"], env_cfg["resource_allocation"])
    # elif agent_cfg["agent_cls"] == "FifoScheduler":
    #     scheduler = FifoScheduler(env_cfg["num_executors"], env_cfg["resource_allocation"])
    # elif agent_cfg["agent_cls"] == "WscptScheduler":
    #     scheduler = WscptScheduler(env_cfg["num_executors"], env_cfg["resource_allocation"])
    # elif agent_cfg["agent_cls"] == "McScheduler":
    #     scheduler = McScheduler(env_cfg["num_executors"], env_cfg["resource_allocation"])
    # else:
    #     scheduler = make_scheduler(agent_cfg)

    # Define a mapping of agent_cls to scheduler classes or lambda functions

    scheduler_name_mapping = {
        "Decima": "DecimaScheduler",
        "Hyper": "HyperHeuristicScheduler",
    }
    agent_cfg["agent_cls"] = scheduler_name_mapping.get(args.scheduler_name, args.scheduler_name)

    env_cfg["agent_cls"] = agent_cfg["agent_cls"]

    scheduler_mapping = {
        "Hybrid": lambda: HybridHeuristicScheduler(env_cfg["num_executors"],
                                                                     agent_cfg["resource_allocation"],
                                                                     rule_switch_threshold=4),
        "RoundRobin": lambda: RoundRobinScheduler(env_cfg["num_executors"], env_cfg["resource_allocation"]),
        "Sjf": lambda: SjfScheduler(env_cfg["num_executors"], env_cfg["resource_allocation"]),
        "Ljf": lambda: LjfScheduler(env_cfg["num_executors"], env_cfg["resource_allocation"]),
        "Fifo": lambda: FifoScheduler(env_cfg["num_executors"], env_cfg["resource_allocation"]),
        "Wscpt": lambda: WscptScheduler(env_cfg["num_executors"], env_cfg["resource_allocation"]),
        "Mc": lambda: McScheduler(env_cfg["num_executors"], env_cfg["resource_allocation"]),
    }

    # Use the mapping to create the scheduler or fallback to make_scheduler
    scheduler = scheduler_mapping.get(agent_cfg["agent_cls"], lambda: make_scheduler(agent_cfg))()

    if isinstance(scheduler, NeuralScheduler):
        agent_cfg["state_dict_path"] = Path("models/" + args.scheduler_name + "/" + args.train_data +
                                            "/" + args.model_name + "/checkpoints/" + str(args.model_num_train) + "/model.pt")

    avg_job_duration = run_episode(env_cfg, agent_cfg, scheduler, seed = 42+ex_num)

    print(f"Done! Average job duration: {avg_job_duration:.1f}s", flush=True)
    return avg_job_duration

def run_episode(env_cfg,  agent_cfg, scheduler, seed=1234):
    env_cfg["data_sampler_cls"]= env_cfg["test_data"]
    env = gym.make("spark_sched_sim:SparkSchedSimEnv-v0", env_cfg=env_cfg,  agent_cfg= agent_cfg)

    env = NeuralActWrapper(env)
    scales = {
        "num_tasks": env_cfg["num_tasks_scale"],
        "work": env_cfg["work_scale"],
        "cpt": env_cfg["cpt_scale"],
        "num_nodes": env_cfg["num_node_scale"]
    }
    env = scheduler.obs_wrapper_cls(env, scales)
    if isinstance(scheduler, NeuralScheduler):
        scheduler.actor.eval()  # set to evaluation mode

    obs, _ = env.reset(seed=seed, options=None)
    terminated = truncated = False

    while not (terminated or truncated):
        if isinstance(scheduler, NeuralScheduler):
            action, *_ = scheduler(obs)
        else:
            action = scheduler(obs)
        obs, _, terminated, truncated, _ = env.step(action)

    avg_job_duration = metrics.avg_job_duration(env) * 1e-3
    # metrics.print_task_job_time(env)
    # cleanup rendering
    env.close()

    return avg_job_duration

if __name__ == "__main__":
    main()
