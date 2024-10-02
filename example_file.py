import os
import csv
from pathlib import Path
from pprint import pprint

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import gymnasium as gym
import pathlib, sys

from cfg_loader import load
from spark_sched_sim.schedulers import (
    NeuralScheduler,
    make_scheduler,
    HybridHeuristicScheduler,
)
from spark_sched_sim.wrappers import NeuralActWrapper
from spark_sched_sim import metrics
from param import *

args.input_file = './results/0929/ex_list_from_0802_with_new_models.csv'
args.result_folder = './results/0929/'
args.output_file = 'result_list_from_0802_with_new_models.csv'
CFG = load(filename=os.path.join("config", "decima_tpch.yaml"))

def main():
    with open(args.input_file) as f:
        reader = csv.reader(f)
        lines = list(reader)
        parameters_set = []
        par_name = lines[0:][0]
        for line in lines[1:]:
            parameters_set.append(line)

    #print(parameters_set)

    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)

    f = open(args.result_folder + args.output_file, 'w', encoding='UTF8', newline='')
    writer = csv.writer(f)
    writer.writerow(par_name+["avg_job_duration"])


    for ex_id, par_value in enumerate(parameters_set):
        result_set = []
        param_upate(par_name,par_value)
        pprint(vars(args))
        print("parameter:",par_value)
        for ex_num in range(args.num_experiments):
            result = example(ex_id,ex_num)
            result_set.append(result)

        writer.writerow(par_value+result_set)

def example(ex_id,ex_num):
    agent_cfg = CFG["agent"] | {"num_executors": args.num_executors,
        "state_dict_path": Path("models/"+args.scheduler_name+"/" +args.train_data+
                                "/"+ args.model_name +"/checkpoints/"+ str(args.model_num_train)+"/model.pt"),
        "num_heuristics": args.num_heuristics,
        "list_heuristics": args.list_heuristics,
        "resource_allocation": args.resource_allocation,
        "num_resource_heuristics": args.num_resource_heuristics,
        "list_resource_heuristics": args.list_resource_heuristics
    }
    env_cfg = vars(args)
    env_cfg["plot_title"] = Path(args.result_folder+str(ex_id)+"_"+str(ex_num)+".png")
    agent_cfg["agent_cls"] = args.scheduler_name
    env_cfg["agent_cls"] = agent_cfg["agent_cls"]

    #pprint(env_cfg)
    if agent_cfg["agent_cls"] == "HybridHeuristicScheduler":
        scheduler = HybridHeuristicScheduler(env_cfg["num_executors"],agent_cfg["resource_allocation"],rule_switch_threshold=4)
    else:
        scheduler = make_scheduler(agent_cfg)
    avg_job_duration = run_episode(env_cfg, scheduler, seed = 42 + ex_num)

    print(f"Done! Average job duration: {avg_job_duration:.1f}s", flush=True)
    return avg_job_duration

def run_episode(env_cfg, scheduler, seed=1234):
    env_cfg["data_sampler_cls"]= env_cfg["test_data"]
    env = gym.make("spark_sched_sim:SparkSchedSimEnv-v0", env_cfg=env_cfg)

    if isinstance(scheduler, NeuralScheduler) or isinstance(scheduler, HybridHeuristicScheduler):
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
    metrics.print_task_job_time(env)
    # cleanup rendering
    env.close()

    return avg_job_duration


if __name__ == "__main__":
    main()
