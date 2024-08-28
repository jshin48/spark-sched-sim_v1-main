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
    NeuralScheduler,
    make_scheduler,
    HybridHeuristicScheduler,
)
from spark_sched_sim.wrappers import NeuralActWrapper
from spark_sched_sim import metrics
from param import *

args.input_file = './results/0822/ex_list.csv'
args.result_folder = './results/0822/'
args.output_file = 'result_hyper_alibaba.csv'
CFG = load(filename=os.path.join("config", "hyperheuristic_alibaba.yaml"))

def main():
    # with open(args.input_file) as f:
    #     reader = csv.reader(f)
    #     lines = list(reader)
    #     parameters_set = []
    #     par_name = lines[0:][0]
    #     for line in lines[1:]:
    #         parameters_set.append(line)
    with open(args.input_file) as f:
        df = pd.read_csv(f)

    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)

    f = open(args.result_folder + args.output_file, 'w', encoding='UTF8', newline='')
    writer = csv.writer(f)
    writer.writerow(list(df)+["avg_job_duration"])

    # for ex_id, par_value in enumerate(parameters_set):
    #     result_set = []
    #     param_upate(par_name,par_value)
    #     pprint(vars(args))
    #     print("parameter:",par_value)
    #     for ex_num in range(args.num_experiments):
    #         result = example(ex_id,ex_num)
    #         result_set.append(result)
    #
    #     writer.writerow(par_value+result_set)

    #df.drop(["id"], axis=1, inplace=True)
    #df=df.astype({'splitting_rule': 'object'})
    print(df.dtypes)

    for i in range(len(df)):
        result_set = []
        param_upate(list(df),list(df.iloc[i]))
        pprint(vars(args))
        for ex_num in range(args.num_experiments):
            result = example(i,ex_num)
            result_set.append(result)


def example(ex_id,ex_num):
    agent_cfg = CFG["agent"] | {"num_executors": args.num_executors,
        "state_dict_path": Path("models/"+args.scheduler_name+"/" +args.train_data+
                                "/"+ args.model_name +"/checkpoints/"+ str(args.model_num_train)+"/model.pt")}
    agent_cfg.update({ "num_heuristics": args.num_heuristics,
                       "list_heuristics": args.list_heuristics,
                       "resource_allocation": args.resource_allocation,
                       "num_resource_heuristics": args.num_resource_heuristics,
                       "list_resource_heuristics": args.list_resource_heuristics,
                       "input_feature": args.input_feature
                       })
    env_cfg = vars(args)
    env_cfg["plot_title"] = Path(args.result_folder+str(ex_id)+"_"+str(ex_num)+".png")
    agent_cfg["agent_cls"] = args.scheduler_name
    env_cfg["agent_cls"] = agent_cfg["agent_cls"]

    #pprint(env_cfg)
    if agent_cfg["agent_cls"] == "HybridHeuristicScheduler":
        scheduler = HybridHeuristicScheduler(env_cfg["num_executors"],agent_cfg["resource_allocation"],rule_switch_threshold=4)
    else:
        scheduler = make_scheduler(agent_cfg)
    avg_job_duration = run_episode(env_cfg, agent_cfg, scheduler, seed = 42+ ex_num)

    print(f"Done! Average job duration: {avg_job_duration:.1f}s", flush=True)
    return avg_job_duration

def run_episode(env_cfg,  agent_cfg, scheduler, seed=1234):
    env_cfg["data_sampler_cls"]= env_cfg["test_data"]
    env = gym.make("spark_sched_sim:SparkSchedSimEnv-v0", env_cfg=env_cfg,  agent_cfg= agent_cfg)

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
    #metrics.print_task_job_time(env)
    # cleanup rendering
    env.close()

    return avg_job_duration


if __name__ == "__main__":
    main()
