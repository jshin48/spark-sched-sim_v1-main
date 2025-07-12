import argparse
import csv
import time

import pandas as pd

from cfg_loader import load
from trainers import make_trainer

#usage : /usr/bin/python3.10 train_file.py --config_path config/hyperheuristic_tpch.yaml --csv_path results/Nov24/train_list.csv
#usage : /usr/bin/python3.10 train_file.py --config_path config/decima_tpch.yaml --csv_path results/0822/train_decima_tpch.csv

def load_csv(csv_path):
    with open(csv_path) as f:
        reader = csv.reader(f)
        lines = list(reader)
    return lines

def load_dataframe(csv_path,column_names, dtype_dict):
    with open(csv_path) as f:
        df = pd.read_csv(f, skiprows=3, header=None, names=column_names, dtype=dtype_dict)
        print(df)
    return df


def train_model(cfg, lines, df):
    cat1 = lines[0]
    cat2 = lines[1]
    cat3 = lines[2]

    for i in range(len(df)):
        curr_time = time.time()
        for j in range(len(cat1)):
            if cat3[j] == '':
                cfg[cat1[j]][cat2[j]] = df.iloc[i][j]
            else:
                cfg[cat1[j]][cat2[j]][cat3[j]] = df.iloc[i][j]

        cfg['trainer']['artifacts_dir'] = "models/" + str(cfg['agent']['agent_cls']) \
                                          + "/" + str(cfg['env']['data_sampler_cls']) \
                                          + "/" + str(cfg['trainer']['artifacts_dir'])
        print(cfg)
        make_trainer(cfg).train()
        print("Training time:", time.time() - curr_time)


def main():
    parser = argparse.ArgumentParser(description='Process some file paths.')
    parser.add_argument('--config_path', type=str, default='config/hyperheuristic_alibaba.yaml')
    parser.add_argument('--csv_path', type=str, default='results/2025/0707/train_list.csv')

    # Parse arguments
    args = parser.parse_args()

    # Load configuration
    cfg = load(args.config_path)

    # Load CSV file
    lines = load_csv(args.csv_path)

    column_names = [
        "num_iterations", "agent_cls", "input_feature", "num_heuristics", "resource_allocation",
        "checkpointing_freq", "artifacts_dir", "num_executors","data_sampler_cls", "job_arrival_rate",
        "job_arrival_cap", "opt_kwargs","cpt_scale", "num_node_scale", "num_tasks_scale","work_scale"
    ]

    # Define correct data types for each column
    dtype_dict = {
        "num_iterations": "int64",
        "agent_cls": "string",
        "input_feature" : "object",
        "num_heuristics": "int64",
        "resource_allocation": "string",
        "checkpointing_freq": "int64",
        "artifacts_dir": "string",
        "num_executors": "int64",
        "data_sampler_cls": "string",
        "job_arrival_rate": "float64",
        "job_arrival_cap": "int64",
        "opt_kwargs": "float64",
        "cpt_scale": "float64",
        "num_node_scale": "float64",
        "num_tasks_scale": "float64",
        "work_scale": "float64",
    }

    df = load_dataframe(args.csv_path, column_names, dtype_dict)

    # Train model
    train_model(cfg, lines, df)


if __name__ == "__main__":
    main()
