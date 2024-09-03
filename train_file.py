import argparse
import csv
import time

import pandas as pd

from cfg_loader import load
from trainers import make_trainer


#usage : python3 train_file.py config/hyperheuristic_alibaba.yaml results/0822/train_list_feature.csv
#usage : python3 train_file.py config/hyperheuristic_tpch.yaml results/0822/train_list_feature.csv
#usage : python3 train_file.py config/decima_tpch.yaml results/0822/train_decima_tpch.csv

def load_csv(csv_path):
    with open(csv_path) as f:
        reader = csv.reader(f)
        lines = list(reader)
    return lines

def load_dataframe(csv_path):
    with open(csv_path) as f:
        df = pd.read_csv(f, skiprows=2, header=None)
    return df


def train_model(cfg, lines, df):
    cat1 = lines[0]
    cat2 = lines[1]

    for i in range(len(df)):
        curr_time = time.time()
        for j in range(len(cat1)):
            cfg[cat1[j]][cat2[j]] = df.iloc[i][j]
        cfg['trainer']['artifacts_dir'] = "models/" + str(cfg['agent']['agent_cls']) \
                                          + "/" + str(cfg['env']['data_sampler_cls']) \
                                          + "/" + str(cfg['trainer']['artifacts_dir'])
        print(cfg)
        make_trainer(cfg).train()
        print("Training time:", time.time() - curr_time)


def main():
    parser = argparse.ArgumentParser(description='Process some file paths.')
    parser.add_argument('--config_path', type=str, default='config/decima_tpch.yaml')
    parser.add_argument('--csv_path', type=str, default='results/0822/train_decima_tpch.csv')

    # Parse arguments
    args = parser.parse_args()

    # Load configuration
    cfg = load(args.config_path)

    # Load CSV file
    lines = load_csv(args.csv_path)
    df = load_dataframe(args.csv_path)

    # Train model
    train_model(cfg, lines, df)


if __name__ == "__main__":
    main()
