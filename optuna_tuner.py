import optuna
import copy
import argparse
import csv
import time
import pandas as pd
from cfg_loader import load
from trainers import make_trainer
import warnings
import warnings
warnings.filterwarnings("ignore")


TUNING_TYPE = 'emb_hidden_dim'
def load_csv(csv_path):
    with open(csv_path) as f:
        reader = csv.reader(f)
        lines = list(reader)
    return lines

def load_dataframe(csv_path,column_names, dtype_dict):
    with open(csv_path) as f:
        df = pd.read_csv(f, skiprows=3, header=None, names=column_names, dtype=dtype_dict)
        #print(df)
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

        cfg['trainer']['artifacts_dir'] = ("models/" + str(cfg['agent']['agent_cls']) \
                                          + "/" + str(cfg['env']['data_sampler_cls']) \
                                          + "/" + TUNING_TYPE
                                          + str(cfg['trainer']['artifacts_dir'])\
                                          + str(cfg['agent']["embed_dim"]) +"_" \
                                          + str(cfg['agent']["gnn_mlp_kwargs"]["hid_dims"][0]))
        #print(cfg)



def update_cfg(cfg, lines, df):
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

        cfg['trainer']['artifacts_dir'] = ("models/" + str(cfg['agent']['agent_cls']) \
                                          + "/" + str(cfg['env']['data_sampler_cls']) \
                                          + "/" + TUNING_TYPE
                                          + str(cfg['trainer']['artifacts_dir'])\
                                          + str(cfg['agent']["embed_dim"]) +"_" \
                                          + str(cfg['agent']["gnn_mlp_kwargs"]["hid_dims"][0]))
        return cfg

def objective(trial):
    parser = argparse.ArgumentParser(description='Process some file paths.')
    parser.add_argument('--config_path', type=str, default='config/hyperheuristic_tpch.yaml')
    parser.add_argument('--csv_path', type=str, default='results/2025/0407/tuning_dim_Basis.csv')

    # Parse arguments
    args = parser.parse_args()

    # Load configuration
    cfg = load(args.config_path)

    # Load CSV file
    lines = load_csv(args.csv_path)

    column_names = [
        "num_iterations", "agent_cls", "input_feature", "num_heuristics", "resource_allocation",
        "checkpointing_freq", "artifacts_dir", "num_executors", "cpt_scale", "num_node_scale",
        "data_sampler_cls", "job_arrival_rate", "job_arrival_cap", "opt_kwargs"
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
        "cpt_scale": "float64",
        "num_node_scale": "float64",
        "data_sampler_cls": "string",
        "job_arrival_rate": "float64",
        "job_arrival_cap": "int64",
        "opt_kwargs": "float64",
    }

    df = load_dataframe(args.csv_path, column_names, dtype_dict)
    cfg = update_cfg(cfg, lines, df)

    # Inject hyperparameters from Optuna
    cfg["agent"]["embed_dim"] = trial.suggest_categorical("embed_dim", [8, 16, 32, 64])
    hid_dims = trial.suggest_categorical("gnn_hid_dims", [(64,), (64, 64), (128, 64)])
    cfg["agent"]["gnn_mlp_kwargs"]["hid_dims"] = list(hid_dims)

    # Shorter training for tuning
    cfg["trainer"]["num_iterations"] = 15  # or lower
    cfg["trainer"]["num_sequences"] = 1
    cfg["trainer"]["num_rollouts"] = 1
    cfg["trainer"]["use_tensorboard"] = False

    # Train model
    trainer = make_trainer(cfg)
    trainer.train()
    print("Train with", cfg["agent"]["embed_dim"], "embedding dimension and", cfg["agent"]["gnn_mlp_kwargs"]["hid_dims"], "hidden dimensions",
          ",obj:",-trainer.best_avg_job_duration)

    return -trainer.best_avg_job_duration  # If lower job duration is better


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name="hyperheuristic_tuning",
        storage="sqlite:///hh_study.db",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=30, n_jobs=1)  # Parallel tuning
