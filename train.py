from cfg_loader import load
from trainers import make_trainer
import time

if __name__ == "__main__":
    curr_time = time.time()
    cfg = load('config/decima_tpch.yaml')
    make_trainer(cfg).train()
    print("Training time:", time.time()-curr_time)
