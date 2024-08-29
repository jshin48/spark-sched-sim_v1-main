from cfg_loader import load
from trainers import make_trainer
import time
import matplotlib.pyplot as plt

lrs = []
losses = []

if __name__ == "__main__":
    curr_time = time.time()
    cfg = load('config/decima_tpch.yaml')

    for i in range(10):
        cfg['trainer']['opt_kwargs']['lr'] = 1e-5 * (10 ** (i))
        print(cfg)
        Trainer = make_trainer(cfg)
        Trainer.train()
        #print("Training time:", time.time()-curr_time)

        lrs.append(cfg['trainer']['opt_kwargs']['lr'])
        print(i,"th iteration Trainer.loss:",Trainer.loss)
        losses.append(Trainer.loss)

    # Plot the learning rate range test
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.show()


