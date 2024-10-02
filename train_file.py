from cfg_loader import load
from trainers import make_trainer
import time,csv

if __name__ == "__main__":
    cfg = load('config/hyperheuristic_tpch.yaml')

    with open("results/0929/train_list_hyper.csv") as f:
        reader = csv.reader(f)
        lines = list(reader)

    cat1 = lines[0]
    cat2 = lines[1]

    for line in lines[2:]:
        curr_time = time.time()
        line=[int(line[0]),str(line[1]),str(line[2]),int(line[3]),float(line[4]),
              int(line[5]),str(line[6]),str(line[7]),int(line[8]),str(line[9]),int(line[10])]
        for i in range(len(line)):
            cfg[cat1[i]][cat2[i]]=line[i]
        cfg['trainer']['artifacts_dir'] = "models/"+str(cfg['agent']['agent_cls'])\
                                          +"/"+str(cfg['env']['data_sampler_cls'])\
                                          +"/"+str(cfg['trainer']['artifacts_dir'])
        print(cfg)
        make_trainer(cfg).train()
        print("Training time:", time.time()-curr_time)






