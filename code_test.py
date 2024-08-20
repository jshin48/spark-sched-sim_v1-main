import os
import csv
from pathlib import Path
from pprint import pprint

import torch
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd
from param import *


args.input_file = './results/0806/ex_list_cross.csv'
args.result_folder = './results/0806/'
args.output_file = 'result_ex_list_cross.csv'

def main():
    with open(args.input_file) as f:
        df = pd.read_csv(f)
        # reader = csv.reader(f)
        # lines = list(reader)
        # parameters_set = []
        # par_name = lines[0:][0]
        # for line in lines[1:]:
        #     parameters_set.append(line)
    print(df.dtypes)
    df.drop(["id"], axis=1, inplace=True)
    for i in range(len(df)):
        param_upate(list(df),list(df.iloc[i]))
        pprint(vars(args))

if __name__ == "__main__":
    main()



# def softmax_with_temperature(logits, temperature=0.5):
#     scaled_logits = logits / temperature
#     max_logits = torch.max(scaled_logits)  # For numerical stability
#     exp_logits = torch.exp(scaled_logits - max_logits)
#     return exp_logits / torch.sum(exp_logits)
#
# def _sample(logits, temperature=0.5):
#     pi = softmax_with_temperature(logits, temperature).detach().numpy()
#     print("action probability:", [round(pi[0], 2), round(pi[1], 2)])
#     idx = random.choices(np.arange(pi.size), pi)[0]
#     lgprob = np.log(pi[idx])
#     return idx, lgprob
#
# # Example usage
# logits = torch.tensor([0.0071, -0.0127])
# idx, lgprob = _sample(logits, temperature=0.1)
# print(f"Selected action: {idx}, Log probability: {lgprob}")