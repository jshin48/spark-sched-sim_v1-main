import torch
import torch.nn.functional as F
import numpy as np
import random

def softmax_with_temperature(logits, temperature=0.5):
    scaled_logits = logits / temperature
    max_logits = torch.max(scaled_logits)  # For numerical stability
    exp_logits = torch.exp(scaled_logits - max_logits)
    return exp_logits / torch.sum(exp_logits)

def _sample(logits, temperature=0.5):
    pi = softmax_with_temperature(logits, temperature).detach().numpy()
    print("action probability:", [round(pi[0], 2), round(pi[1], 2)])
    idx = random.choices(np.arange(pi.size), pi)[0]
    lgprob = np.log(pi[idx])
    return idx, lgprob

# Example usage
logits = torch.tensor([0.0071, -0.0127])
idx, lgprob = _sample(logits, temperature=0.1)
print(f"Selected action: {idx}, Log probability: {lgprob}")