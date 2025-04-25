import random
import numpy as np
import torch

def worker_seeds(rank=0, seed=42):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
