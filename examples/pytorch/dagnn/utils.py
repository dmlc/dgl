import random

import numpy as np
import torch
from torch.nn import functional as F


def evaluate(model, graph, feats, labels, idxs):
    model.eval()
    with torch.no_grad():
        logits = model(graph, feats)
        results = ()
        for idx in idxs:
            loss = F.cross_entropy(logits[idx], labels[idx])
            acc = torch.sum(
                logits[idx].argmax(dim=1) == labels[idx]
            ).item() / len(idx)
            results += (loss, acc)
    return results


def generate_random_seeds(seed, nums):
    random.seed(seed)
    return [random.randint(1, 999999999) for _ in range(nums)]


def set_random_state(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
