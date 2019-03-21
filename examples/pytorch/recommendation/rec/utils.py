import torch

def cuda(x):
    if torch.cuda.is_available():
        return x.cuda() if not x.is_cuda else x
    else:
        return x
