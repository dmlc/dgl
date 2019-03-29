import torch

def cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x
