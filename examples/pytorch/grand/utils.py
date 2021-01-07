import torch as th
import torch.nn as nn
import torch.nn.functional as F

def consis_loss(logps, temp, lam):
    ps = [th.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p/len(ps)

    sharp_p = (th.pow(avg_p, 1./temp) / th.sum(th.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += th.mean((p-sharp_p).pow(2).sum(1))
    loss = loss/len(ps)
    return lam * loss