import torch.nn as nn


def GlorotOrthogonal(tensor, scale=2.0):
    if tensor is not None:
        nn.init.orthogonal_(tensor.data)
        scale /= (tensor.size(-2) + tensor.size(-1)) * tensor.var()
        tensor.data *= scale.sqrt()
