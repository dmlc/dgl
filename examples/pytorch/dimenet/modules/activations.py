import torch


def swish(x):
    """
    Swish activation function,
    from Ramachandran, Zopf, Le 2017. "Searching for Activation Functions"
    """
    return x * torch.sigmoid(x)
