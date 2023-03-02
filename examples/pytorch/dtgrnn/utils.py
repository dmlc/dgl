import dgl
import numpy as np
import scipy.sparse as sparse
import torch
import torch.nn as nn


class NormalizationLayer(nn.Module):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    # Here we shall expect mean and std be scaler
    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()


def get_learning_rate(optimizer):
    for param in optimizer.param_groups:
        return param["lr"]
