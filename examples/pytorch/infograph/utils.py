""" Credit: https://github.com/fanyun-sun/InfoGraph """

import math

import torch as th
import torch.nn.functional as F


def get_positive_expectation(p_samples, average=True):
    """Computes the positive part of a JS Divergence.
    Args:
        p_samples: Positive samples.
        average: Average the result over samples.
    Returns:
        th.Tensor
    """
    log_2 = math.log(2.0)
    Ep = log_2 - F.softplus(-p_samples)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, average=True):
    """Computes the negative part of a JS Divergence.
    Args:
        q_samples: Negative samples.
        average: Average the result over samples.
    Returns:
        th.Tensor
    """
    log_2 = math.log(2.0)
    Eq = F.softplus(-q_samples) + q_samples - log_2

    if average:
        return Eq.mean()
    else:
        return Eq


def local_global_loss_(l_enc, g_enc, graph_id):
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    device = g_enc.device

    pos_mask = th.zeros((num_nodes, num_graphs)).to(device)
    neg_mask = th.ones((num_nodes, num_graphs)).to(device)

    for nodeidx, graphidx in enumerate(graph_id):
        pos_mask[nodeidx][graphidx] = 1.0
        neg_mask[nodeidx][graphidx] = 0.0

    res = th.mm(l_enc, g_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))

    return E_neg - E_pos


def global_global_loss_(sup_enc, unsup_enc):
    num_graphs = sup_enc.shape[0]
    device = sup_enc.device

    pos_mask = th.eye(num_graphs).to(device)
    neg_mask = 1 - pos_mask

    res = th.mm(sup_enc, unsup_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, average=False)
    E_pos = (E_pos * pos_mask).sum() / pos_mask.sum()
    E_neg = get_negative_expectation(res * neg_mask, average=False)
    E_neg = (E_neg * neg_mask).sum() / neg_mask.sum()

    return E_neg - E_pos
