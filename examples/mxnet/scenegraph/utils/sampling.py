import dgl
import mxnet as mx
import numpy as np
from dgl.utils import toindex


def l0_sample(g, positive_max=128, negative_ratio=3):
    """sampling positive and negative edges"""
    if g is None:
        return None
    n_eids = g.number_of_edges()
    pos_eids = np.where(g.edata["rel_class"].asnumpy() > 0)[0]
    neg_eids = np.where(g.edata["rel_class"].asnumpy() == 0)[0]
    if len(pos_eids) == 0:
        return None

    positive_num = min(len(pos_eids), positive_max)
    negative_num = min(len(neg_eids), positive_num * negative_ratio)
    pos_sample = np.random.choice(pos_eids, positive_num, replace=False)
    neg_sample = np.random.choice(neg_eids, negative_num, replace=False)
    weights = np.zeros(n_eids)
    # np.add.at(weights, pos_sample, 1)
    weights[pos_sample] = 1
    weights[neg_sample] = 1
    # g.edata['sample_weights'] = mx.nd.array(weights, ctx=g.edata['rel_class'].context)
    # return g
    eids = np.where(weights > 0)[0]
    sub_g = g.edge_subgraph(toindex(eids.tolist()))
    sub_g.copy_from_parent()
    sub_g.edata["sample_weights"] = mx.nd.array(
        weights[eids], ctx=g.edata["rel_class"].context
    )
    return sub_g
