import dgl
from dgl.utils import toindex
import mxnet as mx
import numpy as np

def l0_sample(g, positive_max=128, negative_ratio=3):
    n_eids = g.number_of_edges()
    pos_eids = np.where(g.edata['rel_class'].asnumpy() > 0)[0]
    neg_eids = np.where(g.edata['rel_class'].asnumpy() == 0)[0]
    assert len(pos_eids) > 0

    pos_pool = []
    for eid in pos_eids:
        count_pool = [eid for j in range(int(g.edata['rel_count'][eid].asscalar()))]
        pos_pool += count_pool
    np.random.shuffle(pos_pool)

    positive_num = min(len(pos_pool), positive_max)
    negative_num = min(len(neg_eids), positive_num * negative_ratio)
    pos_sample = pos_pool[0:positive_num]
    # pos_sample = np.random.choice(pos_eids, positive_num, replace=False)
    neg_sample = np.random.choice(neg_eids, negative_num, replace=False)
    weights = np.zeros(n_eids)
    # weights[pos_sample] = 1
    np.add.at(weights, pos_sample, 1)
    weights[neg_sample] = 1
    eids = np.where(weights > 0)[0]
    sub_g = g.edge_subgraph(toindex(eids.tolist()))
    sub_g.copy_from_parent()
    sub_g.edata['sample_weights'] = mx.nd.array(weights[eids],
                                                ctx=g.edata['rel_class'].context)
    return sub_g