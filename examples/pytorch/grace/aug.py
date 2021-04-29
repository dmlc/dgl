# Data augmentation on graphs via edge dropping and feature masking

import torch as th
import numpy as np
import dgl

def aug(graph, x, feat_drop_rate, edge_mask_rate):
    ng = drop_edge(graph, edge_mask_rate)
    feat = drop_feat(x, feat_drop_rate)
    ng = ng.add_self_loop()

    return ng, feat

def drop_edge(graph, drop_prob):
    E = graph.num_edges()

    mask_rates = th.FloatTensor(np.ones(E) * drop_prob)
    masks = th.bernoulli(1 - mask_rates)
    edge_idx = masks.nonzero().squeeze(1)

    sg = dgl.edge_subgraph(graph, edge_idx, preserve_nodes=True)
    
    return sg

def drop_feat(x, drop_prob):
    D = x.shape[1]
    mask_rates = th.FloatTensor(np.ones(D) * drop_prob)
    masks = th.bernoulli(1 - mask_rates)

    x = x.clone()
    x[:, masks] = 0

    return x