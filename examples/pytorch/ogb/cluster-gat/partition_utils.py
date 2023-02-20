from time import time

import dgl

import numpy as np
from dgl import backend as F
from dgl.transforms import metis_partition


def get_partition_list(g, psize):
    p_gs = metis_partition(g, psize)
    graphs = []
    for k, val in p_gs.items():
        nids = val.ndata[dgl.NID]
        nids = F.asnumpy(nids)
        graphs.append(nids)
    return graphs
