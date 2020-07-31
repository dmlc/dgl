from time import time

import numpy as np

from utils import arg_list

from dgl.transform import metis_partition
from dgl import backend as F
import dgl

def get_partition_list(g, psize):
    p_gs = metis_partition(g, psize)
    graphs = []
    for k, val in p_gs.items():
        nids = val.ndata[dgl.NID]
        nids = F.asnumpy(nids)
        graphs.append(nids)
    return graphs

def get_subgraph(g, par_arr, i, psize, batch_size):
    par_batch_ind_arr = [par_arr[s] for s in range(
        i * batch_size, (i + 1) * batch_size) if s < psize]
    g1 = g.subgraph(np.concatenate(
        par_batch_ind_arr).reshape(-1).astype(np.int64))
    return g1
