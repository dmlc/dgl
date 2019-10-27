from time import time

import metis
import numpy as np

from utils import arg_list

def get_partition_list(g, psize):
    tmp_time = time()
    ng = g.to_networkx()
    print("getting adj using time{:.4f}".format(time() - tmp_time))
    print("run metis with partition size {}".format(psize))
    _, nd_group = metis.part_graph(ng, psize)
    print("metis finished in {} seconds.".format(time() - tmp_time))
    print("train group {}".format(len(nd_group)))
    al = arg_list(nd_group)
    return al

def get_subgraph(g, par_arr, i, psize, batch_size):
    par_batch_ind_arr = [par_arr[s] for s in range(
        i * batch_size, (i + 1) * batch_size) if s < psize]
    g1 = g.subgraph(np.concatenate(
        par_batch_ind_arr).reshape(-1).astype(np.int64))
    return g1
