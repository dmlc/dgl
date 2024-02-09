import os

import torch
from partition_utils import *


class ClusterIter(object):
    """The partition sampler given a DGLGraph and partition number.
    The metis is used as the graph partition backend.
    """

    def __init__(self, dn, g, psize, batch_size):
        """Initialize the sampler.

        Paramters
        ---------
        dn : str
            The dataset name.
        g  : DGLGraph
            The full graph of dataset
        psize: int
            The partition number
        batch_size: int
            The number of partitions in one batch
        """
        self.psize = psize
        self.batch_size = batch_size
        # cache the partitions of known datasets&partition number
        if dn:
            fn = os.path.join("./datasets/", dn + "_{}.npy".format(psize))
            if os.path.exists(fn):
                self.par_li = np.load(fn, allow_pickle=True)
            else:
                os.makedirs("./datasets/", exist_ok=True)
                self.par_li = get_partition_list(g, psize)
                self.par_li = np.array(self.par_li, dtype=object)
                np.save(fn, self.par_li)
        else:
            self.par_li = get_partition_list(g, psize)
        par_list = []
        for p in self.par_li:
            par = torch.Tensor(p)
            par_list.append(par)
        self.par_list = par_list

    def __len__(self):
        return self.psize

    def __getitem__(self, idx):
        return self.par_li[idx]


def subgraph_collate_fn(g, batch):
    nids = np.concatenate(batch).reshape(-1).astype(np.int64)
    g1 = g.subgraph(nids)
    g1 = dgl.remove_self_loop(g1)
    g1 = dgl.add_self_loop(g1)
    return g1
