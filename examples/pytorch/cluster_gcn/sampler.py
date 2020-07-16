import random

import dgl.function as fn
import torch

from partition_utils import *


class ClusterIter(object):
    '''The partition sampler given a DGLGraph and partition number.
    The metis is used as the graph partition backend.
    '''
    def __init__(self, g, psize, batch_size, seed_nid, use_pp=True):
        """Initialize the sampler.

        Paramters
        ---------
        g  : DGLGraph
            The full graph of dataset
        psize: int
            The partition number
        batch_size: int
            The number of partitions in one batch
        seed_nid: np.ndarray
            The training nodes ids, used to extract the training graph
        use_pp: bool
            Whether to use precompute of AX
        """
        self.use_pp = use_pp
        self.g = g.subgraph(seed_nid)
        self.g.copy_from_parent()

        # precalc the aggregated features from training graph only
        if use_pp:
            self.precalc(self.g)
            print('precalculating')

        self.psize = psize
        self.batch_size = batch_size
        self.par_li = get_partition_list(self.g, psize)
        self.max = int((psize) // batch_size)
        random.shuffle(self.par_li)

    def precalc(self, g):
        norm = self.get_norm(g)
        g.ndata['norm'] = norm
        features = g.ndata['features']
        print("features shape, ", features.shape)
        with torch.no_grad():
            g.update_all(fn.copy_src(src='features', out='m'),
                         fn.sum(msg='m', out='features'),
                         None)
            pre_feats = g.ndata['features'] * norm
            # use graphsage embedding aggregation style
            g.ndata['features'] = torch.cat([features, pre_feats], dim=1)

    # use one side normalization
    def get_norm(self, g):
        norm = 1. / g.in_degrees().float().unsqueeze(1)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.g.ndata['features'].device)
        return norm

    def __len__(self):
        return self.max

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.max:
            result = self.par_li[self.n]
            self.n += 1
            return result
        else:
            random.shuffle(self.par_li)
            raise StopIteration
