import math
import os
import pickle

import numpy as np
import numpy.random as npr
import scipy as sp
import networkx as nx
from torch.utils.data import Dataset

from .. import backend as F
from ..batch import batch
from ..graph import DGLGraph
from ..utils import Index

def sbm(n_blocks, block_size, p, q, rng=None):
    """ (Symmetric) Stochastic Block Model

    Parameters
    ----------
    n_blocks : number of blocks
    block_size : block size
    p : probability for intra-community edge
    q : probability for inter-community edge
    """
    n = n_blocks * block_size
    p /= n
    q /= n
    rng = np.random.RandomState() if rng is None else rng

    rows = []
    cols = []
    for i in range(n_blocks):
        for j in range(i, n_blocks):
            density = p if i == j else q
            block = sp.sparse.random(block_size, block_size, density,
                                     random_state=rng, data_rvs=lambda n: np.ones(n))
            rows.append(block.row + i * block_size)
            cols.append(block.col + j * block_size)

    rows = np.hstack(rows)
    cols = np.hstack(cols)
    a = sp.sparse.coo_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(n, n))
    adj = sp.sparse.triu(a) + sp.sparse.triu(a, 1).transpose()
    return adj

class SBMMixture(Dataset):
    def __init__(self, n_graphs, n_nodes, n_communities,
                 k=2, avg_deg=3, p='Appendix C', rng=None):
        """ Symmetric Stochastic Block Model Mixture
        n_graphs : number of graphs
        n_nodes : number of nodes
        n_communities : number of communities
        k : multiplier, optional
        avg_deg : average degree, optional
        p : random density generator, optional
        rng : random number generator, optional
        """
        super(SBMMixture, self).__init__()
        self._n_nodes = n_nodes
        assert n_nodes % n_communities == 0
        block_size = n_nodes // n_communities
        if type(p) is str:
            p = {'Appendix C' : self._appendix_c}[p]
        self._k = k
        self._avg_deg = avg_deg
        self._gs = [DGLGraph() for i in range(n_graphs)]
        adjs = [sbm(n_communities, block_size, *p()) for i in range(n_graphs)]
        for g, adj in zip(self._gs, adjs):
            g.from_scipy_sparse_matrix(adj)
        self._lgs = [g.line_graph() for g in self._gs]
        in_degrees = lambda g: g.in_degrees(Index(F.arange(g.number_of_nodes(),
                        dtype=F.int64))).unsqueeze(1).float()
        self._g_degs = [in_degrees(g) for g in self._gs]
        self._lg_degs = [in_degrees(lg) for lg in self._lgs]
        self._eid2nids = list(zip(*[g.edges(sorted=True) for g in self._gs]))[0]

    def __len__(self):
        return len(self._gs)

    def __getitem__(self, idx):
        return self._gs[idx], self._lgs[idx], \
                self._g_degs[idx], self._lg_degs[idx], self._eid2nids[idx]

    def _appendix_c(self):
        q = npr.uniform(0, self._avg_deg - math.sqrt(self._avg_deg))
        p = self._k * self._avg_deg - q
        return p, q

    def collate_fn(self, x):
        g, lg, deg_g, deg_lg, eid2nid = zip(*x)
        g_batch = batch(g)
        lg_batch = batch(lg)
        degg_batch = F.pack(deg_g)
        deglg_batch = F.pack(deg_lg)
        eid2nid_batch = F.pack([x + i * self._n_nodes for i, x in enumerate(eid2nid)])
        return g_batch, lg_batch, degg_batch, deglg_batch, eid2nid_batch
