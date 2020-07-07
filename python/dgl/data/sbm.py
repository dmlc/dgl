"""Dataset for stochastic block model."""
import math
import random

import numpy as np
import numpy.random as npr
import scipy as sp

from .utils import deprecate_class
from ..graph import DGLGraph, batch
from ..utils import Index


def sbm(n_blocks, block_size, p, q, rng=None):
    """ (Symmetric) Stochastic Block Model

    Parameters
    ----------
    n_blocks : int
        Number of blocks.
    block_size : int
        Block size.
    p : float
        Probability for intra-community edge.
    q : float
        Probability for inter-community edge.
    rng : numpy.random.RandomState, optional
        Random number generator.

    Returns
    -------
    scipy sparse matrix
        The adjacency matrix of generated graph.
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


class SBMMixtureDataset(object):
    r""" Symmetric Stochastic Block Model Mixture

    Reference: Appendix C of "Supervised Community Detection with Hierarchical
               Graph Neural Networks" (https://arxiv.org/abs/1705.08415).

    Parameters
    ----------
    n_graphs : int
        Number of graphs.
    n_nodes : int
        Number of nodes.
    n_communities : int
        Number of communities.
    k : int, optional
        Multiplier.
    avg_deg : int, optional
        Average degree.
    pq : list of pair of nonnegative float or str, optional
        Random densities.
    rng : numpy.random.RandomState, optional
        Random number generator.

    Returns
    -------
    SBMMixtureDataset object

    Examples
    --------
    >>> data = SBMMixtureDataset(n_graphs=16, n_nodes=10000, n_communities=2)
    >>> from torch.utils.data import DataLoader
    >>> dataloader = DataLoader(data, batch_size=1, collate_fn=data.collate_fn)
    >>> for graph, line_graph, graph_degrees, line_graph_degrees, pm_pd in dataloader:
    ...     # your code here
    """
    def __init__(self,
                 n_graphs,
                 n_nodes,
                 n_communities,
                 k=2,
                 avg_deg=3,
                 pq='Appendix_C',
                 rng=None):
        self._n_nodes = n_nodes
        assert n_nodes % n_communities == 0
        block_size = n_nodes // n_communities
        self._k = k
        self._avg_deg = avg_deg
        self._graphs = [DGLGraph() for _ in range(n_graphs)]
        if type(pq) is list:
            assert len(pq) == n_graphs
        elif type(pq) is str:
            generator = {'Appendix_C': self._appendix_c}[pq]
            pq = [generator() for _ in range(n_graphs)]
        else:
            raise RuntimeError()
        adjs = [sbm(n_communities, block_size, *x, rng=rng) for x in pq]
        for g, adj in zip(self._graphs, adjs):
            g.from_scipy_sparse_matrix(adj)
        self._line_graphs = [g.line_graph(backtracking=False) for g in self._graphs]
        in_degrees = lambda g: g.in_degrees(
            Index(np.arange(0, g.number_of_nodes()))).unsqueeze(1).float()
        self._graph_degrees = [in_degrees(g) for g in self._graphs]
        self._line_graph_degrees = [in_degrees(lg) for lg in self._line_graphs]
        self._pm_pds = list(zip(*[g.edges() for g in self._graphs]))[0]

    def __len__(self):
        return len(self._graphs)

    def __getitem__(self, idx):
        return self._graphs[idx], self._line_graphs[idx], \
                self._graph_degrees[idx], self._line_graph_degrees[idx], self._pm_pds[idx]

    def _appendix_c(self):
        q = npr.uniform(0, self._avg_deg - math.sqrt(self._avg_deg))
        p = self._k * self._avg_deg - q
        if random.random() < 0.5:
            return p, q
        else:
            return q, p

    def collate_fn(self, x):
        g, lg, deg_g, deg_lg, pm_pd = zip(*x)
        g_batch = batch(g)
        lg_batch = batch(lg)
        degg_batch = np.concatenate(deg_g, axis=0)
        deglg_batch = np.concatenate(deg_lg, axis=0)
        pm_pd_batch = np.concatenate([x + i * self._n_nodes for i, x in enumerate(pm_pd)], axis=0)
        return g_batch, lg_batch, degg_batch, deglg_batch, pm_pd_batch


class SBMMixture(SBMMixtureDataset):
    def __init__(self,
                 n_graphs,
                 n_nodes,
                 n_communities,
                 k=2,
                 avg_deg=3,
                 pq='Appendix_C',
                 rng=None):
        deprecate_class('SBMMixture', 'SBMMixtureDataset')
        super(SBMMixture, self).__init__(n_graphs=n_graphs,
                                         n_nodes=n_nodes,
                                         n_communities=n_communities,
                                         k=k,
                                         avg_deg=avg_deg,
                                         pq=pq,
                                         rng=rng)
