"""Dataset for stochastic block model."""
import math
import os
import random

import numpy as np
import numpy.random as npr
import scipy as sp

from .. import batch
from ..convert import from_scipy
from .dgl_dataset import DGLDataset
from .utils import load_graphs, load_info, save_graphs, save_info


def sbm(n_blocks, block_size, p, q, rng=None):
    """(Symmetric) Stochastic Block Model

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
            block = sp.sparse.random(
                block_size,
                block_size,
                density,
                random_state=rng,
                data_rvs=lambda n: np.ones(n),
            )
            rows.append(block.row + i * block_size)
            cols.append(block.col + j * block_size)

    rows = np.hstack(rows)
    cols = np.hstack(cols)
    a = sp.sparse.coo_matrix(
        (np.ones(rows.shape[0]), (rows, cols)), shape=(n, n)
    )
    adj = sp.sparse.triu(a) + sp.sparse.triu(a, 1).transpose()
    return adj


class SBMMixtureDataset(DGLDataset):
    r"""Symmetric Stochastic Block Model Mixture

    Reference: Appendix C of `Supervised Community Detection with Hierarchical Graph Neural Networks <https://arxiv.org/abs/1705.08415>`_

    Parameters
    ----------
    n_graphs : int
        Number of graphs.
    n_nodes : int
        Number of nodes.
    n_communities : int
        Number of communities.
    k : int, optional
        Multiplier. Default: 2
    avg_deg : int, optional
        Average degree. Default: 3
    pq : list of pair of nonnegative float or str, optional
        Random densities. This parameter is for future extension,
        for now it's always using the default value.
        Default: Appendix_C
    rng : numpy.random.RandomState, optional
        Random number generator. If not given, it's numpy.random.RandomState() with `seed=None`,
        which read data from /dev/urandom (or the Windows analogue) if available or seed from
        the clock otherwise.
        Default: None

    Raises
    ------
    RuntimeError is raised if pq is not a list or string.

    Examples
    --------
    >>> data = SBMMixtureDataset(n_graphs=16, n_nodes=10000, n_communities=2)
    >>> from torch.utils.data import DataLoader
    >>> dataloader = DataLoader(data, batch_size=1, collate_fn=data.collate_fn)
    >>> for graph, line_graph, graph_degrees, line_graph_degrees, pm_pd in dataloader:
    ...     # your code here
    """

    def __init__(
        self,
        n_graphs,
        n_nodes,
        n_communities,
        k=2,
        avg_deg=3,
        pq="Appendix_C",
        rng=None,
    ):
        self._n_graphs = n_graphs
        self._n_nodes = n_nodes
        self._n_communities = n_communities
        assert n_nodes % n_communities == 0
        self._block_size = n_nodes // n_communities
        self._k = k
        self._avg_deg = avg_deg
        self._pq = pq
        self._rng = rng
        super(SBMMixtureDataset, self).__init__(
            name="sbmmixture",
            hash_key=(n_graphs, n_nodes, n_communities, k, avg_deg, pq, rng),
        )

    def process(self):
        pq = self._pq
        if type(pq) is list:
            assert len(pq) == self._n_graphs
        elif type(pq) is str:
            generator = {"Appendix_C": self._appendix_c}[pq]
            pq = [generator() for _ in range(self._n_graphs)]
        else:
            raise RuntimeError()
        self._graphs = [
            from_scipy(sbm(self._n_communities, self._block_size, *x))
            for x in pq
        ]
        self._line_graphs = [
            g.line_graph(backtracking=False) for g in self._graphs
        ]
        in_degrees = lambda g: g.in_degrees().float()
        self._graph_degrees = [in_degrees(g) for g in self._graphs]
        self._line_graph_degrees = [in_degrees(lg) for lg in self._line_graphs]
        self._pm_pds = list(zip(*[g.edges() for g in self._graphs]))[0]

    @property
    def graph_path(self):
        return os.path.join(self.save_path, "graphs_{}.bin".format(self.hash))

    @property
    def line_graph_path(self):
        return os.path.join(
            self.save_path, "line_graphs_{}.bin".format(self.hash)
        )

    @property
    def info_path(self):
        return os.path.join(self.save_path, "info_{}.pkl".format(self.hash))

    def has_cache(self):
        return (
            os.path.exists(self.graph_path)
            and os.path.exists(self.line_graph_path)
            and os.path.exists(self.info_path)
        )

    def save(self):
        save_graphs(self.graph_path, self._graphs)
        save_graphs(self.line_graph_path, self._line_graphs)
        save_info(
            self.info_path,
            {
                "graph_degree": self._graph_degrees,
                "line_graph_degree": self._line_graph_degrees,
                "pm_pds": self._pm_pds,
            },
        )

    def load(self):
        self._graphs, _ = load_graphs(self.graph_path)
        self._line_graphs, _ = load_graphs(self.line_graph_path)
        info = load_info(self.info_path)
        self._graph_degrees = info["graph_degree"]
        self._line_graph_degrees = info["line_graph_degree"]
        self._pm_pds = info["pm_pds"]

    def __len__(self):
        r"""Number of graphs in the dataset."""
        return len(self._graphs)

    def __getitem__(self, idx):
        r"""Get one example by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        graph: :class:`dgl.DGLGraph`
            The original graph
        line_graph: :class:`dgl.DGLGraph`
            The line graph of `graph`
        graph_degree: numpy.ndarray
            In degrees for each node in `graph`
        line_graph_degree: numpy.ndarray
            In degrees for each node in `line_graph`
        pm_pd: numpy.ndarray
            Edge indicator matrices Pm and Pd
        """
        return (
            self._graphs[idx],
            self._line_graphs[idx],
            self._graph_degrees[idx],
            self._line_graph_degrees[idx],
            self._pm_pds[idx],
        )

    def _appendix_c(self):
        q = npr.uniform(0, self._avg_deg - math.sqrt(self._avg_deg))
        p = self._k * self._avg_deg - q
        if random.random() < 0.5:
            return p, q
        else:
            return q, p

    def collate_fn(self, x):
        r"""The `collate` function for dataloader

        Parameters
        ----------
        x : tuple
            a batch of data that contains:

            - graph: :class:`dgl.DGLGraph`
                The original graph
            - line_graph: :class:`dgl.DGLGraph`
                The line graph of `graph`
            - graph_degree: numpy.ndarray
                In degrees for each node in `graph`
            - line_graph_degree: numpy.ndarray
                In degrees for each node in `line_graph`
            - pm_pd: numpy.ndarray
                Edge indicator matrices Pm and Pd

        Returns
        -------
        g_batch: :class:`dgl.DGLGraph`
            Batched graphs
        lg_batch: :class:`dgl.DGLGraph`
            Batched line graphs
        degg_batch: numpy.ndarray
            A batch of in degrees for each node in `g_batch`
        deglg_batch: numpy.ndarray
            A batch of in degrees for each node in `lg_batch`
        pm_pd_batch: numpy.ndarray
            A batch of edge indicator matrices Pm and Pd
        """
        g, lg, deg_g, deg_lg, pm_pd = zip(*x)
        g_batch = batch.batch(g)
        lg_batch = batch.batch(lg)
        degg_batch = np.concatenate(deg_g, axis=0)
        deglg_batch = np.concatenate(deg_lg, axis=0)
        pm_pd_batch = np.concatenate(
            [x + i * self._n_nodes for i, x in enumerate(pm_pd)], axis=0
        )
        return g_batch, lg_batch, degg_batch, deglg_batch, pm_pd_batch


SBMMixture = SBMMixtureDataset
