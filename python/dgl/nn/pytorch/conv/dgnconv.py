"""Torch Module for Directional Graph Networks Convolution Layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
from functools import partial
import torch
import torch.nn as nn
from .pnaconv import AGGREGATORS, SCALERS, PNAConv, PNAConvTower, scale_identity

def aggregate_dir_av(h, eig_s, eig_d, eig_idx):
    """directional average aggregation"""
    h_mod = torch.mul(h, (
        torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]) /
            (torch.sum(torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]),
            keepdim=True, dim=1) + 1e-30)).unsqueeze(-1))
    return torch.sum(h_mod, dim=1)

def aggregate_dir_dx(h, eig_s, eig_d, h_in, eig_idx):
    """directional derivative aggregation"""
    eig_w = ((
        eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]) /
        (torch.sum(
            torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]),
            keepdim=True, dim=1) + 1e-30
        )
    ).unsqueeze(-1)
    h_mod = torch.mul(h, eig_w)
    return torch.abs(torch.sum(h_mod, dim=1) - torch.sum(eig_w, dim=1) * h_in)

for k in range(1, 4):
    AGGREGATORS[f'dir{k}-av'] = partial(aggregate_dir_av, eig_idx=k)
    AGGREGATORS[f'dir{k}-dx'] = partial(aggregate_dir_dx, eig_idx=k)

class DGNConvTower(PNAConvTower):
    """A single DGN tower with modified reduce function"""
    def message(self, edges):
        """message function for DGN layer"""
        if self.edge_feat_size > 0:
            f = torch.cat([edges.src['h'], edges.dst['h'], edges.data['a']], dim=-1)
        else:
            f = torch.cat([edges.src['h'], edges.dst['h']], dim=-1)
        return {'msg': self.M(f), 'eig_s': edges.src['eig'], 'eig_d': edges.dst['eig']}

    def reduce_func(self, nodes):
        """reduce function for DGN layer"""
        h_in = nodes.data['h']
        eig_s = nodes.mailbox['eig_s']
        eig_d = nodes.mailbox['eig_d']
        msg = nodes.mailbox['msg']
        degree = msg.size(1)

        h = []
        for aggregator in self.aggregators:
            agg_name = aggregator.__name__
            if agg_name.startswith('dir'):
                if agg_name.endswith('av'):
                    h.append(aggregator(msg, eig_s, eig_d))
                else:
                    h.append(aggregator(msg, eig_s, eig_d, h_in))
            else:
                h.append(aggregator(msg))
        h = torch.cat(h, dim=1)
        h = torch.cat([
            scaler(h, D=degree, delta=self.delta) if scaler is not scale_identity else h
            for scaler in self.scalers
        ], dim=1)
        return {'h_neigh': h}

class DGNConv(PNAConv):
    r"""Directional Graph Network Layer from `Directional Graph Networks
    <https://arxiv.org/abs/2010.02863>`__

    DGN provides two special directional aggregators according to the vector field
    :math:`F`, which is defined as the gradient of the low-frequency eigenvectors of graph
    laplacian.

    The directional average aggregator is defined as
    :math:`h_i' = \sum_{j\in\mathcal{N}(i)}\frac{|F_{i,j}|\cdot h_j}{||F_{i,:}||_1+\epsilon}`

    The directional derivative aggregator is defined as
    :math:`h_i' = \sum_{j\in\mathcal{N}(i)}\frac{F_{i,j}\cdot h_j}{||F_{i,:}||_1+\epsilon}
    -h_i\cdot\sum_{j\in\mathcal{N}(i)}\frac{F_{i,j}}{||F_{i,:}||_1+\epsilon}`

    :math:`\epsilon` is the infinitesimal to keep the computation numerically stable.

    Parameters
    ----------
    in_size : int
        Input feature size; i.e. the size of :math:`h_i^l`.
    out_size : int
        Output feature size; i.e. the size of :math:`h_i^{l+1}`.
    aggregators : list of str
        List of aggregation function names(each aggregator specifies a way to aggregate
        messages from neighbours), selected from:

        * ``mean``: the mean of neighbour messages

        * ``max``: the maximum of neighbour messages

        * ``min``: the minimum of neighbour messages

        * ``std``: the standard deviation of neighbour messages

        * ``var``: the variance of neighbour messages

        * ``sum``: the sum of neighbour messages

        * ``moment3``, ``moment4``, ``moment5``: the normalized moments aggregation
        :math:`(E[(X-E[X])^n])^{1/n}`

        * ``dir{k}-av``: directional average aggregation with directions defined by the k-th
        smallest eigenvectors. k can be selected from 1, 2, 3.

        * ``dir{k}-dx``: directional derivative aggregation with directions defined by the k-th
        smallest eigenvectors. k can be selected from 1, 2, 3.

        Note that using directional aggregation requires the LaplacianPE transform on the input
        graph for eigenvector computation (the PE size must be >= k above).
    scalers: list of str
        List of scaler function names, selected from:

        * ``identity``: no scaling

        * ``amplification``: multiply the aggregated message by :math:`\log(d+1)/\delta`,
        where :math:`d` is the in-degree of the node.

        * ``attenuation``: multiply the aggregated message by :math:`\delta/\log(d+1)`
    delta: float
        The in-degree-related normalization factor computed over the training set, used by scalers
        for normalization. :math:`E[\log(d+1)]`, where :math:`d` is the degree for each node
        in the training set.
    dropout: float, optional
        The dropout ratio. Default: 0.0.
    num_towers: int, optional
        The number of towers used. Default: 1. Note that in_size and out_size must be divisible
        by num_towers.
    edge_feat_size: int, optional
        The edge feature size. Default: 0.
    residual : bool, optional
        The bool flag that determines whether to add a residual connection for the
        output. Default: True. If in_size and out_size of the DGN conv layer are not
        the same, this flag will be set as False forcibly.

    Example
    -------
    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import DGNConv
    >>> from dgl import LaplacianPE
    >>>
    >>> transform = LaplacianPE(k=3)
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = transform(g)
    >>> feat = th.ones(6, 10)
    >>> conv = DGNConv(10, 10, ['dir1-av', 'dir1-dx', 'sum'], ['identity', 'amplification'], 2.5)
    >>> ret = conv(g, feat)
    """
    def __init__(self, in_size, out_size, aggregators, scalers, delta,
        dropout=0., num_towers=1, edge_feat_size=0, residual=True):
        aggregators = [AGGREGATORS[aggr] for aggr in aggregators]
        scalers = [SCALERS[scale] for scale in scalers]

        self.in_size = in_size
        self.out_size = out_size
        assert in_size % num_towers == 0, 'in_size must be divisible by num_towers'
        assert out_size % num_towers == 0, 'out_size must be divisible by num_towers'
        self.tower_in_size = in_size // num_towers
        self.tower_out_size = out_size // num_towers
        self.edge_feat_size = edge_feat_size
        self.residual = residual
        if self.in_size != self.out_size:
            self.residual = False

        self.towers = nn.ModuleList([
            DGNConvTower(
                self.tower_in_size, self.tower_out_size,
                aggregators, scalers, delta,
                dropout=dropout, edge_feat_size=edge_feat_size
            ) for _ in range(num_towers)
        ])

        self.mixing_layer = nn.Sequential(
            nn.Linear(out_size, out_size),
            nn.LeakyReLU()
        )
