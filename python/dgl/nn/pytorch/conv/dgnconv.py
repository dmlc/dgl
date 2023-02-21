"""Torch Module for Directional Graph Networks Convolution Layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
from functools import partial

import torch
import torch.nn as nn

from .pnaconv import AGGREGATORS, PNAConv, PNAConvTower, SCALERS


def aggregate_dir_av(h, eig_s, eig_d, eig_idx):
    """directional average aggregation"""
    h_mod = torch.mul(
        h,
        (
            torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx])
            / (
                torch.sum(
                    torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]),
                    keepdim=True,
                    dim=1,
                )
                + 1e-30
            )
        ).unsqueeze(-1),
    )
    return torch.sum(h_mod, dim=1)


def aggregate_dir_dx(h, eig_s, eig_d, h_in, eig_idx):
    """directional derivative aggregation"""
    eig_w = (
        (eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx])
        / (
            torch.sum(
                torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]),
                keepdim=True,
                dim=1,
            )
            + 1e-30
        )
    ).unsqueeze(-1)
    h_mod = torch.mul(h, eig_w)
    return torch.abs(torch.sum(h_mod, dim=1) - torch.sum(eig_w, dim=1) * h_in)


for k in range(1, 4):
    AGGREGATORS[f"dir{k}-av"] = partial(aggregate_dir_av, eig_idx=k - 1)
    AGGREGATORS[f"dir{k}-dx"] = partial(aggregate_dir_dx, eig_idx=k - 1)


class DGNConvTower(PNAConvTower):
    """A single DGN tower with modified reduce function"""

    def message(self, edges):
        """message function for DGN layer"""
        if self.edge_feat_size > 0:
            f = torch.cat(
                [edges.src["h"], edges.dst["h"], edges.data["a"]], dim=-1
            )
        else:
            f = torch.cat([edges.src["h"], edges.dst["h"]], dim=-1)
        return {
            "msg": self.M(f),
            "eig_s": edges.src["eig"],
            "eig_d": edges.dst["eig"],
        }

    def reduce_func(self, nodes):
        """reduce function for DGN layer"""
        h_in = nodes.data["h"]
        eig_s = nodes.mailbox["eig_s"]
        eig_d = nodes.mailbox["eig_d"]
        msg = nodes.mailbox["msg"]
        degree = msg.size(1)

        h = []
        for agg in self.aggregators:
            if agg.startswith("dir"):
                if agg.endswith("av"):
                    h.append(AGGREGATORS[agg](msg, eig_s, eig_d))
                else:
                    h.append(AGGREGATORS[agg](msg, eig_s, eig_d, h_in))
            else:
                h.append(AGGREGATORS[agg](msg))
        h = torch.cat(h, dim=1)
        h = torch.cat(
            [
                SCALERS[scaler](h, D=degree, delta=self.delta)
                if scaler != "identity"
                else h
                for scaler in self.scalers
            ],
            dim=1,
        )
        return {"h_neigh": h}


class DGNConv(PNAConv):
    r"""Directional Graph Network Layer from `Directional Graph Networks
    <https://arxiv.org/abs/2010.02863>`__

    DGN introduces two special directional aggregators according to the vector field
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
        for normalization. :math:`E[\log(d+1)]`, where :math:`d` is the in-degree for each node
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
    >>> # DGN requires precomputed eigenvectors, with 'eig' as feature name.
    >>> transform = LaplacianPE(k=3, feat_name='eig')
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = transform(g)
    >>> eig = g.ndata['eig']
    >>> feat = th.ones(6, 10)
    >>> conv = DGNConv(10, 10, ['dir1-av', 'dir1-dx', 'sum'], ['identity', 'amplification'], 2.5)
    >>> ret = conv(g, feat, eig_vec=eig)
    """

    def __init__(
        self,
        in_size,
        out_size,
        aggregators,
        scalers,
        delta,
        dropout=0.0,
        num_towers=1,
        edge_feat_size=0,
        residual=True,
    ):
        super(DGNConv, self).__init__(
            in_size,
            out_size,
            aggregators,
            scalers,
            delta,
            dropout,
            num_towers,
            edge_feat_size,
            residual,
        )

        self.towers = nn.ModuleList(
            [
                DGNConvTower(
                    self.tower_in_size,
                    self.tower_out_size,
                    aggregators,
                    scalers,
                    delta,
                    dropout=dropout,
                    edge_feat_size=edge_feat_size,
                )
                for _ in range(num_towers)
            ]
        )

        self.use_eig_vec = False
        for aggr in aggregators:
            if aggr.startswith("dir"):
                self.use_eig_vec = True
                break

    def forward(self, graph, node_feat, edge_feat=None, eig_vec=None):
        r"""
        Description
        -----------
        Compute DGN layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        node_feat : torch.Tensor
            The input feature of shape :math:`(N, h_n)`. :math:`N` is the number of
            nodes, and :math:`h_n` must be the same as in_size.
        edge_feat : torch.Tensor, optional
            The edge feature of shape :math:`(M, h_e)`. :math:`M` is the number of
            edges, and :math:`h_e` must be the same as edge_feat_size.
        eig_vec : torch.Tensor, optional
            K smallest non-trivial eigenvectors of Graph Laplacian of shape :math:`(N, K)`.
            It is only required when :attr:`aggregators` contains directional aggregators.

        Returns
        -------
        torch.Tensor
            The output node feature of shape :math:`(N, h_n')` where :math:`h_n'`
            should be the same as out_size.
        """
        with graph.local_scope():
            if self.use_eig_vec:
                graph.ndata["eig"] = eig_vec
            return super().forward(graph, node_feat, edge_feat)
