"""Torch Module for E(n) Equivariant Graph Convolution Layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch
import torch.nn as nn

class EGNNConv(nn.Module):
    r"""E(n) Equivariant Convolutional Layer from `E(n) Equivariant Graph
    Neural Networks <https://arxiv.org/abs/2102.09844>`__

    .. math::
        m_{ij}=\phi_e(h_i^l, h_j^l, ||x_i^l-x_j^l||^2, a_{ij})
        x_i^{l+1} = x_i^l + C\sum(x_i^l-x_j^l)\phi_x(m_{ij})
        m_i = \sum m_{ij}
        h_i^{l+1} = \phi_h(h_i^l, m_i)

    where :math:`h_i`, :math:`x_i`, :math:`a_{ij}` are node features, coordinate
    features, and edge features respectively. :math:`\phi_e`, :math:`\phi_h`, and
    :math:`\phi_x` are two-layer MLPs. :math:`C` is a constant for normalization,
    computed as :math:`1/|\mathcal{N}(i)|`.

    Parameters
    ----------
    in_size : int
        Input feature size; i.e. the size of :math:`h_i^l`.
    hidden_size : int
        Hidden feature size; i.e. the size of hidden layer in :math:`\phi(\cdot)`.
    out_size : int
        Output feature size; i.e. the size of :math:`h_i^{l+1}`.
    edge_feat_size : int, optional
        Edge feature size; i.e. the size of :math:`a_{ij}`. Default: 0.

    Example
    -------
    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import EGNNConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> node_feat, coord_feat, edge_feat = th.ones(6, 10), th.ones(6, 3), th.ones(6, 2)
    >>> conv = EGNNConv(10, 10, 10, 2)
    >>> h, x = conv(g, node_feat, coord_feat, edge_feat)
    """
    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=0):
        super(EGNNConv, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()

        # \phi_e
        self.edge_mlp = nn.Sequential(
            # +1 for the radial feature: |x_i - x_j|^2
            nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn
        )

        # \phi_h
        self.node_mlp = nn.Sequential(
            nn.Linear(in_size + hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, out_size)
        )

        # \phi_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1, bias=False)
        )

        def message_func(edges):
            coord_diff = edges.src['x'] - edges.dst['x']
            radial = coord_diff.square().sum(dim=1).unsqueeze(-1)
            # normalize coordinate difference
            coord_diff = coord_diff / (radial + 1e-30)

            # concat features for edge mlp
            if self.edge_feat_size > 0:
                f = torch.cat([edges.src['h'], edges.dst['h'], radial, edges.data['a']], dim=-1)
            else:
                f = torch.cat([edges.src['h'], edges.dst['h'], radial], dim=-1)

            msg_h = self.edge_mlp(f)
            msg_x = self.coord_mlp(msg_h) * coord_diff

            return {'msg_x': msg_x, 'msg_h': msg_h}

        def reduce_func(nodes):
            h = torch.sum(nodes.mailbox['msg_h'], dim=1)
            x = torch.mean(nodes.mailbox['msg_x'], dim=1)

            return {'h_neigh': h, 'x_neigh': x}

        self.message_func = message_func
        self.reduce_func = reduce_func

    def forward(self, graph, node_feat, coord_feat, edge_feat=None):
        r"""
        Description
        -----------
        Compute EGNN layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        node_feat : torch.Tensor
            The input feature of shape :math:`(N, h_n)`. :math:`N` is the number of
            nodes, and :math:`h_n` must be the same as in_size.
        coord_feat : torch.Tensor
            The coordinate feature of shape :math:`(N, *)` :math:`N` is the
            number of nodes, and :math:`*` could be of any shape.
        edge_feat : torch.Tensor, optional
            The edge feature of shape :math:`(M, h_e)`. :math:`M` is the number of
            edges, and :math:`h_e` must be the same as edge_feat_size.

        Returns
        -------
        node_feat_out : torch.Tensor
            The output node feature of shape :math:`(N, h_n')` where :math:`h_n'`
            should be the same as out_size.
        coord_feat_out: torch.Tensor
            The output coordinate feature of shape :math:`(N, *)` where :math:`*`
            should be the same as input coordinate shape.
        """
        with graph.local_scope():
            # node feature
            graph.ndata['h'] = node_feat
            # coordinate feature
            graph.ndata['x'] = coord_feat
            # edge feature
            if self.edge_feat_size > 0 and edge_feat is not None:
                graph.edata['a'] = edge_feat

            graph.update_all(self.message_func, self.reduce_func)
            h_neigh, x_neigh = graph.ndata['h_neigh'], graph.ndata['x_neigh']
            
            h = self.node_mlp(
                torch.cat([node_feat, h_neigh], dim=-1)
            )
            x = coord_feat + x_neigh

            return h, x
