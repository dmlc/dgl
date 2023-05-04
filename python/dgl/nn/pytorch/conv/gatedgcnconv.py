"""Torch Module for GatedGCN layer"""
# pylint: disable= no-member, arguments-differ, invalid-name, cell-var-from-loop
import torch
import torch.nn.functional as F
from torch import nn

from .... import function as fn


class GatedGCNConv(nn.Module):
    r"""GatedGCN: Residual Gated Graph ConvNets
        <https://arxiv.org/pdf/1711.07553v2.pdf>`

    .. math::
        e_{ij}^{l+1}=D^{l} h_{i}^{l}+E^{l} h_{j}^{l}+C^l e_{ij}^{l}

        norm_{ij}=\Sigma_{j\in N_{i}} \sigma\left(e_{ij}^{l+1}\right)+\varepsilon

        \hat{e}_{ij}^{l+1}=\sigma(e_{ij}^{l+1}) \div norm_{ij}

        h_{i}^{l+1}=A^l h_{i}^{l}+\Sigma_{j \in N_{i}} \hat{e}_{ij}^{l+1} \odot B^l h_{j}^{l}

    Parameters
    ----------
    in_feats : int
        Input feature size; i.e, the number of dimensions of :math:`\mathbf{x}_i`.
    edge_feats: int
        Edge feature size; i.e., the number of dimensions of :math:\mathbf{e}_{j,i}`.
    out_feats : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(t+1)}`.
    dropout : float, optional
        Dropout rate on node and edge feature. Defaults: ``0``.
    batch_norm : bool
        Whether to include batch normalization on node . Default: ``True``.
    residual : bool
        Whether to include residual connection . Default: ``True``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Example
    -------
    >>> import dgl
    >>> import torch as th
    >>> import torch.nn.functional as F
    >>> from dgl.nn import GatedGCNConv
    >>> num_nodes, num_edges = 8, 30
    >>> graph = dgl.rand_graph(num_nodes,num_edges)
    >>> node_feats = th.rand((num_nodes, 20))
    >>> edge_feats = th.rand((num_edges, 12))
    >>> gatedGCN = GatedGCNConv(input_feats=20,
    ...                         edge_feats=12,
    ...                         output_feats=20,
    ...                         dropout=0.2,
    ...                         batch_norm=True,
    ...                         residual=True,
    ...                         activation=F.relu)
    >>> new_node_feats, new_edge_feats = gatedGCN(graph, node_feats, edge_feats)
    >>> new_node_feats.shape, new_edge_feats.shape
    (torch.Size([8, 20]), torch.Size([30, 20]))

    """

    def __init__(
        self,
        input_feats,
        edge_feats,
        output_feats,
        dropout=0,
        batch_norm=True,
        residual=True,
        activation=F.relu,
    ):
        super(GatedGCNConv, self).__init__()
        self._input_feats = input_feats
        self._output_feats = output_feats
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        self.residual = residual

        if input_feats != output_feats or edge_feats != output_feats:
            self.residual = False

        # Linearly tranform the node features.
        self.A = nn.Linear(input_feats, output_feats, bias=True)
        self.B = nn.Linear(input_feats, output_feats, bias=True)
        self.D = nn.Linear(input_feats, output_feats, bias=True)
        self.E = nn.Linear(input_feats, output_feats, bias=True)

        # Linearly tranform the edge features.
        self.C = nn.Linear(edge_feats, output_feats, bias=True)

        # Batch normalization on the node/edge features.
        self.bn_node = nn.BatchNorm1d(output_feats)
        self.bn_edge = nn.BatchNorm1d(output_feats)

        self.activation = activation

    def forward(self, graph, feat, edge_feat):
        """

        Description
        -----------
        Compute gated graph convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`N`
            is the number of nodes of the graph and :math:`D_{in}` is the
            input feature size.
        edge_feat : torch.Tensor
            The input edge feature of shape :math:`(E, D_{in_{edge}})`,
            where :math:`E` is the number of edges and :math:`D_{in_{edge}}`
            the size of the edge features.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is the output feature size.
        torch.Tensor
            The output edge feature of shape :math:`(E, D_{out})` where :math:`D_{out}`
            is the output feature size.
        """
        with graph.local_scope():
            # For residual connection
            h_in = feat
            e_in = edge_feat

            # Linearly tranform the node features.
            graph.ndata["h"] = feat
            graph.ndata["Ah"] = self.A(feat)
            graph.ndata["Bh"] = self.B(feat)
            graph.ndata["Dh"] = self.D(feat)
            graph.ndata["Eh"] = self.E(feat)

            # Linearly tranform the edge features.
            graph.edata["e"] = edge_feat
            graph.edata["Ce"] = self.C(edge_feat)

            graph.apply_edges(fn.u_add_v("Dh", "Eh", "DEh"))

            # Get edge feature
            graph.edata["e"] = graph.edata["DEh"] + graph.edata["Ce"]
            graph.edata["sigma"] = torch.sigmoid(graph.edata["e"])

            graph.update_all(
                fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
            )
            graph.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
            graph.ndata["h"] = graph.ndata["Ah"] + graph.ndata[
                "sum_sigma_h"
            ] / (graph.ndata["sum_sigma"] + 1e-6)

            # Result of graph convolution.
            feat = graph.ndata["h"]
            edge_feat = graph.edata["e"]

            # Batch normalization.
            if self.batch_norm:
                feat = self.bn_node(feat)
                edge_feat = self.bn_edge(edge_feat)

            # Non-linear activation.
            if self.activation:
                feat = self.activation(feat)
                edge_feat = self.activation(edge_feat)

            feat = self.dropout(feat)
            edge_feat = self.dropout(edge_feat)

            # Residual connection.
            if self.residual:
                feat = h_in + feat
                edge_feat = e_in + edge_feat

            return feat, edge_feat
