"""Torch Module for GatedGCN layer"""
# pylint: disable= no-member, arguments-differ, invalid-name, cell-var-from-loop
import torch
import torch.nn.functional as F
from torch import nn

from .... import function as fn


class GatedGCNConv(nn.Module):
    r"""Gated graph convolutional layer from `Benchmarking Graph Neural Networks
    <https://arxiv.org/abs/2003.00982>`__

    .. math::
        e_{ij}^{l+1}=D^l h_{i}^{l}+E^l h_{j}^{l}+C^l e_{ij}^{l}

        norm_{ij}=\Sigma_{j\in N_{i}} \sigma\left(e_{ij}^{l+1}\right)+\varepsilon

        \hat{e}_{ij}^{l+1}=\sigma(e_{ij}^{l+1}) / norm_{ij}

        h_{i}^{l+1}=A^l h_{i}^{l}+\Sigma_{j \in N_{i}} \hat{e}_{ij}^{l+1} \odot B^l h_{j}^{l}

    where :math:`h_{i}^{l}` is node :math:`i` feature of layer :math:`l`,
    :math:`e_{ij}^{l}` is edge :math:`ij` feature of layer :math:`l`,
    :math:`\sigma` is sigmoid function, :math:`\varepsilon` is a small fixed constant
    for numerical stability, :math:`A^l, B^l, C^l, D^l, E^l` are linear layers.

    Parameters
    ----------
    input_feats : int
        Input feature size; i.e, the number of dimensions of :math:`h_{i}^{l}`.
    edge_feats: int
        Edge feature size; i.e., the number of dimensions of :math:`e_{ij}^{l}`.
    output_feats : int
        Output feature size; i.e., the number of dimensions of :math:`h_{i}^{l+1}`.
    dropout : float, optional
        Dropout rate on node and edge feature. Default: ``0``.
    batch_norm : bool, optional
        Whether to include batch normalization on node and edge feature. Default: ``True``.
    residual : bool, optional
        Whether to include residual connections. Default: ``True``.
    activation : callable activation function/layer or None, optional
        If not None, apply an activation function to the updated node features.
        Default: ``F.relu``.

    Example
    -------
    >>> import dgl
    >>> import torch as th
    >>> import torch.nn.functional as F
    >>> from dgl.nn import GatedGCNConv

    >>> num_nodes, num_edges = 8, 30
    >>> graph = dgl.rand_graph(num_nodes,num_edges)
    >>> node_feats = th.rand(num_nodes, 20)
    >>> edge_feats = th.rand(num_edges, 12)
    >>> gatedGCN = GatedGCNConv(20, 12, 20)
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
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        self.residual = residual

        if input_feats != output_feats or edge_feats != output_feats:
            self.residual = False

        # Linearly transform the node features.
        self.A = nn.Linear(input_feats, output_feats, bias=True)
        self.B = nn.Linear(input_feats, output_feats, bias=True)
        self.D = nn.Linear(input_feats, output_feats, bias=True)
        self.E = nn.Linear(input_feats, output_feats, bias=True)

        # Linearly transform the edge features.
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
            The input edge feature of shape :math:`(E, D_{edge})`,
            where :math:`E` is the number of edges and :math:`D_{edge}`
            is the size of the edge features.

        Returns
        -------
        torch.Tensor
            The output node feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is the output feature size.
        torch.Tensor
            The output edge feature of shape :math:`(E, D_{out})` where :math:`D_{out}`
            is the output feature size.
        """
        with graph.local_scope():
            # For residual connection
            h_in = feat
            e_in = edge_feat

            graph.ndata["Ah"] = self.A(feat)
            graph.ndata["Bh"] = self.B(feat)
            graph.ndata["Dh"] = self.D(feat)
            graph.ndata["Eh"] = self.E(feat)
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

            # Residual connection.
            if self.residual:
                feat = h_in + feat
                edge_feat = e_in + edge_feat

            feat = self.dropout(feat)
            edge_feat = self.dropout(edge_feat)

            return feat, edge_feat
