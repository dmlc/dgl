"""Torch Module for GNNExplainer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch
import tqdm

from math import sqrt
from torch import nn

from ....base import NID, EID
from ....subgraph import khop_in_subgraph

class GNNExplainer(nn.Module):
    r"""

    Description
    -----------
    GNNExplainer model from paper `GNNExplainer: Generating Explanations for
    Graph Neural Networks <https://arxiv.org/abs/1903.03894>`__ for identifying
    compact subgraph structures and small subsets of node features that play a
    critical role in GNN-based node classification and graph classification.

    Parameters
    ----------
    model : nn.Module
        The GNN model to explain.

        * The required arguments of its forward function are graph and feat.
          The latter one is for input node features.
        * It should also optionally take an eweight argument for edge weights
          and multiply the messages by it in message passing.
        * The output of its forward function is the logits for the predicted
          node/graph classes.

        See also the example below.
    num_hops : int
        The number of hops for GNN information aggregation.
    lr : float, optional
        The learning rate to use, default to 0.01.
    num_epochs : int, optional
        The number of epochs to train.
    log : bool, optional
        If True, it will log the computation process, default to True.
    """
    def __init__(self,
                 model,
                 num_hops,
                 lr=0.01,
                 num_epochs=100,
                 log=True):
        super(GNNExplainer, self).__init__()
        self.model = model
        self.num_hops = num_hops
        self.lr = lr
        self.num_epochs = num_epochs
        self.log = log

    def _init_masks(self, feat, graph):
        r"""TODO"""
        num_nodes, feat_size = feat.size()
        num_edges = graph.num_edges()

        std = 0.1
        feat_mask = nn.Parameter(torch.randn(1, feat_size) * std)

        std = nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * num_nodes))
        edge_mask = torch.nn.Parameter(torch.randn(num_edges) * std)

        return feat_mask, edge_mask

    def _loss(self, )

    def explain_node(self, node_id, graph, feat, **kwargs):
        r"""Learns and returns a node feature mask and subgraph that play a
        crucial role to explain the prediction made by the GNN for node
        :attr:`node_id`.

        Parameters
        ----------
        node_id : int
            The node to explain.
        graph : DGLGraph
            A homogeneous graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, *)`. :math:`N` is the
            number of nodes, and :math:`*` could be of any shape.
        kwargs : dict
            Additional arguments passed to the GNN model. Tensors whose
            first dimension is the number of nodes or edges will be
            assumed to be node/edge features.

        Returns
        -------
        TODO
        """
        self.model.eval()
        num_nodes = graph.num_nodes()
        num_edges = graph.num_edges()

        # Extract node-centered k-hop subgraph and
        # its associated node and edge features.
        sg = khop_in_subgraph(graph, node_id, self.num_hops)
        sg_nodes = sg.ndata[NID]
        sg_edges = sg.edata[EID]
        feat = feat[sg_nodes]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item[sg_nodes]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item[sg_edges]
            kwargs[key] = item

        # Get the initial prediction
        with torch.no_grad():
            logits = self.model(graph=sg, feat=feat, **kwargs)
            log_probs = logits.log_softmax(dim=-1)
            pred_label = log_probs.argmax(dim=-1)

        feat_mask, edge_mask = self._init_masks(feat, sg)
        device = feat.device
        feat_mask = feat_mask.to(device)
        edge_mask = edge_mask.to(device)

        params = [feat_mask, edge_mask]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        if self.log:
            pbar = tqdm(total=self.num_epochs)
            pbar.set_description(f'Explain node {node_id}')

        for _ in range(self.num_epochs):
            optimizer.zero_grad()
            feat = feat * feat_mask.sigmoid()
            logits = self.model(graph=sg, feat=feat,
                                eweight=edge_mask.sigmoid())
            log_probs = logits.log_softmax(dim=-1)

    def explain_graph(self):
        r"""TODO"""
        self.model.eval()
