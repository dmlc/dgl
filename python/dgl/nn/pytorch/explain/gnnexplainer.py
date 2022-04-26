"""Torch Module for GNNExplainer"""
# pylint: disable= no-member, arguments-differ, invalid-name
from math import sqrt
import torch

from torch import nn
from tqdm import tqdm

from ....base import NID, EID
from ....subgraph import khop_in_subgraph

class GNNExplainer(nn.Module):
    r"""GNNExplainer model from `GNNExplainer: Generating Explanations for
    Graph Neural Networks <https://arxiv.org/abs/1903.03894>`__

    It identifies compact subgraph structures and small subsets of node features that play a
    critical role in GNN-based node classification and graph classification.

    To generate an explanation, it learns an edge mask :math:`M` and a feature mask :math:`F`
    by optimizing the following objective function.

    .. math::
      l(y, \hat{y}) + \alpha_1 \|M\|_1 + \alpha_2 H(M) + \beta_1 \|F\|_1 + \beta_2 H(F)

    where :math:`l` is the loss function, :math:`y` is the original model prediction,
    :math:`\hat{y}` is the model prediction with the edge and feature mask applied, :math:`H` is
    the entropy function.

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

        See also the example in :func:`explain_node` and :func:`explain_graph`.
    num_hops : int
        The number of hops for GNN information aggregation.
    lr : float, optional
        The learning rate to use, default to 0.01.
    num_epochs : int, optional
        The number of epochs to train.
    alpha1 : float, optional
        A higher value will make the explanation edge masks more sparse by decreasing
        the sum of the edge mask.
    alpha2 : float, optional
        A higher value will make the explanation edge masks more sparse by decreasing
        the entropy of the edge mask.
    beta1 : float, optional
        A higher value will make the explanation node feature masks more sparse by
        decreasing the mean of the node feature mask.
    beta2 : float, optional
        A higher value will make the explanation node feature masks more sparse by
        decreasing the entropy of the node feature mask.
    log : bool, optional
        If True, it will log the computation process, default to True.
    """

    def __init__(self,
                 model,
                 num_hops,
                 lr=0.01,
                 num_epochs=100,
                 *,
                 alpha1=0.005,
                 alpha2=1.0,
                 beta1=1.0,
                 beta2=0.1,
                 log=True):
        super(GNNExplainer, self).__init__()
        self.model = model
        self.num_hops = num_hops
        self.lr = lr
        self.num_epochs = num_epochs
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.log = log

    def _init_masks(self, graph, feat):
        r"""Initialize learnable feature and edge mask.

        Parameters
        ----------
        graph : DGLGraph
            Input graph.
        feat : Tensor
            Input node features.

        Returns
        -------
        feat_mask : Tensor
            Feature mask of shape :math:`(1, D)`, where :math:`D`
            is the feature size.
        edge_mask : Tensor
            Edge mask of shape :math:`(E)`, where :math:`E` is the
            number of edges.
        """
        num_nodes, feat_size = feat.size()
        num_edges = graph.num_edges()
        device = feat.device

        std = 0.1
        feat_mask = nn.Parameter(torch.randn(1, feat_size, device=device) * std)

        std = nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * num_nodes))
        edge_mask = nn.Parameter(torch.randn(num_edges, device=device) * std)

        return feat_mask, edge_mask

    def _loss_regularize(self, loss, feat_mask, edge_mask):
        r"""Add regularization terms to the loss.

        Parameters
        ----------
        loss : Tensor
            Loss value.
        feat_mask : Tensor
            Feature mask of shape :math:`(1, D)`, where :math:`D`
            is the feature size.
        edge_mask : Tensor
            Edge mask of shape :math:`(E)`, where :math:`E`
            is the number of edges.

        Returns
        -------
        Tensor
            Loss value with regularization terms added.
        """
        # epsilon for numerical stability
        eps = 1e-15

        edge_mask = edge_mask.sigmoid()
        # Edge mask sparsity regularization
        loss = loss + self.alpha1 * torch.sum(edge_mask)
        # Edge mask entropy regularization
        ent = - edge_mask * torch.log(edge_mask + eps) - \
            (1 - edge_mask) * torch.log(1 - edge_mask + eps)
        loss = loss + self.alpha2 * ent.mean()

        feat_mask = feat_mask.sigmoid()
        # Feature mask sparsity regularization
        loss = loss + self.beta1 * torch.mean(feat_mask)
        # Feature mask entropy regularization
        ent = - feat_mask * torch.log(feat_mask + eps) - \
            (1 - feat_mask) * torch.log(1 - feat_mask + eps)
        loss = loss + self.beta2 * ent.mean()

        return loss

    def explain_node(self, node_id, graph, feat, **kwargs):
        r"""Learn and return a node feature mask and subgraph that play a
        crucial role to explain the prediction made by the GNN for node
        :attr:`node_id`.

        Parameters
        ----------
        node_id : int
            The node to explain.
        graph : DGLGraph
            A homogeneous graph.
        feat : Tensor
            The input feature of shape :math:`(N, D)`. :math:`N` is the
            number of nodes, and :math:`D` is the feature size.
        kwargs : dict
            Additional arguments passed to the GNN model. Tensors whose
            first dimension is the number of nodes or edges will be
            assumed to be node/edge features.

        Returns
        -------
        new_node_id : Tensor
            The new ID of the input center node.
        sg : DGLGraph
            The subgraph induced on the k-hop in-neighborhood of :attr:`node_id`.
        feat_mask : Tensor
            Learned feature importance mask of shape :math:`(D)`, where :math:`D` is the
            feature size. The values are within range :math:`(0, 1)`.
            The higher, the more important.
        edge_mask : Tensor
            Learned importance mask of the edges in the subgraph, which is a tensor
            of shape :math:`(E)`, where :math:`E` is the number of edges in the
            subgraph. The values are within range :math:`(0, 1)`.
            The higher, the more important.

        Examples
        --------

        >>> import dgl
        >>> import dgl.function as fn
        >>> import torch
        >>> import torch.nn as nn
        >>> from dgl.data import CoraGraphDataset
        >>> from dgl.nn import GNNExplainer

        >>> # Load dataset
        >>> data = CoraGraphDataset()
        >>> g = data[0]
        >>> features = g.ndata['feat']
        >>> labels = g.ndata['label']
        >>> train_mask = g.ndata['train_mask']

        >>> # Define a model
        >>> class Model(nn.Module):
        ...     def __init__(self, in_feats, out_feats):
        ...         super(Model, self).__init__()
        ...         self.linear = nn.Linear(in_feats, out_feats)
        ...
        ...     def forward(self, graph, feat, eweight=None):
        ...         with graph.local_scope():
        ...             feat = self.linear(feat)
        ...             graph.ndata['h'] = feat
        ...             if eweight is None:
        ...                 graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        ...             else:
        ...                 graph.edata['w'] = eweight
        ...                 graph.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
        ...             return graph.ndata['h']

        >>> # Train the model
        >>> model = Model(features.shape[1], data.num_classes)
        >>> criterion = nn.CrossEntropyLoss()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        >>> for epoch in range(10):
        ...     logits = model(g, features)
        ...     loss = criterion(logits[train_mask], labels[train_mask])
        ...     optimizer.zero_grad()
        ...     loss.backward()
        ...     optimizer.step()

        >>> # Explain the prediction for node 10
        >>> explainer = GNNExplainer(model, num_hops=1)
        >>> new_center, sg, feat_mask, edge_mask = explainer.explain_node(10, g, features)
        >>> new_center
        tensor([1])
        >>> sg.num_edges()
        12
        >>> # Old IDs of the nodes in the subgraph
        >>> sg.ndata[dgl.NID]
        tensor([ 9, 10, 11, 12])
        >>> # Old IDs of the edges in the subgraph
        >>> sg.edata[dgl.EID]
        tensor([51, 53, 56, 48, 52, 57, 47, 50, 55, 46, 49, 54])
        >>> feat_mask
        tensor([0.2638, 0.2738, 0.3039,  ..., 0.2794, 0.2643, 0.2733])
        >>> edge_mask
        tensor([0.0937, 0.1496, 0.8287, 0.8132, 0.8825, 0.8515, 0.8146, 0.0915, 0.1145,
                0.9011, 0.1311, 0.8437])
        """
        self.model.eval()
        num_nodes = graph.num_nodes()
        num_edges = graph.num_edges()

        # Extract node-centered k-hop subgraph and
        # its associated node and edge features.
        sg, inverse_indices = khop_in_subgraph(graph, node_id, self.num_hops)
        sg_nodes = sg.ndata[NID].long()
        sg_edges = sg.edata[EID].long()
        feat = feat[sg_nodes]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[sg_nodes]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[sg_edges]
            kwargs[key] = item

        # Get the initial prediction.
        with torch.no_grad():
            logits = self.model(graph=sg, feat=feat, **kwargs)
            pred_label = logits.argmax(dim=-1)

        feat_mask, edge_mask = self._init_masks(sg, feat)

        params = [feat_mask, edge_mask]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        if self.log:
            pbar = tqdm(total=self.num_epochs)
            pbar.set_description('Explain node {node_id}')

        for _ in range(self.num_epochs):
            optimizer.zero_grad()
            h = feat * feat_mask.sigmoid()
            logits = self.model(graph=sg, feat=h,
                                eweight=edge_mask.sigmoid(), **kwargs)
            log_probs = logits.log_softmax(dim=-1)
            loss = -log_probs[inverse_indices, pred_label[inverse_indices]]
            loss = self._loss_regularize(loss, feat_mask, edge_mask)
            loss.backward()
            optimizer.step()

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

        feat_mask = feat_mask.detach().sigmoid().squeeze()
        edge_mask = edge_mask.detach().sigmoid()

        return inverse_indices, sg, feat_mask, edge_mask

    def explain_graph(self, graph, feat, **kwargs):
        r"""Learn and return a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for a graph.

        Parameters
        ----------
        graph : DGLGraph
            A homogeneous graph.
        feat : Tensor
            The input feature of shape :math:`(N, D)`. :math:`N` is the
            number of nodes, and :math:`D` is the feature size.
        kwargs : dict
            Additional arguments passed to the GNN model. Tensors whose
            first dimension is the number of nodes or edges will be
            assumed to be node/edge features.

        Returns
        -------
        feat_mask : Tensor
            Learned feature importance mask of shape :math:`(D)`, where :math:`D` is the
            feature size. The values are within range :math:`(0, 1)`.
            The higher, the more important.
        edge_mask : Tensor
            Learned importance mask of the edges in the graph, which is a tensor
            of shape :math:`(E)`, where :math:`E` is the number of edges in the
            graph. The values are within range :math:`(0, 1)`. The higher,
            the more important.

        Examples
        --------

        >>> import dgl.function as fn
        >>> import torch
        >>> import torch.nn as nn
        >>> from dgl.data import GINDataset
        >>> from dgl.dataloading import GraphDataLoader
        >>> from dgl.nn import AvgPooling, GNNExplainer

        >>> # Load dataset
        >>> data = GINDataset('MUTAG', self_loop=True)
        >>> dataloader = GraphDataLoader(data, batch_size=64, shuffle=True)

        >>> # Define a model
        >>> class Model(nn.Module):
        ...     def __init__(self, in_feats, out_feats):
        ...         super(Model, self).__init__()
        ...         self.linear = nn.Linear(in_feats, out_feats)
        ...         self.pool = AvgPooling()
        ...
        ...     def forward(self, graph, feat, eweight=None):
        ...         with graph.local_scope():
        ...             feat = self.linear(feat)
        ...             graph.ndata['h'] = feat
        ...             if eweight is None:
        ...                 graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        ...             else:
        ...                 graph.edata['w'] = eweight
        ...                 graph.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
        ...             return self.pool(graph, graph.ndata['h'])

        >>> # Train the model
        >>> feat_size = data[0][0].ndata['attr'].shape[1]
        >>> model = Model(feat_size, data.gclasses)
        >>> criterion = nn.CrossEntropyLoss()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        >>> for bg, labels in dataloader:
        ...     logits = model(bg, bg.ndata['attr'])
        ...     loss = criterion(logits, labels)
        ...     optimizer.zero_grad()
        ...     loss.backward()
        ...     optimizer.step()

        >>> # Explain the prediction for graph 0
        >>> explainer = GNNExplainer(model, num_hops=1)
        >>> g, _ = data[0]
        >>> features = g.ndata['attr']
        >>> feat_mask, edge_mask = explainer.explain_graph(g, features)
        >>> feat_mask
        tensor([0.2362, 0.2497, 0.2622, 0.2675, 0.2649, 0.2962, 0.2533])
        >>> edge_mask
        tensor([0.2154, 0.2235, 0.8325, ..., 0.7787, 0.1735, 0.1847])
        """
        self.model.eval()

        # Get the initial prediction.
        with torch.no_grad():
            logits = self.model(graph=graph, feat=feat, **kwargs)
            pred_label = logits.argmax(dim=-1)

        feat_mask, edge_mask = self._init_masks(graph, feat)

        params = [feat_mask, edge_mask]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        if self.log:
            pbar = tqdm(total=self.num_epochs)
            pbar.set_description('Explain graph')

        for _ in range(self.num_epochs):
            optimizer.zero_grad()
            h = feat * feat_mask.sigmoid()
            logits = self.model(graph=graph, feat=h,
                                eweight=edge_mask.sigmoid(), **kwargs)
            log_probs = logits.log_softmax(dim=-1)
            loss = -log_probs[0, pred_label[0]]
            loss = self._loss_regularize(loss, feat_mask, edge_mask)
            loss.backward()
            optimizer.step()

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

        feat_mask = feat_mask.detach().sigmoid().squeeze()
        edge_mask = edge_mask.detach().sigmoid()

        return feat_mask, edge_mask
