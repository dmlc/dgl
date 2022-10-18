"""Torch Module for GNNExplainer"""
# pylint: disable= no-member, arguments-differ, invalid-name
from math import sqrt
import torch

from torch import nn
from tqdm import tqdm

from ....base import NID, EID
from ....subgraph import khop_in_subgraph

__all__ = ['GNNExplainer', 'HeteroGNNExplainer']

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
            The subgraph induced on the k-hop in-neighborhood of the input center node.
        feat_mask : Tensor
            Learned node feature importance mask of shape :math:`(D)`, where :math:`D` is the
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
            pbar.set_description(f'Explain node {node_id}')

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

class HeteroGNNExplainer(nn.Module):
    r"""GNNExplainer model from `GNNExplainer: Generating Explanations for
    Graph Neural Networks <https://arxiv.org/abs/1903.03894>`__, adapted for heterogeneous graphs

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
        super(HeteroGNNExplainer, self).__init__()
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
        feat : dict[str, Tensor]
            The dictionary that associates input node features (values) with
            the respective node types (keys) present in the graph.

        Returns
        -------
        feat_masks : dict[str, Tensor]
            The dictionary that associates the node feature masks (values) with
            the respective node types (keys). The feature masks are of shape :math:`(1, D_t)`,
            where :math:`D_t` is the feature size for node type :math:`t`.
        edge_masks : dict[tuple[str], Tensor]
            The dictionary that associates the edge masks (values) with
            the respective canonical edge types (keys). The edge masks are of shape :math:`(E_t)`,
            where :math:`E_t` is the number of edges for canonical edge type :math:`t`.
        """
        device = graph.device
        feat_masks = {}
        std = 0.1
        for node_type, feature in feat.items():
            num_nodes, feat_size = feature.size()
            feat_masks[node_type] = nn.Parameter(torch.randn(1, feat_size, device=device) * std)

        edge_masks = {}
        for canonical_etype in graph.canonical_etypes:
            src_num_nodes = graph.num_nodes(canonical_etype[0])
            dst_num_nodes = graph.num_nodes(canonical_etype[-1])
            num_nodes_sum = src_num_nodes + dst_num_nodes
            num_edges = graph.num_edges(canonical_etype)
            std = nn.init.calculate_gain('relu')
            if num_nodes_sum > 0:
                std *= sqrt(2.0 / num_nodes_sum)
            edge_masks[canonical_etype] = nn.Parameter(
                torch.randn(num_edges, device=device) * std)

        return feat_masks, edge_masks

    def _loss_regularize(self, loss, feat_masks, edge_masks):
        r"""Add regularization terms to the loss.

        Parameters
        ----------
        loss : Tensor
            Loss value.
        feat_masks : dict[str, Tensor]
            The dictionary that associates the node feature masks (values) with
            the respective node types (keys). The feature masks are of shape :math:`(1, D_t)`,
            where :math:`D_t` is the feature size for node type :math:`t`.
        edge_masks : dict[tuple[str], Tensor]
            The dictionary that associates the edge masks (values) with
            the respective canonical edge types (keys). The edge masks are of shape :math:`(E_t)`,
            where :math:`E_t` is the number of edges for canonical edge type :math:`t`.

        Returns
        -------
        Tensor
            Loss value with regularization terms added.
        """
        # epsilon for numerical stability
        eps = 1e-15

        for edge_mask in edge_masks.values():
            edge_mask = edge_mask.sigmoid()
            # Edge mask sparsity regularization
            loss = loss + self.alpha1 * torch.sum(edge_mask)
            # Edge mask entropy regularization
            ent = - edge_mask * torch.log(edge_mask + eps) - \
                (1 - edge_mask) * torch.log(1 - edge_mask + eps)
            loss = loss + self.alpha2 * ent.mean()

        for feat_mask in feat_masks.values():
            feat_mask = feat_mask.sigmoid()
            # Feature mask sparsity regularization
            loss = loss + self.beta1 * torch.mean(feat_mask)
            # Feature mask entropy regularization
            ent = - feat_mask * torch.log(feat_mask + eps) - \
                (1 - feat_mask) * torch.log(1 - feat_mask + eps)
            loss = loss + self.beta2 * ent.mean()

        return loss

    def explain_node(self, ntype, node_id, graph, feat, **kwargs):
        r"""Learn and return node feature masks and a subgraph that play a
        crucial role to explain the prediction made by the GNN for node
        :attr:`node_id` of type :attr:`ntype`.

        Parameters
        ----------
        ntype : str
            The type of the node to explain. :attr:`model` must be trained to
            make predictions for this particular node type.
        node_id : int
            The ID of the node to explain.
        graph : DGLGraph
            A heterogeneous graph.
        feat : dict[str, Tensor]
            The dictionary that associates input node features (values) with
            the respective node types (keys) present in the graph.
            The input features are of shape :math:`(N_t, D_t)`. :math:`N_t` is the
            number of nodes for node type :math:`t`, and :math:`D_t` is the feature size for
            node type :math:`t`
        kwargs : dict
            Additional arguments passed to the GNN model.

        Returns
        -------
        new_node_id : Tensor
            The new ID of the input center node.
        sg : DGLGraph
            The subgraph induced on the k-hop in-neighborhood of the input center node.
        feat_mask : dict[str, Tensor]
            The dictionary that associates the learned node feature importance masks (values) with
            the respective node types (keys). The masks are of shape :math:`(D_t)`, where
            :math:`D_t` is the node feature size for node type :attr:`t`. The values are within range
            :math:`(0, 1)`. The higher, the more important.
        edge_mask : dict[Tuple[str], Tensor]
            The dictionary that associates the learned edge importance masks (values) with
            the respective canonical edge types (keys). The masks are of shape :math:`(E_t)`,
            where :math:`E_t` is the number of edges for canonical edge type :math:`t` in the
            subgraph. The values are within range :math:`(0, 1)`.
            The higher, the more important.

        Examples
        --------

        >>> import dgl
        >>> import torch as th
        >>> import torch.nn as nn
        >>> import torch.nn.functional as F
        >>> from dgl.backend import copy_to
        >>> from dgl.nn import HeteroGNNExplainer

        >>> # Create an example heterogenous dataset
        >>> def heterograph0():
        ...     g = dgl.heterograph({
        ...         ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 1, 1]),
        ...         ('creator', 'creates', 'game'): ([0, 1], [0, 1])}, device=dgl.backend.cpu())
        ...     g.nodes['user'].data['h'] = copy_to(th.randn((g.number_of_nodes('user'), 3)),
        ...                                         dgl.backend.cpu())
        ...     g.nodes['game'].data['h'] = copy_to(th.randn((g.number_of_nodes('game'), 2)),
        ...                                         dgl.backend.cpu())
        ...     g.nodes['creator'].data['h'] = copy_to(th.randn((g.number_of_nodes('creator'), 3)),
        ...                                              dgl.backend.cpu())
        ...     g.edges['plays'].data['h'] = copy_to(th.randn((g.number_of_edges('plays'), 1)),
        ...                                          dgl.backend.cpu())
        ...     g.edges['creates'].data['h'] = copy_to(th.randn((g.number_of_edges('creates'), 5)),
        ...                                             dgl.backend.cpu())
        ...     return g

        >>> # Define a GCN layer that uses eweight
        >>> class GCNLayer(th.nn.Module):
        ...     def __init__(self, in_dim, out_dim, graph_relation_list):
        ...         super(GCNLayer, self).__init__()
        ...
        ...         self.relation_weight_matrix = th.nn.ModuleDict({
        ...             ''.join(relation): th.nn.Linear(in_dim, out_dim)
        ...             for relation in graph_relation_list
        ...         })
        ...
        ...     def forward(self, graph, node_feature_dict, eweight=None):
        ...         rel_func_dict = {}
        ...         for relation in graph.canonical_etypes:
        ...             relation_str = ''.join(relation)
        ...             wh = self.relation_weight_matrix[relation_str]\
        ...                (node_feature_dict[relation[0]])
        ...             graph.nodes[relation[0]].data[f'wh_{relation}'] = wh
        ...             if eweight is None:
        ...                 rel_func_dict[relation] = (dgl.function.copy_u(
        ...                     f'wh_{relation}', 'm'), dgl.function.mean('m', 'h'))
        ...             else:
        ...                 graph.edges[relation].data['w'] = eweight[relation]
        ...                 rel_func_dict[relation] = (dgl.function.u_mul_e(
        ...                     f'wh_{relation}', 'w', 'm'),
        ...                                            dgl.function.mean('m', 'h'))
        ...         graph.multi_update_all(rel_func_dict, 'sum')
        ...         return {ntype: graph.nodes[ntype].data['h']
        ...                 for ntype in graph.ntypes}

        >>> # Define a model
        >>> class Model(nn.Module):
        ...     def __init__(self, in_dim, hidden_dim, num_classes, graph_relation_list):
        ...         super(Model, self).__init__()
        ...         self.layer1 = GCNLayer(in_dim,hidden_dim,graph_relation_list)
        ...         self.layer1 = GCNLayer(hidden_dim,num_classes,graph_relation_list)
        ...
        ...     def forward(self, graph, feat, eweight=None):
        ...         x = self.layer(graph, feat, eweight=eweight)
        ...         for ntype in graph.ntypes:
        ...             x[ntype] = th.nn.functional.relu(x[ntype])
        ...         x = self.layer2(graph, x)
        ...         return x

        >>> # get the hetero-graph
        >>> g = heterograph0()

        >>> # constants related to GNN
        >>> input_dim = 5
        >>> hidden_dim = 10
        >>> output_dim = 1

        >>> # add self-loop and backward edges to the heterograph
        >>> transform1 = dgl.transforms.AddSelfLoop(new_etypes=True)
        >>> g = transform1(g)
        >>> transform2 = dgl.transforms.AddReverse(copy_edata=True)
        >>> g = transform2(g)

        >>> # define random feature for the node types
        >>> feat = {ntype: th.zeros((g.num_nodes(ntype), input_dim)) for ntype in g.ntypes}

        >>> # define and train the model
        >>> model = Model(input_dim, hidden_dim, output_dim, g.canonical_etypes)
        >>> optimizer = th.optim.Adam(model.parameters())
        >>> for epoch in range(10):
        ...         logits = model(g, feat)['creator']
        ...         loss = F.cross_entropy(logits, th.tensor([[1.],[1.]]))
        ...         optimizer.zero_grad()
        ...         loss.backward()
        ...         optimizer.step()

        >>> # Explain the prediction for graph
        >>> explainer = HeteroGNNExplainer(model, num_hops=1)
        >>> new_center, sg, feat_mask, edge_mask = explainer.explain_node('creator', 0, g, feat)
        >>> new_center
        tensor([0])
        >>> sg
        Graph(num_nodes={'creator': 1, 'game': 1, 'user': 0},
      num_edges={('creator', 'creates', 'game'): 1,
        ('creator', 'self', 'creator'): 2,
        ('game', 'rev_creates', 'creator'): 1,
        ('game', 'rev_plays', 'user'): 0,
        ('game', 'self', 'game'): 2,
        ('user', 'plays', 'game'): 0,
        ('user', 'self', 'user'): 0},
        metagraph=[('creator', 'game', 'creates'),
            ('creator', 'creator', 'self'),
            ('game', 'creator', 'rev_creates'),
            ('game', 'user', 'rev_plays'),
            ('game', 'game', 'self'),
            ('user', 'game', 'plays'),
            ('user', 'user', 'self')])
        >>> feat_mask
        {'creator': tensor([0.2642, 0.3033, 0.2860, 0.2705, 0.2613]),
        'game': tensor([0.2612, 0.2879, 0.2377, 0.2743, 0.2808]),
        'user': tensor([0.2628, 0.2766, 0.2642, 0.2893, 0.2990])}
        >>> edge_mask
        {('creator', 'creates', 'game'): tensor([0.0418]),
         ('creator', 'self', 'creator'): tensor([0.9269, 0.9802]),
        ('game', 'rev_creates', 'creator'): tensor([0.1715]),
        ('game', 'rev_plays', 'user'): tensor([]),
        ('game', 'self', 'game'): tensor([0.9914, 0.9457]),
        ('user', 'plays', 'game'): tensor([]),
        ('user', 'self', 'user'): tensor([])}
        """
        self.model.eval()
        device = graph.device

        # Extract node-centered k-hop subgraph and
        # its associated node and edge features.
        sg, inverse_indices = khop_in_subgraph(graph, {ntype: node_id}, self.num_hops)
        inverse_indices = inverse_indices[ntype]
        sg_nodes = sg.ndata[NID]
        sg_edges = sg.edata[EID]
        sg_feat = {}

        for node_type in sg_nodes.keys():
            sg_feat[node_type] = feat[node_type][sg_nodes[node_type].long()]

        # Get the initial prediction.
        with torch.no_grad():
            self.model = self.model.to(device=device)
            logits = self.model(graph=sg, feat=sg_feat, **kwargs)[ntype]
            pred_label = logits.argmax(dim=-1)

        feat_mask, edge_mask = self._init_masks(sg, sg_feat)

        params = [*feat_mask.values(), *edge_mask.values()]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        if self.log:
            pbar = tqdm(total=self.num_epochs)
            pbar.set_description(f'Explain node {node_id} with type {ntype}')

        for _ in range(self.num_epochs):
            optimizer.zero_grad()
            h = {}
            for node_type in sg_feat:
                h[node_type] = sg_feat[node_type] * feat_mask[node_type].sigmoid()
            eweight = {}
            for canonical_etype in edge_mask:
                eweight[canonical_etype] = edge_mask[canonical_etype].sigmoid()
            logits = self.model(graph=sg, feat=h,
                                eweight=eweight, **kwargs)[ntype]
            log_probs = logits.log_softmax(dim=-1)
            loss = -log_probs[inverse_indices, pred_label[inverse_indices]]
            loss = self._loss_regularize(loss, feat_mask, edge_mask)
            loss.backward()
            optimizer.step()

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

        for node_type in feat_mask:
            feat_mask[node_type] = feat_mask[node_type].detach().sigmoid().squeeze()
        for canonical_etype in edge_mask:
            edge_mask[canonical_etype] = edge_mask[canonical_etype].detach().sigmoid()

        return inverse_indices, sg, feat_mask, edge_mask

    def explain_graph(self, graph, feat, **kwargs):
        r"""Learn and return a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for a graph.

        Parameters
        ----------
        graph : DGLGraph
            A heterogeneous graph.
        feat : dict[str, Tensor]
            Dictionary of { ntypes: features }
            The dictionary that associates input node features(values) with
            the respective node types(keys) present in the graph.
            The input feature of shape :math:`(N, D)`. :math:`N` is the
            number of nodes, and :math:`D` is the feature size.
        kwargs : dict
            Additional arguments passed to the GNN model. Tensors whose
            first dimension is the number of nodes or edges will be
            assumed to be node/edge features.

        Returns
        -------
        feat_mask : dict[str, Tensor]
            Dictionary of { ntypes: features }
            The dictionary that associates the feature masks(values) with
            the respective node types(keys).
            Learned feature importance mask of shape :math:`(D)`, where :math:`D` is the
            feature size. The values are within range :math:`(0, 1)`.
            The higher, the more important.
        edge_mask : dict[Tuple[str], Tensor]
            Dictionary of { canonical_etypes: features }
            The dictionary that  associates the edge masks(values) with
            the respective canonical edge types(keys).
            Learned importance mask of the edges in the graph, which is a tensor
            of shape :math:`(E)`, where :math:`E` is the number of edges in the
            graph. The values are within range :math:`(0, 1)`. The higher,
            the more important.

        Examples
        --------

        >>> import dgl
        >>> import torch as th
        >>> import torch.nn as nn
        >>> import torch.nn.functional as F
        >>> from dgl.backend import copy_to
        >>> from dgl.nn import HeteroGNNExplainer

        >>> # Create an example heterogenous dataset
        >>> def heterograph0():
        ...     g = dgl.heterograph({
        ...         ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 1, 1]),
        ...         ('creator', 'creates', 'game'): ([0, 1], [0, 1])}, device=dgl.backend.cpu())
        ...     g.nodes['user'].data['h'] = copy_to(th.randn((g.number_of_nodes('user'), 3)),
        ...                                         dgl.backend.cpu())
        ...     g.nodes['game'].data['h'] = copy_to(th.randn((g.number_of_nodes('game'), 2)),
        ...                                         dgl.backend.cpu())
        ...     g.nodes['creator'].data['h'] = copy_to(th.randn((g.number_of_nodes('creator'), 3)),
        ...                                              dgl.backend.cpu())
        ...     g.edges['plays'].data['h'] = copy_to(th.randn((g.number_of_edges('plays'), 1)),
        ...                                          dgl.backend.cpu())
        ...     g.edges['creates'].data['h'] = copy_to(th.randn((g.number_of_edges('creates'), 5)),
        ...                                             dgl.backend.cpu())
        ...     return g

        >>> # Define a GCN layer that uses eweight
        >>> class GCNLayer(th.nn.Module):
        ...     def __init__(self, in_dim, out_dim, graph_relation_list):
        ...         super(GCNLayer, self).__init__()
        ...
        ...         self.relation_weight_matrix = th.nn.ModuleDict({
        ...             ''.join(relation): th.nn.Linear(in_dim, out_dim)
        ...             for relation in graph_relation_list
        ...         })
        ...
        ...     def forward(self, graph, node_feature_dict, eweight=None):
        ...         rel_func_dict = {}
        ...         for relation in graph.canonical_etypes:
        ...             relation_str = ''.join(relation)
        ...             wh = self.relation_weight_matrix[relation_str]\
        ...                (node_feature_dict[relation[0]])
        ...             graph.nodes[relation[0]].data[f'wh_{relation}'] = wh
        ...             if eweight is None:
        ...                 rel_func_dict[relation] = (dgl.function.copy_u(
        ...                     f'wh_{relation}', 'm'), dgl.function.mean('m', 'h'))
        ...             else:
        ...                 graph.edges[relation].data['w'] = eweight[relation]
        ...                 rel_func_dict[relation] = (dgl.function.u_mul_e(
        ...                     f'wh_{relation}', 'w', 'm'),
        ...                                            dgl.function.mean('m', 'h'))
        ...         graph.multi_update_all(rel_func_dict, 'sum')
        ...         return {ntype: graph.nodes[ntype].data['h']
        ...                 for ntype in graph.ntypes}

        >>> # Define a model
        >>> class Model(nn.Module):
        ...     def __init__(self, in_dim, hidden_dim, num_classes, graph_relation_list):
        ...         super(Model, self).__init__()
        ...         self.layer = GCNLayer(in_dim,hidden_dim,graph_relation_list)
        ...         self.linear = th.nn.Linear(hidden_dim,num_classes)
        ...
        ...     def forward(self, graph, feat, eweight=None):
        ...         x = self.layer(graph, feat, eweight=eweight)
        ...
        ...         with graph.local_scope():
        ...             graph.ndata['x'] = x
        ...             hg = 0
        ...             for ntype in graph.ntypes:
        ...                 if graph.num_nodes(ntype):
        ...                     hg = hg + dgl.mean_nodes(graph, 'x', ntype=ntype)
        ...
        ...                 return self.linear(hg)

        >>> # get the hetero-graph
        >>> g = heterograph0()

        >>> # constants related to GNN
        >>> input_dim = 5
        >>> hidden_dim = 10
        >>> output_dim = 1

        >>> # add self-loop and backward edges to the heterograph
        >>> transform1 = dgl.transforms.AddSelfLoop(new_etypes=True)
        >>> g = transform1(g)
        >>> transform2 = dgl.transforms.AddReverse(copy_edata=True)
        >>> g = transform2(g)

        >>> # define random feature for the node types
        >>> feat = {ntype: th.zeros((g.num_nodes(ntype), input_dim)) for ntype in g.ntypes}

        >>> # define and train the model
        >>> model = Model(input_dim, hidden_dim, output_dim, g.canonical_etypes)
        >>> optimizer = th.optim.Adam(model.parameters())
        >>> for epoch in range(10):
        ...         logits = model(g, feat)
        ...         loss = F.cross_entropy(logits, th.tensor([[1.]]))
        ...         optimizer.zero_grad()
        ...         loss.backward()
        ...         optimizer.step()

        >>> # Explain the prediction for graph
        >>> explainer = HeteroGNNExplainer(model, num_hops=1)
        >>> feat_mask, edge_mask = explainer.explain_graph(g, feat)
        >>> feat_mask
        {'creator': tensor([0.2783, 0.2653, 0.2823, 0.2693, 0.2606]),
        'game': tensor([0.2532, 0.2781, 0.2650, 0.2846, 0.2872]),
        'user': tensor([0.2760, 0.3348, 0.2896, 0.2561, 0.2842])}
        >>> edge_mask
        {('creator', 'creates', 'game'): tensor([0.1256, 0.8210]),
        ('creator', 'self', 'creator'): tensor([0.8184, 0.2053, 0.9353, 0.1962]),
        ('game', 'rev_creates', 'creator'): tensor([0.8273, 0.8215]),
        ('game', 'rev_plays', 'user'): tensor([0.1637, 0.1762, 0.1447, 0.2001]),
        ('game', 'self', 'game'): tensor([0.1917, 0.9373, 0.8214, 0.8965]),
        ('user', 'plays', 'game'): tensor([0.7941, 0.1954, 0.2036, 0.0748]),
        ('user', 'self', 'user'): tensor([0.1303, 0.1315, 0.1977, 0.9092, 0.9010, 0.8202])}
        """
        self.model.eval()
        device = graph.device

        # Get the initial prediction.
        with torch.no_grad():
            self.model = self.model.to(device=device)
            logits = self.model(graph=graph, feat=feat, **kwargs)
            pred_label = logits.argmax(dim=-1)

        feat_mask, edge_mask = self._init_masks(graph, feat)

        params = [*feat_mask.values(), *edge_mask.values()]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        if self.log:
            pbar = tqdm(total=self.num_epochs)
            pbar.set_description('Explain graph')

        for _ in range(self.num_epochs):
            optimizer.zero_grad()
            h = {}
            for node_type in feat:
                h[node_type] = feat[node_type] * feat_mask[node_type].sigmoid()
            eweight = {}
            for canonical_etype in edge_mask:
                eweight[canonical_etype] = edge_mask[canonical_etype].sigmoid()
            logits = self.model(graph=graph, feat=h,
                                eweight=eweight, **kwargs)
            log_probs = logits.log_softmax(dim=-1)
            loss = -log_probs[0, pred_label[0]]
            loss = self._loss_regularize(loss, feat_mask, edge_mask)
            loss.backward()
            optimizer.step()

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

        for node_type in feat_mask:
            feat_mask[node_type] = feat_mask[node_type].detach().sigmoid().squeeze()

        for canonical_etype in edge_mask:
            edge_mask[canonical_etype] = edge_mask[canonical_etype].detach().sigmoid()

        return feat_mask, edge_mask
