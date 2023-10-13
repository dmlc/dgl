"""Torch Module for PGExplainer"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .... import batch, ETYPE, khop_in_subgraph, NID, to_homogeneous

__all__ = ["PGExplainer", "HeteroPGExplainer"]


class PGExplainer(nn.Module):
    r"""PGExplainer from `Parameterized Explainer for Graph Neural Network
    <https://arxiv.org/pdf/2011.04573>`

    PGExplainer adopts a deep neural network (explanation network) to
    parameterize the generation process of explanations, which enables it to
    explain multiple instances collectively. PGExplainer models the underlying
    structure as edge distributions, from which the explanatory graph is
    sampled.

    Parameters
    ----------
    model : nn.Module
        The GNN model to explain that tackles multiclass graph classification

        * Its forward function must have the form
          :attr:`forward(self, graph, nfeat, embed, edge_weight)`.
        * The output of its forward function is the logits if embed=False else
          the intermediate node embeddings.
    num_features : int
        Node embedding size used by :attr:`model`.
    num_hops : int, optional
        The number of hops for GNN information aggregation, which must match the
        number of message passing layers employed by the GNN to be explained.
    explain_graph : bool, optional
        Whether to initialize the model for graph-level or node-level predictions.
    coff_budget : float, optional
        Size regularization to constrain the explanation size. Default: 0.01.
    coff_connect : float, optional
        Entropy regularization to constrain the connectivity of explanation. Default: 5e-4.
    sample_bias : float, optional
        Some members of a population are systematically more likely to be selected
        in a sample than others. Default: 0.0.
    """

    def __init__(
        self,
        model,
        num_features,
        num_hops=None,
        explain_graph=True,
        coff_budget=0.01,
        coff_connect=5e-4,
        sample_bias=0.0,
    ):
        super(PGExplainer, self).__init__()

        self.model = model
        self.graph_explanation = explain_graph
        # Node explanation requires additional self-embedding data.
        self.num_features = num_features * (2 if self.graph_explanation else 3)
        self.num_hops = num_hops

        # training hyperparameters for PGExplainer
        self.coff_budget = coff_budget
        self.coff_connect = coff_connect
        self.sample_bias = sample_bias

        self.init_bias = 0.0

        # Explanation network in PGExplainer
        self.elayers = nn.Sequential(
            nn.Linear(self.num_features, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def set_masks(self, graph, edge_mask=None):
        r"""Set the edge mask that plays a crucial role to explain the
        prediction made by the GNN for a graph. Initialize learnable edge
        mask if it is None.

        Parameters
        ----------
        graph : DGLGraph
            A homogeneous graph.
        edge_mask : Tensor, optional
            Learned importance mask of the edges in the graph, which is a tensor
            of shape :math:`(E)`, where :math:`E` is the number of edges in the
            graph. The values are within range :math:`(0, 1)`. The higher,
            the more important. Default: None.
        """
        if edge_mask is None:
            num_nodes = graph.num_nodes()
            num_edges = graph.num_edges()

            init_bias = self.init_bias
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (2 * num_nodes)
            )
            self.edge_mask = torch.randn(num_edges) * std + init_bias
        else:
            self.edge_mask = edge_mask

        self.edge_mask = self.edge_mask.to(graph.device)

    def clear_masks(self):
        r"""Clear the edge mask that play a crucial role to explain the
        prediction made by the GNN for a graph.
        """
        self.edge_mask = None

    def parameters(self):
        r"""
        Returns an iterator over the `Parameter` objects of the `nn.Linear`
        layers in the `self.elayers` sequential module. Each `Parameter`
        object contains the weight and bias parameters of an `nn.Linear`
        layer, as learned during training.

        Returns
        -------
        iterator
            An iterator over the `Parameter` objects of the `nn.Linear`
            layers in the `self.elayers` sequential module.
        """
        return self.elayers.parameters()

    def loss(self, prob, ori_pred):
        r"""The loss function that is used to learn the edge
        distribution.

        Parameters
        ----------
        prob: Tensor
            Tensor contains a set of probabilities for each possible
            class label of some model for all the batched graphs,
            which is of shape :math:`(B, L)`, where :math:`L` is the
            different types of label in the dataset and :math:`B` is
            the batch size.
        ori_pred: Tensor
            Tensor of shape :math:`(B, 1)`, representing the original prediction
            for the graph, where :math:`B` is the batch size.

        Returns
        -------
        float
            The function that returns the sum of the three loss components,
            which is a scalar tensor representing the total loss.
        """
        target_prob = prob.gather(-1, ori_pred.unsqueeze(-1))
        # 1e-6 added to prob to avoid taking the logarithm of zero
        target_prob += 1e-6
        # computing the log likelihood for a single prediction
        pred_loss = torch.mean(-torch.log(target_prob))

        # size
        edge_mask = self.sparse_mask_values
        if self.coff_budget <= 0:
            size_loss = self.coff_budget * torch.sum(edge_mask)
        else:
            size_loss = self.coff_budget * F.relu(
                torch.sum(edge_mask) - self.coff_budget
            )

        # entropy
        scale = 0.99
        edge_mask = self.edge_mask * (2 * scale - 1.0) + (1.0 - scale)
        mask_ent = -edge_mask * torch.log(edge_mask) - (
            1 - edge_mask
        ) * torch.log(1 - edge_mask)
        mask_ent_loss = self.coff_connect * torch.mean(mask_ent)

        loss = pred_loss + size_loss + mask_ent_loss
        return loss

    def concrete_sample(self, w, beta=1.0, training=True):
        r"""Sample from the instantiation of concrete distribution when training.

        Parameters
        ----------
        w : Tensor
            A tensor representing the log of the prior probability of choosing the edges.
        beta : float, optional
            Controls the degree of randomness in the output of the sigmoid function.
        training : bool, optional
            Randomness is injected during training.

        Returns
        -------
        Tensor
            If training is set to True, the output is a tensor of probabilities that
            represent the probability of activating the gate for each input element.
            If training is set to False, the output is also a tensor of probabilities,
            but they are determined solely by the log_alpha values, without adding any
            random noise.
        """
        if training:
            bias = self.sample_bias
            random_noise = torch.rand(w.size()).to(w.device)
            random_noise = bias + (1 - 2 * bias) * random_noise
            gate_inputs = torch.log(random_noise) - torch.log(
                1.0 - random_noise
            )
            gate_inputs = (gate_inputs + w) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(w)

        return gate_inputs

    def train_step(self, graph, feat, temperature, **kwargs):
        r"""Compute the loss of the explanation network for graph classification

        Parameters
        ----------
        graph : DGLGraph
            Input batched homogeneous graph.
        feat : Tensor
            The input feature of shape :math:`(N, D)`. :math:`N` is the
            number of nodes, and :math:`D` is the feature size.
        temperature : float
            The temperature parameter fed to the sampling procedure.
        kwargs : dict
            Additional arguments passed to the GNN model.

        Returns
        -------
        Tensor
            A scalar tensor representing the loss.
        """
        assert (
            self.graph_explanation
        ), '"explain_graph" must be True when initializing the module.'

        self.model = self.model.to(graph.device)
        self.elayers = self.elayers.to(graph.device)

        pred = self.model(graph, feat, embed=False, **kwargs)
        pred = pred.argmax(-1).data

        prob, _ = self.explain_graph(
            graph, feat, temperature, training=True, **kwargs
        )

        loss = self.loss(prob, pred)
        return loss

    def train_step_node(self, nodes, graph, feat, temperature, **kwargs):
        r"""Compute the loss of the explanation network for node classification

        Parameters
        ----------
        nodes : int, iterable[int], tensor
            The nodes from the graph used to train the explanation network,
            which cannot have any duplicate value.
        graph : DGLGraph
            Input homogeneous graph.
        feat : Tensor
            The input feature of shape :math:`(N, D)`. :math:`N` is the
            number of nodes, and :math:`D` is the feature size.
        temperature : float
            The temperature parameter fed to the sampling procedure.
        kwargs : dict
            Additional arguments passed to the GNN model.

        Returns
        -------
        Tensor
            A scalar tensor representing the loss.
        """
        assert (
            not self.graph_explanation
        ), '"explain_graph" must be False when initializing the module.'

        self.model = self.model.to(graph.device)
        self.elayers = self.elayers.to(graph.device)

        if isinstance(nodes, torch.Tensor):
            nodes = nodes.tolist()
        if isinstance(nodes, int):
            nodes = [nodes]

        prob, _, batched_graph, inverse_indices = self.explain_node(
            nodes, graph, feat, temperature, training=True, **kwargs
        )

        pred = self.model(
            batched_graph, self.batched_feats, embed=False, **kwargs
        )
        pred = pred.argmax(-1).data

        loss = self.loss(prob[inverse_indices], pred[inverse_indices])
        return loss

    def explain_graph(
        self, graph, feat, temperature=1.0, training=False, **kwargs
    ):
        r"""Learn and return an edge mask that plays a crucial role to
        explain the prediction made by the GNN for a graph. Also, return
        the prediction made with the edges chosen based on the edge mask.

        Parameters
        ----------
        graph : DGLGraph
            A homogeneous graph.
        feat : Tensor
            The input feature of shape :math:`(N, D)`. :math:`N` is the
            number of nodes, and :math:`D` is the feature size.
        temperature : float
            The temperature parameter fed to the sampling procedure.
        training : bool
            Training the explanation network.
        kwargs : dict
            Additional arguments passed to the GNN model.

        Returns
        -------
        Tensor
            Classification probabilities given the masked graph. It is a tensor
            of shape :math:`(B, L)`, where :math:`L` is the different types of
            label in the dataset, and :math:`B` is the batch size.
        Tensor
            Edge weights which is a tensor of shape :math:`(E)`, where :math:`E`
            is the number of edges in the graph. A higher weight suggests a
            larger contribution of the edge.

        Examples
        --------

        >>> import torch as th
        >>> import torch.nn as nn
        >>> import dgl
        >>> from dgl.data import GINDataset
        >>> from dgl.dataloading import GraphDataLoader
        >>> from dgl.nn import GraphConv, PGExplainer
        >>> import numpy as np

        >>> # Define the model
        >>> class Model(nn.Module):
        ...     def __init__(self, in_feats, out_feats):
        ...         super().__init__()
        ...         self.conv = GraphConv(in_feats, out_feats)
        ...         self.fc = nn.Linear(out_feats, out_feats)
        ...         nn.init.xavier_uniform_(self.fc.weight)
        ...
        ...     def forward(self, g, h, embed=False, edge_weight=None):
        ...         h = self.conv(g, h, edge_weight=edge_weight)
        ...
        ...         if embed:
        ...             return h
        ...
        ...         with g.local_scope():
        ...             g.ndata['h'] = h
        ...             hg = dgl.mean_nodes(g, 'h')
        ...             return self.fc(hg)

        >>> # Load dataset
        >>> data = GINDataset('MUTAG', self_loop=True)
        >>> dataloader = GraphDataLoader(data, batch_size=64, shuffle=True)

        >>> # Train the model
        >>> feat_size = data[0][0].ndata['attr'].shape[1]
        >>> model = Model(feat_size, data.gclasses)
        >>> criterion = nn.CrossEntropyLoss()
        >>> optimizer = th.optim.Adam(model.parameters(), lr=1e-2)
        >>> for bg, labels in dataloader:
        ...     preds = model(bg, bg.ndata['attr'])
        ...     loss = criterion(preds, labels)
        ...     optimizer.zero_grad()
        ...     loss.backward()
        ...     optimizer.step()

        >>> # Initialize the explainer
        >>> explainer = PGExplainer(model, data.gclasses)

        >>> # Train the explainer
        >>> # Define explainer temperature parameter
        >>> init_tmp, final_tmp = 5.0, 1.0
        >>> optimizer_exp = th.optim.Adam(explainer.parameters(), lr=0.01)
        >>> for epoch in range(20):
        ...     tmp = float(init_tmp * np.power(final_tmp / init_tmp, epoch / 20))
        ...     for bg, labels in dataloader:
        ...          loss = explainer.train_step(bg, bg.ndata['attr'], tmp)
        ...          optimizer_exp.zero_grad()
        ...          loss.backward()
        ...          optimizer_exp.step()

        >>> # Explain the prediction for graph 0
        >>> graph, l = data[0]
        >>> graph_feat = graph.ndata.pop("attr")
        >>> probs, edge_weight = explainer.explain_graph(graph, graph_feat)
        """
        assert (
            self.graph_explanation
        ), '"explain_graph" must be True when initializing the module.'

        self.model = self.model.to(graph.device)
        self.elayers = self.elayers.to(graph.device)

        embed = self.model(graph, feat, embed=True, **kwargs)
        embed = embed.data

        col, row = graph.edges()
        col_emb = embed[col.long()]
        row_emb = embed[row.long()]
        emb = torch.cat([col_emb, row_emb], dim=-1)
        emb = self.elayers(emb)
        values = emb.reshape(-1)

        values = self.concrete_sample(
            values, beta=temperature, training=training
        )
        self.sparse_mask_values = values

        reverse_eids = graph.edge_ids(row, col).long()
        edge_mask = (values + values[reverse_eids]) / 2

        self.set_masks(graph, edge_mask)

        # the model prediction with the updated edge mask
        logits = self.model(graph, feat, edge_weight=self.edge_mask, **kwargs)
        probs = F.softmax(logits, dim=-1)

        if training:
            probs = probs.data
        else:
            self.clear_masks()

        return (probs, edge_mask)

    def explain_node(
        self, nodes, graph, feat, temperature=1.0, training=False, **kwargs
    ):
        r"""Learn and return an edge mask that plays a crucial role to
        explain the prediction made by the GNN for provided set of node IDs.
        Also, return the prediction made with the graph and edge mask.

        Parameters
        ----------
        nodes : int, iterable[int], tensor
            The nodes from the graph, which cannot have any duplicate value.
        graph : DGLGraph
            A homogeneous graph.
        feat : Tensor
            The input feature of shape :math:`(N, D)`. :math:`N` is the
            number of nodes, and :math:`D` is the feature size.
        temperature : float
            The temperature parameter fed to the sampling procedure.
        training : bool
            Training the explanation network.
        kwargs : dict
            Additional arguments passed to the GNN model.

        Returns
        -------
        Tensor
            Classification probabilities given the masked graph. It is a tensor
            of shape :math:`(N, L)`, where :math:`L` is the different types
            of node labels in the dataset, and :math:`N` is the number of nodes
            in the graph.
        Tensor
            Edge weights which is a tensor of shape :math:`(E)`, where :math:`E`
            is the number of edges in the graph. A higher weight suggests a
            larger contribution of the edge.
        DGLGraph
            The batched set of subgraphs induced on the k-hop in-neighborhood
            of the input center nodes.
        Tensor
            The new IDs of the subgraph center nodes.

        Examples
        --------

        >>> import dgl
        >>> import numpy as np
        >>> import torch

        >>> # Define the model
        >>> class Model(torch.nn.Module):
        ...     def __init__(self, in_feats, out_feats):
        ...         super().__init__()
        ...         self.conv1 = dgl.nn.GraphConv(in_feats, out_feats)
        ...         self.conv2 = dgl.nn.GraphConv(out_feats, out_feats)
        ...
        ...     def forward(self, g, h, embed=False, edge_weight=None):
        ...         h = self.conv1(g, h, edge_weight=edge_weight)
        ...         if embed:
        ...             return h
        ...         return self.conv2(g, h)

        >>> # Load dataset
        >>> data = dgl.data.CoraGraphDataset(verbose=False)
        >>> g = data[0]
        >>> features = g.ndata["feat"]
        >>> labels = g.ndata["label"]

        >>> # Train the model
        >>> model = Model(features.shape[1], data.num_classes)
        >>> criterion = torch.nn.CrossEntropyLoss()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        >>> for epoch in range(20):
        ...     logits = model(g, features)
        ...     loss = criterion(logits, labels)
        ...     optimizer.zero_grad()
        ...     loss.backward()
        ...     optimizer.step()

        >>> # Initialize the explainer
        >>> explainer = dgl.nn.PGExplainer(
        ...     model, data.num_classes, num_hops=2, explain_graph=False
        ... )

        >>> # Train the explainer
        >>> # Define explainer temperature parameter
        >>> init_tmp, final_tmp = 5.0, 1.0
        >>> optimizer_exp = torch.optim.Adam(explainer.parameters(), lr=0.01)
        >>> epochs = 10
        >>> for epoch in range(epochs):
        ...     tmp = float(init_tmp * np.power(final_tmp / init_tmp, epoch / epochs))
        ...     loss = explainer.train_step_node(g.nodes(), g, features, tmp)
        ...     optimizer_exp.zero_grad()
        ...     loss.backward()
        ...     optimizer_exp.step()

        >>> # Explain the prediction for graph 0
        >>> probs, edge_weight, bg, inverse_indices = explainer.explain_node(
        ...     0, g, features
        ... )
        """
        assert (
            not self.graph_explanation
        ), '"explain_graph" must be False when initializing the module.'
        assert (
            self.num_hops is not None
        ), '"num_hops" must be provided when initializing the module.'

        if isinstance(nodes, torch.Tensor):
            nodes = nodes.tolist()
        if isinstance(nodes, int):
            nodes = [nodes]

        self.model = self.model.to(graph.device)
        self.elayers = self.elayers.to(graph.device)

        batched_graph = []
        batched_embed = []
        for node_id in nodes:
            sg, inverse_indices = khop_in_subgraph(
                graph, node_id, self.num_hops
            )
            sg.ndata["feat"] = feat[sg.ndata[NID].long()]
            sg.ndata["train"] = torch.tensor(
                [nid in inverse_indices for nid in sg.nodes()], device=sg.device
            )

            embed = self.model(sg, sg.ndata["feat"], embed=True, **kwargs)
            embed = embed.data

            col, row = sg.edges()
            col_emb = embed[col.long()]
            row_emb = embed[row.long()]
            self_emb = embed[inverse_indices[0]].repeat(sg.num_edges(), 1)
            emb = torch.cat([col_emb, row_emb, self_emb], dim=-1)
            batched_embed.append(emb)
            batched_graph.append(sg)

        batched_graph = batch(batched_graph)

        batched_embed = torch.cat(batched_embed)
        batched_embed = self.elayers(batched_embed)
        values = batched_embed.reshape(-1)

        values = self.concrete_sample(
            values, beta=temperature, training=training
        )
        self.sparse_mask_values = values

        col, row = batched_graph.edges()
        reverse_eids = batched_graph.edge_ids(row, col).long()
        edge_mask = (values + values[reverse_eids]) / 2

        self.set_masks(batched_graph, edge_mask)

        batched_feats = batched_graph.ndata["feat"]
        # the model prediction with the updated edge mask
        logits = self.model(
            batched_graph, batched_feats, edge_weight=self.edge_mask, **kwargs
        )
        probs = F.softmax(logits, dim=-1)

        batched_inverse_indices = (
            batched_graph.ndata["train"].nonzero().squeeze(1)
        )

        if training:
            self.batched_feats = batched_feats
            probs = probs.data
        else:
            self.clear_masks()

        return (
            probs,
            edge_mask,
            batched_graph,
            batched_inverse_indices,
        )


class HeteroPGExplainer(PGExplainer):
    r"""PGExplainer from `Parameterized Explainer for Graph Neural Network
    <https://arxiv.org/pdf/2011.04573>`__, adapted for heterogeneous graphs

    PGExplainer adopts a deep neural network (explanation network) to
    parameterize the generation process of explanations, which enables it to
    explain multiple instances collectively. PGExplainer models the underlying
    structure as edge distributions, from which the explanatory graph is
    sampled.

    Parameters
    ----------
    model : nn.Module
        The GNN model to explain that tackles multiclass graph classification

        * Its forward function must have the form
          :attr:`forward(self, graph, nfeat, embed, edge_weight)`.
        * The output of its forward function is the logits if embed=False else
          the intermediate node embeddings.
    num_features : int
        Node embedding size used by :attr:`model`.
    coff_budget : float, optional
        Size regularization to constrain the explanation size. Default: 0.01.
    coff_connect : float, optional
        Entropy regularization to constrain the connectivity of explanation. Default: 5e-4.
    sample_bias : float, optional
        Some members of a population are systematically more likely to be selected
        in a sample than others. Default: 0.0.
    """

    def train_step(self, graph, feat, temperature, **kwargs):
        # pylint: disable=useless-super-delegation
        r"""Compute the loss of the explanation network for graph classification

        Parameters
        ----------
        graph : DGLGraph
            Input batched heterogeneous graph.
        feat : dict[str, Tensor]
            A dict mapping node types (keys) to feature tensors (values).
            The input features are of shape :math:`(N_t, D_t)`. :math:`N_t` is
            the number of nodes for node type :math:`t`, and :math:`D_t` is the
            feature size for node type :math:`t`
        temperature : float
            The temperature parameter fed to the sampling procedure.
        kwargs : dict
            Additional arguments passed to the GNN model.

        Returns
        -------
        Tensor
            A scalar tensor representing the loss.
        """
        return super().train_step(graph, feat, temperature, **kwargs)

    def train_step_node(self, nodes, graph, feat, temperature, **kwargs):
        r"""Compute the loss of the explanation network for node classification

        Parameters
        ----------
        nodes : dict[str, Iterable[int]]
            A dict mapping node types (keys) to an iterable set of node ids (values).
        graph : DGLGraph
            Input heterogeneous graph.
        feat : dict[str, Tensor]
            A dict mapping node types (keys) to feature tensors (values).
            The input features are of shape :math:`(N_t, D_t)`. :math:`N_t` is
            the number of nodes for node type :math:`t`, and :math:`D_t` is the
            feature size for node type :math:`t`
        temperature : float
            The temperature parameter fed to the sampling procedure.
        kwargs : dict
            Additional arguments passed to the GNN model.

        Returns
        -------
        Tensor
            A scalar tensor representing the loss.
        """
        assert (
            not self.graph_explanation
        ), '"explain_graph" must be False when initializing the module.'

        self.model = self.model.to(graph.device)
        self.elayers = self.elayers.to(graph.device)

        prob, _, batched_graph, inverse_indices = self.explain_node(
            nodes, graph, feat, temperature, training=True, **kwargs
        )

        pred = self.model(
            batched_graph, self.batched_feats, embed=False, **kwargs
        )
        pred = {ntype: pred[ntype].argmax(-1).data for ntype in pred.keys()}

        loss = self.loss(
            torch.cat(
                [prob[ntype][nid] for ntype, nid in inverse_indices.items()]
            ),
            torch.cat(
                [pred[ntype][nid] for ntype, nid in inverse_indices.items()]
            ),
        )
        return loss

    def explain_graph(
        self, graph, feat, temperature=1.0, training=False, **kwargs
    ):
        r"""Learn and return an edge mask that plays a crucial role to
        explain the prediction made by the GNN for a graph. Also, return
        the prediction made with the edges chosen based on the edge mask.

        Parameters
        ----------
        graph : DGLGraph
            A heterogeneous graph.
        feat : dict[str, Tensor]
            A dict mapping node types (keys) to feature tensors (values).
            The input features are of shape :math:`(N_t, D_t)`. :math:`N_t` is
            the number of nodes for node type :math:`t`, and :math:`D_t` is the
            feature size for node type :math:`t`
        temperature : float
            The temperature parameter fed to the sampling procedure.
        training : bool
            Training the explanation network.
        kwargs : dict
            Additional arguments passed to the GNN model.

        Returns
        -------
        Tensor
            Classification probabilities given the masked graph. It is a tensor
            of shape :math:`(B, L)`, where :math:`L` is the different types of
            label in the dataset, and :math:`B` is the batch size.
        dict[str, Tensor]
            A dict mapping edge types (keys) to edge tensors (values) of shape
            :math:`(E_t)`, where :math:`E_t` is the number of edges in the graph
            for edge type :math:`t`.  A higher weight suggests a larger
            contribution of the edge.

        Examples
        --------

        >>> import dgl
        >>> import torch as th
        >>> import torch.nn as nn
        >>> import numpy as np

        >>> # Define the model
        >>> class Model(nn.Module):
        ...     def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        ...         super().__init__()
        ...         self.conv = dgl.nn.HeteroGraphConv(
        ...             {rel: dgl.nn.GraphConv(in_feats, hid_feats) for rel in rel_names},
        ...             aggregate="sum",
        ...         )
        ...         self.fc = nn.Linear(hid_feats, out_feats)
        ...         nn.init.xavier_uniform_(self.fc.weight)
        ...
        ...     def forward(self, g, h, embed=False, edge_weight=None):
        ...         if edge_weight:
        ...             mod_kwargs = {
        ...                 etype: {"edge_weight": mask} for etype, mask in edge_weight.items()
        ...             }
        ...             h = self.conv(g, h, mod_kwargs=mod_kwargs)
        ...         else:
        ...             h = self.conv(g, h)
        ...
        ...         if embed:
        ...             return h
        ...
        ...         with g.local_scope():
        ...             g.ndata["h"] = h
        ...             hg = 0
        ...             for ntype in g.ntypes:
        ...                 hg = hg + dgl.mean_nodes(g, "h", ntype=ntype)
        ...             return self.fc(hg)

        >>> # Load dataset
        >>> input_dim = 5
        >>> hidden_dim = 5
        >>> num_classes = 2
        >>> g = dgl.heterograph({("user", "plays", "game"): ([0, 1, 1, 2], [0, 0, 1, 1])})
        >>> g.nodes["user"].data["h"] = th.randn(g.num_nodes("user"), input_dim)
        >>> g.nodes["game"].data["h"] = th.randn(g.num_nodes("game"), input_dim)

        >>> transform = dgl.transforms.AddReverse()
        >>> g = transform(g)

        >>> # define and train the model
        >>> model = Model(input_dim, hidden_dim, num_classes, g.canonical_etypes)
        >>> optimizer = th.optim.Adam(model.parameters())
        >>> for epoch in range(10):
        ...     logits = model(g, g.ndata["h"])
        ...     loss = th.nn.functional.cross_entropy(logits, th.tensor([1]))
        ...     optimizer.zero_grad()
        ...     loss.backward()
        ...     optimizer.step()

        >>> # Initialize the explainer
        >>> explainer = dgl.nn.HeteroPGExplainer(model, hidden_dim)

        >>> # Train the explainer
        >>> # Define explainer temperature parameter
        >>> init_tmp, final_tmp = 5.0, 1.0
        >>> optimizer_exp = th.optim.Adam(explainer.parameters(), lr=0.01)
        >>> for epoch in range(20):
        ...     tmp = float(init_tmp * np.power(final_tmp / init_tmp, epoch / 20))
        ...     loss = explainer.train_step(g, g.ndata["h"], tmp)
        ...     optimizer_exp.zero_grad()
        ...     loss.backward()
        ...     optimizer_exp.step()

        >>> # Explain the graph
        >>> feat = g.ndata.pop("h")
        >>> probs, edge_mask = explainer.explain_graph(g, feat)
        """
        assert (
            self.graph_explanation
        ), '"explain_graph" must be True when initializing the module.'

        self.model = self.model.to(graph.device)
        self.elayers = self.elayers.to(graph.device)

        embed = self.model(graph, feat, embed=True, **kwargs)
        for ntype, emb in embed.items():
            graph.nodes[ntype].data["emb"] = emb.data
        homo_graph = to_homogeneous(graph, ndata=["emb"])
        homo_embed = homo_graph.ndata["emb"]

        col, row = homo_graph.edges()
        col_emb = homo_embed[col.long()]
        row_emb = homo_embed[row.long()]
        emb = torch.cat([col_emb, row_emb], dim=-1)
        emb = self.elayers(emb)
        values = emb.reshape(-1)

        values = self.concrete_sample(
            values, beta=temperature, training=training
        )
        self.sparse_mask_values = values

        reverse_eids = homo_graph.edge_ids(row, col).long()
        edge_mask = (values + values[reverse_eids]) / 2

        self.set_masks(homo_graph, edge_mask)

        # convert the edge mask back into heterogeneous format
        hetero_edge_mask = self._edge_mask_to_heterogeneous(
            edge_mask=edge_mask,
            homograph=homo_graph,
            heterograph=graph,
        )

        # the model prediction with the updated edge mask
        logits = self.model(graph, feat, edge_weight=hetero_edge_mask, **kwargs)
        probs = F.softmax(logits, dim=-1)

        if training:
            probs = probs.data
        else:
            self.clear_masks()

        return (probs, hetero_edge_mask)

    def explain_node(
        self, nodes, graph, feat, temperature=1.0, training=False, **kwargs
    ):
        r"""Learn and return an edge mask that plays a crucial role to
        explain the prediction made by the GNN for provided set of node IDs.
        Also, return the prediction made with the batched graph and edge mask.

        Parameters
        ----------
        nodes : dict[str, Iterable[int]]
            A dict mapping node types (keys) to an iterable set of node ids (values).
        graph : DGLGraph
            A heterogeneous graph.
        feat : dict[str, Tensor]
            A dict mapping node types (keys) to feature tensors (values).
            The input features are of shape :math:`(N_t, D_t)`. :math:`N_t` is
            the number of nodes for node type :math:`t`, and :math:`D_t` is the
            feature size for node type :math:`t`
        temperature : float
            The temperature parameter fed to the sampling procedure.
        training : bool
            Training the explanation network.
        kwargs : dict
            Additional arguments passed to the GNN model.

        Returns
        -------
        dict[str, Tensor]
            A dict mapping node types (keys) to classification probabilities
            for node labels (values). The values are tensors of shape
            :math:`(N_t, L)`, where :math:`L` is the different types of node
            labels in the dataset, and :math:`N_t` is the number of nodes in
            the graph for node type :math:`t`.
        dict[str, Tensor]
            A dict mapping edge types (keys) to edge tensors (values) of shape
            :math:`(E_t)`, where :math:`E_t` is the number of edges in the graph
            for edge type :math:`t`.  A higher weight suggests a larger
            contribution of the edge.
        DGLGraph
            The batched set of subgraphs induced on the k-hop in-neighborhood
            of the input center nodes.
        dict[str, Tensor]
            A dict mapping node types (keys) to a tensor of node IDs (values)
            which correspond to the subgraph center nodes.

        Examples
        --------

        >>> import dgl
        >>> import torch as th
        >>> import torch.nn as nn
        >>> import numpy as np

        >>> # Define the model
        >>> class Model(nn.Module):
        ...     def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        ...         super().__init__()
        ...         self.conv = dgl.nn.HeteroGraphConv(
        ...             {rel: dgl.nn.GraphConv(in_feats, hid_feats) for rel in rel_names},
        ...             aggregate="sum",
        ...         )
        ...         self.fc = nn.Linear(hid_feats, out_feats)
        ...         nn.init.xavier_uniform_(self.fc.weight)
        ...
        ...     def forward(self, g, h, embed=False, edge_weight=None):
        ...         if edge_weight:
        ...             mod_kwargs = {
        ...                 etype: {"edge_weight": mask} for etype, mask in edge_weight.items()
        ...             }
        ...             h = self.conv(g, h, mod_kwargs=mod_kwargs)
        ...         else:
        ...             h = self.conv(g, h)
        ...
        ...         return h

        >>> # Load dataset
        >>> input_dim = 5
        >>> hidden_dim = 5
        >>> num_classes = 2
        >>> g = dgl.heterograph({("user", "plays", "game"): ([0, 1, 1, 2], [0, 0, 1, 1])})
        >>> g.nodes["user"].data["h"] = th.randn(g.num_nodes("user"), input_dim)
        >>> g.nodes["game"].data["h"] = th.randn(g.num_nodes("game"), input_dim)

        >>> transform = dgl.transforms.AddReverse()
        >>> g = transform(g)

        >>> # define and train the model
        >>> model = Model(input_dim, hidden_dim, num_classes, g.canonical_etypes)
        >>> optimizer = th.optim.Adam(model.parameters())
        >>> for epoch in range(10):
        ...     logits = model(g, g.ndata["h"])['user']
        ...     loss = th.nn.functional.cross_entropy(logits, th.tensor([1,1,1]))
        ...     optimizer.zero_grad()
        ...     loss.backward()
        ...     optimizer.step()

        >>> # Initialize the explainer
        >>> explainer = dgl.nn.HeteroPGExplainer(
        ...     model, hidden_dim, num_hops=2, explain_graph=False
        ... )

        >>> # Train the explainer
        >>> # Define explainer temperature parameter
        >>> init_tmp, final_tmp = 5.0, 1.0
        >>> optimizer_exp = th.optim.Adam(explainer.parameters(), lr=0.01)
        >>> for epoch in range(20):
        ...     tmp = float(init_tmp * np.power(final_tmp / init_tmp, epoch / 20))
        ...     loss = explainer.train_step_node(
        ...         { ntype: g.nodes(ntype) for ntype in g.ntypes },
        ...         g, g.ndata["h"], tmp
        ...     )
        ...     optimizer_exp.zero_grad()
        ...     loss.backward()
        ...     optimizer_exp.step()

        >>> # Explain the graph
        >>> feat = g.ndata.pop("h")
        >>> probs, edge_mask, bg, inverse_indices = explainer.explain_node(
        ...     { "user": [0] }, g, feat
        ... )
        """
        assert (
            not self.graph_explanation
        ), '"explain_graph" must be False when initializing the module.'
        assert (
            self.num_hops is not None
        ), '"num_hops" must be provided when initializing the module.'

        self.model = self.model.to(graph.device)
        self.elayers = self.elayers.to(graph.device)

        batched_embed = []
        batched_homo_graph = []
        batched_hetero_graph = []
        for target_ntype, target_nids in nodes.items():
            if isinstance(target_nids, torch.Tensor):
                target_nids = target_nids.tolist()

            for target_nid in target_nids:
                sg, inverse_indices = khop_in_subgraph(
                    graph, {target_ntype: target_nid}, self.num_hops
                )

                for sg_ntype in sg.ntypes:
                    sg_feat = feat[sg_ntype][sg.ndata[NID][sg_ntype].long()]
                    train_mask = [
                        sg_ntype in inverse_indices
                        and node_id in inverse_indices[sg_ntype]
                        for node_id in sg.nodes(sg_ntype)
                    ]

                    sg.nodes[sg_ntype].data["feat"] = sg_feat
                    sg.nodes[sg_ntype].data["train"] = torch.tensor(
                        train_mask, device=sg.device
                    )

                embed = self.model(sg, sg.ndata["feat"], embed=True, **kwargs)
                for ntype in embed.keys():
                    sg.nodes[ntype].data["emb"] = embed[ntype].data

                homo_sg = to_homogeneous(sg, ndata=["emb"])
                homo_sg_embed = homo_sg.ndata["emb"]

                col, row = homo_sg.edges()
                col_emb = homo_sg_embed[col.long()]
                row_emb = homo_sg_embed[row.long()]
                self_emb = homo_sg_embed[
                    inverse_indices[target_ntype][0]
                ].repeat(sg.num_edges(), 1)
                emb = torch.cat([col_emb, row_emb, self_emb], dim=-1)
                batched_embed.append(emb)
                batched_homo_graph.append(homo_sg)
                batched_hetero_graph.append(sg)

        batched_homo_graph = batch(batched_homo_graph)
        batched_hetero_graph = batch(batched_hetero_graph)

        batched_embed = torch.cat(batched_embed)
        batched_embed = self.elayers(batched_embed)
        values = batched_embed.reshape(-1)

        values = self.concrete_sample(
            values, beta=temperature, training=training
        )
        self.sparse_mask_values = values

        col, row = batched_homo_graph.edges()
        reverse_eids = batched_homo_graph.edge_ids(row, col).long()
        edge_mask = (values + values[reverse_eids]) / 2

        self.set_masks(batched_homo_graph, edge_mask)

        # Convert the edge mask back into heterogeneous format.
        hetero_edge_mask = self._edge_mask_to_heterogeneous(
            edge_mask=edge_mask,
            homograph=batched_homo_graph,
            heterograph=batched_hetero_graph,
        )

        batched_feats = {
            ntype: batched_hetero_graph.nodes[ntype].data["feat"]
            for ntype in batched_hetero_graph.ntypes
        }

        # The model prediction with the updated edge mask.
        logits = self.model(
            batched_hetero_graph,
            batched_feats,
            edge_weight=hetero_edge_mask,
            **kwargs,
        )
        probs = {
            ntype: F.softmax(logits[ntype], dim=-1) for ntype in logits.keys()
        }

        batched_inverse_indices = {
            ntype: batched_hetero_graph.nodes[ntype]
            .data["train"]
            .nonzero()
            .squeeze(1)
            for ntype in batched_hetero_graph.ntypes
        }

        if training:
            self.batched_feats = batched_feats
            probs = {ntype: probs[ntype].data for ntype in probs.keys()}
        else:
            self.clear_masks()

        return (
            probs,
            hetero_edge_mask,
            batched_hetero_graph,
            batched_inverse_indices,
        )

    def _edge_mask_to_heterogeneous(self, edge_mask, homograph, heterograph):
        r"""Convert an edge mask from homogeneous mappings built through
        embeddings into heterogenous format by leveraging the context from
        the source DGLGraphs in homogenous and heterogeneous form.

        The `edge_mask` needs to have been built using the embedding of the
        homogenous graph format for the mappings to work correctly.

        Parameters
        ----------
        edge_mask : dict[str, Tensor]
            A dict mapping node types (keys) to a tensor of edge weights (values).
        homograph : DGLGraph
            The homogeneous form of the source graph.
        heterograph : DGLGraph
            The heterogeneous form of the source graph.

        Returns
        -------
        dict[str, Tensor]
            A dict mapping node types (keys) to tensors of node ids (values)
        """
        return {
            etype: edge_mask[
                (homograph.edata[ETYPE] == heterograph.get_etype_id(etype))
                .nonzero()
                .squeeze(1)
            ]
            for etype in heterograph.canonical_etypes
        }
