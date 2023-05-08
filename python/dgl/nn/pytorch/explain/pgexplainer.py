"""Torch Module for PGExplainer"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

__all__ = ["PGExplainer"]


class PGExplainer(nn.Module):
    r"""PGExplainer from `Parameterized Explainer for Graph Neural Network
    <https://arxiv.org/pdf/2011.04573>`

    PGExplainer adopts a deep neural network (explanation network) to parameterize the generation
    process of explanations, which enables it to explain multiple instances
    collectively. PGExplainer models the underlying structure as edge
    distributions, from which the explanatory graph is sampled.

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

    def __init__(
        self,
        model,
        num_features,
        coff_budget=0.01,
        coff_connect=5e-4,
        sample_bias=0.0,
    ):
        super(PGExplainer, self).__init__()

        self.model = model
        self.num_features = num_features * 2

        # training hyperparameters for PGExplainer
        self.coff_budget = coff_budget
        self.coff_connect = coff_connect
        self.sample_bias = sample_bias

        self.init_bias = 0.0

        # Explanation network in PGExplainer
        self.elayers = nn.Sequential(
            nn.Linear(self.num_features, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def set_masks(self, graph, feat, edge_mask=None):
        r"""Set the edge mask that plays a crucial role to explain the
        prediction made by the GNN for a graph. Initialize learnable edge
        mask if it is None.

        Parameters
        ----------
        graph : DGLGraph
            A homogeneous graph.
        feat : Tensor
            The input feature of shape :math:`(N, D)`. :math:`N` is the
            number of nodes, and :math:`D` is the feature size.
        edge_mask : Tensor, optional
            Learned importance mask of the edges in the graph, which is a tensor
            of shape :math:`(E)`, where :math:`E` is the number of edges in the
            graph. The values are within range :math:`(0, 1)`. The higher,
            the more important. Default: None.
        """
        num_nodes, _ = feat.shape
        num_edges = graph.num_edges()

        init_bias = self.init_bias
        std = nn.init.calculate_gain("relu") * math.sqrt(2.0 / (2 * num_nodes))

        if edge_mask is None:
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
            which is of shape :math:`(1, L)`, where :math:`L` is the
            different types of label in the dataset.
        ori_pred: Tensor
            Tensor of shape ::math:`(1, 1)`, representing the original prediction
            for the graph.

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

    def train_step(self, graph, feat, tmp, **kwargs):
        r"""Training the explanation network by gradient descent(GD)
        using Adam optimizer

        Parameters
        ----------
        graph : DGLGraph
            A homogeneous graph.
        feat : Tensor
            The input feature of shape :math:`(N, D)`. :math:`N` is the
            number of nodes, and :math:`D` is the feature size.
        tmp : float
            The temperature parameter fed to the sampling procedure.
        kwargs : dict
            Additional arguments passed to the GNN model.

        Returns
        -------
        Tensor
            A scalar tensor representing the loss.
        """
        self.model = self.model.to(graph.device)
        self.elayers = self.elayers.to(graph.device)

        pred = self.model(graph, feat, embed=False, **kwargs).argmax(-1).data

        prob, edge_mask = self.explain_graph(
            graph, feat, tmp=tmp, training=True, **kwargs
        )

        self.edge_mask = edge_mask

        loss_tmp = self.loss(prob, pred)
        return loss_tmp

    def explain_graph(self, graph, feat, tmp=1.0, training=False, **kwargs):
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
        tmp : float
            The temperature parameter fed to the sampling procedure.
        training : bool
            Training the explanation network.
        kwargs : dict
            Additional arguments passed to the GNN model.

        Returns
        -------
        Tensor
            Classification probabilities given the masked graph. It is a tensor of
            shape :math:`(1, L)`, where :math:`L` is the different types of label
            in the dataset.
        Tensor
            Edge weights which is a tensor of shape :math:`(E)`, where :math:`E`
            is the number of edges in the graph. A higher weight suggests a larger
            contribution of the edge.

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
        ...         if not embed:
        ...             g.ndata['h'] = h
        ...             hg = dgl.mean_nodes(g, 'h')
        ...             return self.fc(hg)
        ...         else:
        ...             return h

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
        self.model = self.model.to(graph.device)
        self.elayers = self.elayers.to(graph.device)

        embed = self.model(graph, feat, embed=True, **kwargs)
        embed = embed.data

        edge_idx = graph.edges()

        node_size = embed.shape[0]
        row, col = edge_idx
        col_emb = embed[col.long()]
        row_emb = embed[row.long()]
        emb = torch.cat([col_emb, row_emb], dim=-1)

        emb = self.elayers(emb)
        values = emb.reshape(-1)

        values = self.concrete_sample(values, beta=tmp, training=training)
        self.sparse_mask_values = values

        mask_sparse = torch.sparse_coo_tensor(
            [edge_idx[0].tolist(), edge_idx[1].tolist()],
            values.tolist(),
            (node_size, node_size),
        )
        mask_sigmoid = mask_sparse.to_dense()
        # set the symmetric edge weights
        reverse_eids = graph.edge_ids(row, col).long()
        edge_mask = (values + values[reverse_eids]) / 2

        # clear the weights and set the edge mask
        self.clear_masks()
        self.set_masks(graph, feat, edge_mask)

        # the model prediction with the updated edge mask
        logits = self.model(graph, feat, edge_weight=self.edge_mask, **kwargs)
        probs = F.softmax(logits, dim=-1)

        self.clear_masks()

        return probs.data, edge_mask
