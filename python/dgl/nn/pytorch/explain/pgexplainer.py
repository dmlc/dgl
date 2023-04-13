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

     PGExplainer adopts a deep neural network to parameterize the generation
     process of explanations, which enables it to explain multiple instances
     collectively. PGExplainer models the underlying structure as edge
     distributions, where the explanatory graph is sampled from.

     Parameters
    ----------
    model : nn.Module
        The GNN model to explain that tackles multiclass graph classification

        * Its forward function must have the form
          :attr:`forward(self, graph, nfeat, embed, edge_weight)`.
        * The output of its forward function is the logits if embed=False else
        return the graph embedding.
    num_features : int
        Number of input feature for the explanation network.
    epochs : int, optional
        Number of epochs to train the explanation network. Default: 10.
    lr : float, optional
        Learning rate to train the explanation network. Default: 0.01.
    coff_size : float, optional
        Size regularization to constrain the explanation size. Default: 0.01.
    coff_ent : float, optional
        Entropy regularization to constrain the connectivity of explanation. Default: 5e-4.
    t0 : float, optional
        The temperature at the first epoch. Default: 5.0.
    t1 : float, optional
        The temperature at the final epoch. Default: 1.0.
    sample_bias : float, optional
        Some members of a population are systematically more likely to be selected
        in a sample than others. Default: 0.0.
    """

    def __init__(
            self,
            model,
            num_features,
            epochs=10,
            lr=0.01,
            coff_size=0.01,
            coff_ent=5e-4,
            t0=5.0,
            t1=1.0,
            sample_bias=0.0,
    ):
        super(PGExplainer, self).__init__()

        self.model = model
        self.num_features = num_features * 2

        # training parameters for PGExplainer
        self.epochs = epochs
        self.lr = lr
        self.coff_size = coff_size
        self.coff_ent = coff_ent
        self.t0 = t0
        self.t1 = t1
        self.sample_bias = sample_bias

        self.init_bias = 0.0

        # Explanation model in PGExplainer
        self.elayers = nn.ModuleList()
        self.elayers.append(
            nn.Sequential(nn.Linear(self.num_features, 64), nn.ReLU())
        )
        self.elayers.append(nn.Linear(64, 1))

    def set_masks(self, graph, feat, edge_mask=None):
        r"""Set the edge mask that play a crucial role to explain the
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
        N, F = feat.shape
        E = graph.num_edges()

        init_bias = self.init_bias
        std = nn.init.calculate_gain("relu") * math.sqrt(2.0 / (2 * N))

        if edge_mask is None:
            self.edge_mask = torch.randn(E) * std + init_bias
        else:
            self.edge_mask = edge_mask

        self.edge_mask = self.edge_mask.to(graph.device)

    def clear_masks(self):
        r"""Clear the edge mask that play a crucial role to explain the
        prediction made by the GNN for a graph.
        """
        self.edge_mask = None

    def loss(self, prob, ori_pred):
        r"""The loss function that is used to learn the edge
        distribution.

        Parameters
        ----------
        prob:  Tensor
            Tensor contains a set of probabilities for each possible
            class label of some model.
        ori_pred: int
            Integer representing the original prediction.

        Returns
        -------
        float
            The function returns the sum of the three loss components,
            which is a scalar tensor representing the total loss.
        """
        logit = prob[ori_pred]
        # 1e-6 added to logit to avoid taking the logarithm of zero
        logit += 1e-6
        # computing the cross-entropy loss for a single prediction
        pred_loss = -torch.log(logit)

        # size
        edge_mask = self.sparse_mask_values
        if self.coff_size <= 0:
            size_loss = self.coff_size * torch.sum(edge_mask)
        else:
            size_loss = self.coff_size * F.relu(
                torch.sum(edge_mask) - self.coff_size
            )

        # entropy
        scale = 0.99
        edge_mask = self.edge_mask * (2 * scale - 1.0) + (1.0 - scale)
        mask_ent = -edge_mask * torch.log(edge_mask) - (
                1 - edge_mask
        ) * torch.log(1 - edge_mask)
        mask_ent_loss = self.coff_ent * torch.mean(mask_ent)

        loss = pred_loss + size_loss + mask_ent_loss

        return loss

    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        r"""Sample from the instantiation of concrete distribution when training.

        Parameters
        ----------
        log_alpha : Tensor
            A tensor representing the log of the prior probability of activating the gate.
        beta : float, optional
            Controls the degree of randomness in the gate's output.
        training : bool, optional
            Indicates whether the gate is being used during training or evaluation.

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
            random_noise = torch.rand(log_alpha.size()).to(log_alpha.device)
            random_noise = bias + (1 - 2 * bias) * random_noise
            gate_inputs = torch.log(random_noise) - torch.log(
                1.0 - random_noise
            )
            gate_inputs = (gate_inputs + log_alpha) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(log_alpha)

        return gate_inputs

    def train_explanation_network(self, dataset, func_extract_feat):
        r""" Training the explanation network by gradient descent(GD)
         using Adam optimizer

        Parameters
        ----------
        dataset : dgl.data
            The dataset to train the importance edge mask.
        func_extract_feat : func
            A function that extracts the node embeddings for each individual graphs.
        """
        optimizer = Adam(self.elayers.parameters(), lr=self.lr)

        ori_pred_dict = {}

        with torch.no_grad():
            for idx, (g, l) in enumerate(dataset):
                feat = func_extract_feat(g)

                self.model.eval()

                logits = self.model(g, feat)
                ori_pred_dict[idx] = logits.argmax(-1).data.cpu()

        # train the mask generator
        for epoch in range(self.epochs):
            loss = 0.0
            pred_list = []

            tmp = float(
                self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs)
            )

            self.elayers.train()
            optimizer.zero_grad()

            for idx, (g, l) in enumerate(dataset):
                feat = func_extract_feat(g)

                prob, edge_mask = self.explain_graph(
                    g, feat, tmp=tmp, training=True
                )

                self.edge_mask = edge_mask

                loss_tmp = self.loss(prob.unsqueeze(dim=0), ori_pred_dict[idx])
                loss_tmp.backward()

                loss += loss_tmp.item()
                pred_label = prob.argmax(-1).item()
                pred_list.append(pred_label)

            optimizer.step()
            print(f"Epoch: {epoch} | Loss: {loss}")

    def explain_graph(self, graph, feat, tmp=1.0, training=False):
        r"""Learn and return an an edge mask that play a crucial role to
        explain the prediction made by the GNN for a graph. Also, return
        the prediction made with just the edge mask.

        Parameters
        ----------
        graph : DGLGraph
            A homogeneous graph.
        feat : Tensor
            The input feature of shape :math:`(N, D)`. :math:`N` is the
            number of nodes, and :math:`D` is the feature size.
        tmp : float
            The temperature parameter fed to the sample procedure
        training : bool
            Indicates whether the concrete_sample is called during
            training or evaluation.

        Returns
        -------
        Tensor, Tensor
            The classification probability for graph with edge mask,
            the probability mask for graph edges.

        """
        embed = self.model(graph, feat, embed=True)
        embed = embed.data.cpu()

        edge_idx = graph.edges()

        node_size = embed.shape[0]
        col, row = edge_idx
        f1 = embed[col.long()]
        f2 = embed[row.long()]
        f12self = torch.cat([f1, f2], dim=-1)

        # using the node embedding to calculate the edge weight
        h = f12self.to(graph.device)
        for elayer in self.elayers:
            h = elayer(h)
        values = h.reshape(-1)

        values = self.concrete_sample(values, beta=tmp, training=training)
        self.sparse_mask_values = values

        mask_sparse = torch.sparse_coo_tensor(
            [edge_idx[0].tolist(), edge_idx[1].tolist()],
            values.tolist(),
            (node_size, node_size),
        )
        mask_sigmoid = mask_sparse.to_dense()
        # set the symmetric edge weights
        sym_mask = (mask_sigmoid + mask_sigmoid.transpose(0, 1)) / 2
        edge_mask = sym_mask[edge_idx[0].long(), edge_idx[1].long()]

        # inverse the weights before sigmoid in MessagePassing Module
        self.clear_masks()
        self.set_masks(graph, feat, edge_mask)

        # the model prediction with edge mask
        logits = self.model(graph, feat, edge_weight=self.edge_mask)
        probs = F.softmax(logits, dim=-1)

        self.clear_masks()

        return probs, edge_mask
