"""Torch Module for PGExplainer"""
import math

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import dgl
import copy
import torch.nn.functional as F
from scipy.sparse import coo_matrix, csr_matrix
import dgl.function as fn
from torch.optim import Adam
from torch.utils.data import Dataset

from ....base import NID
from ....convert import to_networkx
from ....subgraph import node_subgraph
from ....transforms.functional import remove_nodes

__all__ = ["PGExplainer"]

class PGExplainer(nn.Module):

    def __init__(self, model, num_features,
                 epochs=20, lr=0.005, coff_size=0.01, coff_ent=5e-4,
                 t0=5.0, t1=1.0, sample_bias=0.0, num_hops=None, device='cpu'):
        super(PGExplainer, self).__init__()

        self.model = model
        self.device = device
        self.model.to(self.device)
        self.num_features = num_features * 2

        # training parameters for PGExplainer
        self.epochs = epochs
        self.lr = lr
        self.size = coff_size
        self.ent = coff_ent
        self.t0 = t0
        self.t1 = t1
        self.sample_bias = sample_bias

        self.num_hops = self.update_num_hops(num_hops)
        self.init_bias = 0.0

        # Explanation model in PGExplainer
        self.elayers = nn.ModuleList()
        self.elayers.append(nn.Sequential(nn.Linear(self.num_features, 64),
                                          nn.ReLU()))
        self.elayers.append(nn.Linear(64, 1))
        self.elayers.to(self.device)

        self.coeffs = {
            "epochs": epochs,
            "lr": lr,
            "size": coff_size,
            "ent": coff_ent,
            "t0": t0,
            "t1": t1,
            "sample_bias": sample_bias,
        }

    def set_masks(self, g, feat, edge_mask=None):
        N, F = feat.shape
        E = g.num_edges()

        init_bias = self.init_bias
        std = torch.nn.init.calculate_gain('relu') * math.sqrt(2.0 / (2 * N))

        if edge_mask is None:
            self.edge_mask = torch.randn(E) * std + init_bias
        else:
            self.edge_mask = edge_mask

        self.edge_mask = self.edge_mask.to(g.device)

    def clear_masks(self):
        self.edge_mask = None

    def update_num_hops(self, num_hops):
        if num_hops is not None:
            return num_hops

        # @kunal: Tianhao, what should we do here?
        k = 0
        for module in self.model.modules():
            k += 1
        return k

    def loss(self, prob, ori_pred):

        logit = prob[ori_pred]
        logit += 1e-6
        pred_loss = -torch.log(logit)

        # size
        edge_mask = self.sparse_mask_values
        if self.coeffs['size'] <= 0:
            size_loss = self.coeffs["size"] * torch.sum(edge_mask)
        else:
            size_loss = self.coeffs["size"] * F.relu(torch.sum(edge_mask) - self.coeffs['size'])

        # entropy
        scale = 0.99
        edge_mask = self.edge_mask * (2 * scale - 1.0) + (1.0 - scale)
        mask_ent = -edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)
        mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)

        loss = pred_loss + size_loss + mask_ent_loss

        return loss

    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        """Uniform random numbers for the concrete distribution"""

        if training:
            bias = self.coeffs["sample_bias"]
            random_noise = torch.rand(log_alpha.size()).to(log_alpha.device)
            random_noise = bias + (1 - 2 * bias) * random_noise
            gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(log_alpha)

        return gate_inputs

    def explain_graph(self, graph, feat, embed, tmp=1.0, training=False):

        edge_idx = graph.edges()

        node_size = embed.shape[0]
        col, row = edge_idx
        f1 = embed[col]
        f2 = embed[row]
        f12self = torch.cat([f1, f2], dim=-1)

        # using the node embedding to calculate the edge weight
        h = f12self.to(self.device)
        for elayer in self.elayers:
            h = elayer(h)
        values = h.reshape(-1)

        values = self.concrete_sample(values, beta=tmp, training=training)
        self.sparse_mask_values = values

        mask_sparse = torch.sparse_coo_tensor(
            [edge_idx[0].tolist(),edge_idx[1].tolist()],
            values.tolist(),
            (node_size,node_size)
        )
        mask_sigmoid = mask_sparse.to_dense()
        # set the symmetric edge weights
        sym_mask = (mask_sigmoid + mask_sigmoid.transpose(0, 1)) / 2
        edge_mask = sym_mask[edge_idx[0], edge_idx[1]]

        # inverse the weights before sigmoid in MessagePassing Module
        self.clear_masks()
        self.set_masks(graph, feat, edge_mask)

        # the model prediction with edge mask
        logits = self.model(graph, feat, edge_idx)
        probs = F.softmax(logits, dim=-1)

        self.clear_masks()

        return probs, edge_mask

    def train_explanation_network(self, dataset):

        optimizer = Adam(self.elayers.parameters(), lr=self.coeffs['lr'])

        emb_dict = {}
        ori_pred_dict = {}

        with torch.no_grad():
            for idx, (g, l) in enumerate(dataset):
                feat = g.ndata["attr"]

                self.model.eval()

                logits = self.model(g, feat)
                emb = self.model(g, feat, graph=False)
                emb_dict[idx] = emb.data.cpu()
                ori_pred_dict[idx] = logits.argmax(-1).data.cpu()

        # train the mask generator
        for epoch in range(self.epochs):
            loss = 0.0
            pred_list = []

            tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs))

            self.elayers.train()
            optimizer.zero_grad()

            for idx, (g, l) in enumerate(dataset):

                feat = g.ndata["attr"]
                g = g.to(self.device)

                node_idx = g.nodes()

                prob, edge_mask = self.explain_graph(g,
                                                     feat,
                                                     embed=emb_dict[idx],
                                                     tmp=tmp,
                                                     training=True)

                self.edge_mask = edge_mask

                loss_tmp = self.loss(prob.squeeze(), ori_pred_dict[idx])
                loss_tmp.backward()

                loss += loss_tmp.item()
                pred_label = prob.argmax(-1).item()
                pred_list.append(pred_label)

            optimizer.step()
            print(f'Epoch: {epoch} | Loss: {loss}')

    # def forward(self, inputs, training=False):
    #
    #     feat, embed, adj, tmp, label = inputs
    #
    #     adj = coo_matrix(adj)
    #
    #     g = dgl.graph((adj.row, adj.col))
    #     edge_idx = g.edges()
    #
    #     g.ndata['feat'] = embed
    #
    #     f1 = embed[adj.row]
    #     f2 = embed[adj.col]
    #
    #     f12self = torch.cat([f1, f2], dim=-1)
    #
    #     h = f12self
    #     for layer in self.elayers:
    #         h = layer(h)
    #
    #     self.values = h.view(-1)
    #
    #     values = self.concrete_sample(self.values,
    #                                   beta=tmp,
    #                                   training=training)
    #     self.sparse_mask_values = values
    #
    #     adj.row = torch.tensor(adj.row).long()
    #     adj.col = torch.tensor(adj.col).long()
    #
    #     indices = torch.cat([adj.row.unsqueeze(-1),
    #                          adj.col.unsqueeze(-1)],
    #                         dim=-1)
    #     mask_sparse = torch.sparse.FloatTensor(indices.t(),
    #                                            values,
    #                                            size=adj.shape)
    #     mask_sigmoid = mask_sparse.to_dense()
    #     # set the symmetric edge weights
    #     sym_mask = (mask_sigmoid + mask_sigmoid.transpose(0, 1)) / 2
    #     edge_mask = sym_mask[edge_idx[0], edge_idx[1]]
    #     # @kunal: missing set mask
    #     # edge weight for subgraph
    #     # extract the edge mask in the set mask
    #     self.clear_masks()
    #     self.set_masks(g, feat, edge_mask)
    #
    #     # @kunal: using the set mask, get the masked adj matrix
    #     # final edge mask used
    #     masked_adj = self.masked_adj(self.edge_mask, adj)
    #
    #     # @kunal: using mask adj matrix
    #     # constrct the whole graph
    #     # set the edge weight = weight of the mask
    #     # we will have a graph with 'weight' parameter set
    #     mask_g = copy.deepcopy(g)
    #     mask_g.edata['weight'] = masked_adj
    #
    #     output = self.model(mask_g, feat)
    #
    #     # For node classification:
    #     # node_pred = output[nodeid, :]
    #     node_pred = output
    #     res = nn.functional.softmax(node_pred, dim=0)
    #
    #     return res
    #
    # def masked_adj(self, mask, adj):
    #     sym_mask = mask
    #     sym_mask = (sym_mask + sym_mask.t()) / 2
    #
    #     num_nodes = adj.shape[0]
    #     indices = torch.cat([adj.row.unsqueeze(-1),
    #                          adj.col.unsqueeze(-1)], dim=-1)
    #
    #     values = torch.from_numpy(adj.data).float()
    #     sparseadj = torch.sparse.FloatTensor(indices.t(),
    #                                          values,
    #                                          size=(num_nodes, num_nodes))
    #     adj = sparseadj.to_dense()
    #
    #     masked_adj = adj * sym_mask
    #     diag_mask = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)
    #
    #     return masked_adj * diag_mask
