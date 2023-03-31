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
        self.coff_size = coff_size
        self.coff_ent = coff_ent
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

        self.reals = []
        self.preds = []

    def update_num_hops(self, num_hops):
        if num_hops is not None:
            return num_hops

        k = 0
        for module in self.model.modules():
            k += 1
        return k

    def _masked_adj(self, mask, adj):
        sym_mask = mask
        sym_mask = (sym_mask + sym_mask.t()) / 2

        num_nodes = adj.shape[0]
        indices = torch.cat([adj.row.unsqueeze(-1),
                             adj.col.unsqueeze(-1)], dim=-1)

        values = torch.from_numpy(adj.data).float()
        sparseadj = torch.sparse.FloatTensor(indices.t(),
                                             values,
                                             size=(num_nodes, num_nodes))
        adj = sparseadj.to_dense()

        masked_adj = adj * sym_mask
        diag_mask = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)

        return masked_adj * diag_mask

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

    def forward(self, inputs, training=False):

        x, embed, adj, tmp, label = inputs

        adj = coo_matrix(adj)

        g = dgl.graph((adj.row, adj.col))

        g.ndata['x'] = embed

        f1 = embed[adj.row]
        f2 = embed[adj.col]

        f12self = torch.cat([f1, f2], dim=-1)

        h = f12self
        for layer in self.elayers:
            h = layer(h)

        self.values = h.view(-1)

        values = self.concrete_sample(self.values,
                                      beta=tmp,
                                      training=training)
        self.sparse_mask_values = values

        adj.row = torch.tensor(adj.row).long()
        adj.col = torch.tensor(adj.col).long()

        indices = torch.cat([adj.row.unsqueeze(-1),
                             adj.col.unsqueeze(-1)],
                            dim=-1)
        sparsemask = torch.sparse.FloatTensor(indices.t(),
                                              values,
                                              size=adj.shape)
        mask = sparsemask.to_dense()
        # @kunal: missing set mask
        # edge weight for subgraph
        # extract the edge mask in the set mask
        self.__clear_masks__()
        self.__set_masks__(x, edge_index, edge_mask)

        # @kunal: using the set mask, get the masked adj matrix
        # final edge mask used
        masked_adj = self._masked_adj(self.__edge_mask__, adj)

        # @kunal: using mask adj matrix
        # constrct the whole graph
        # set the edge weight = weight of the mask
        # we will have a graph with 'weight' parameter set
        mask_g = copy.deepcopy(g)
        mask_g.edata['weight'] = masked_adj

        output = self.model(mask_g, x)

        # For node classification:
        # node_pred = output[nodeid, :]
        node_pred = output
        res = nn.functional.softmax(node_pred, dim=0)
        return res

    def loss(self, pred, ori_pred_label):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        logit = pred[ori_pred_label]
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
        edge_mask = self.mask * (2 * scale - 1.0) + (1.0 - scale)
        mask_ent = -edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)
        mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)

        loss = pred_loss + size_loss + mask_ent_loss
        return loss

    def train(self, dataset):

        optimizer = Adam(self.elayers.parameters(), lr=self.coeffs['lr'])

        clip_value_min = -0.01
        clip_value_max = 0.01

        epochs = self.coeffs['epochs']
        t0 = self.coeffs['t0']
        t1 = self.coeffs['t1']

        # new add
        for epoch in range(epochs):
            loss = 0
            tmp = t0 * pow(t1 / t0, epoch / epochs)

            for idx, (graph, label) in enumerate(dataset):

                # input feature
                feat = graph.ndata.pop("attr")
                # predicted label
                _, pred_label = torch.max(self.model(graph, feat), 1)

                with torch.no_grad():
                    # embedding for one model
                    emb = self.model(graph, feat, graph=False)

                adj_tensor = graph.adj().to_dense()

                optimizer.zero_grad()

                # forward function
                pred = self((feat, emb, adj_tensor, tmp, label))
                # loss function
                cl = self.loss(pred, pred_label, label)
                cl.backward()

                loss += cl.item()

                torch.nn.utils.clip_grad_value_(self.parameters(),
                                                clip_value_min,
                                                clip_value_max)
                optimizer.step()

            # for gid in range(selected_adjs.shape[0]):
            #     # Load the graph batch from dataset
            #     # fea, emb, adj_tensor, tmp, label = ...
            #
            #     optimizer.zero_grad()
            #     pred = self((fea, emb, adj_tensor, tmp, label))
            #     cl = self.loss(pred, pred_label, label)
            #
            #     loss += cl.item()
            #     cl.backward()
            #     torch.nn.utils.clip_grad_value_(self.parameters(), clip_value_min, clip_value_max)
            #     optimizer.step()
            #
            # if epoch % 1 == 0:
            #     print('epoch', epoch, 'loss', loss)
            #
            #     for gid in range(int(selected_adjs.shape[0] / 10)):
            #         # Load the graph batch from dataset
            #         # fea, emb, adj_tensor, tmp, label = ...
            #
            #         adj_tensor = torch.from_numpy(adj.toarray().astype(np.float32))
            #         self.eval()
            #         self((fea, emb, adj_tensor, 1.0, label), need_grad=False)
            #
            #         self.acc(gid, selected_edge_lists, selected_edge_label_lists)
            #
            #         self.train(dataset)

    def acc(self, gid, edge_lists, edge_label_lists):
        mask = self.masked_adj.detach().numpy()
        edge_labels = edge_label_lists[gid]
        edge_list = edge_lists[gid]

        for (r, c), l in list(zip(edge_list, edge_labels)):
            if r > c:
                self.reals.append(l)
                self.preds.append(mask[r][c])

    def explain_graph(self, fea, emb, adj, label,
                      graphid, topk, node_label_lists,
                      edge_lists, edge_label_lists):

        self(fea, emb, adj, 1.0, label)
        self.acc(graphid, edge_lists, edge_label_lists)

        after_adj_dense = self.masked_adj.detach().numpy()
        after_adj = coo_matrix(after_adj_dense)

        rcd = np.concatenate(
            [np.expand_dims(after_adj.row, -1),
             np.expand_dims(after_adj.col, -1),
             np.expand_dims(after_adj.data, -1)], -1)
        pos_edges = []
        filter_edges = []
        edge_weights = np.triu(after_adj_dense).flatten()

        sorted_edge_weights = np.sort(edge_weights)
        thres_index = max(int(edge_weights.shape[0] - topk), 0)
        thres = sorted_edge_weights[thres_index]

        for r, c, d in rcd:
            r = int(r)
            c = int(c)
            d = float(d)
            if r < c:
                continue
            if d >= thres:
                pos_edges.append((r, c))
            filter_edges.append((r, c))

        node_label = node_label_lists[graphid]
        max_label = np.max(node_label) + 1
        nmb_nodes = len(node_label)

        # Convert to PyTorch tensors and DGL graph
        fea = torch.from_numpy(fea).float()
        emb = torch.from_numpy(emb).float()
        adj = torch.from_numpy(adj).float()
        edge_index = np.stack((rcd[:, 0], rcd[:, 1]), axis=0)
        edge_index = torch.from_numpy(edge_index).long()
        edge_weight = torch.from_numpy(
            after_adj_dense[np.triu_indices(after_adj_dense.shape[0], k=1)]).float()
        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=nmb_nodes, device=self.device)
        g.edata['w'] = edge_weight

        return g, fea, emb, adj, \
               node_label, max_label, pos_edges, \
               filter_edges, thres
