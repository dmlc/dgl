"""
Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks
https://arxiv.org/abs/1503.00075
"""
import itertools
import time

import dgl
import networkx as nx
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)

    def message_func(self, edges):
        return {"h": edges.src["h"], "c": edges.src["c"]}

    def reduce_func(self, nodes):
        h_cat = nodes.mailbox["h"].view(nodes.mailbox["h"].size(0), -1)
        f = th.sigmoid(self.U_f(h_cat)).view(*nodes.mailbox["h"].size())
        c = th.sum(f * nodes.mailbox["c"], 1)
        return {"iou": self.U_iou(h_cat), "c": c}

    def apply_node_func(self, nodes):
        iou = nodes.data["iou"] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data["c"]
        h = o * th.tanh(c)
        return {"h": h, "c": c}


class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {"h": edges.src["h"], "c": edges.src["c"]}

    def reduce_func(self, nodes):
        h_tild = th.sum(nodes.mailbox["h"], 1)
        f = th.sigmoid(self.U_f(nodes.mailbox["h"]))
        c = th.sum(f * nodes.mailbox["c"], 1)
        return {"iou": self.U_iou(h_tild), "c": c}

    def apply_node_func(self, nodes):
        iou = nodes.data["iou"] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data["c"]
        h = o * th.tanh(c)
        return {"h": h, "c": c}


class TreeLSTM(nn.Module):
    def __init__(
        self,
        num_vocabs,
        x_size,
        h_size,
        num_classes,
        dropout,
        cell_type="nary",
        pretrained_emb=None,
    ):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.embedding = nn.Embedding(num_vocabs, x_size)
        if pretrained_emb is not None:
            print("Using glove")
            self.embedding.weight.data.copy_(pretrained_emb)
            self.embedding.weight.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, num_classes)
        cell = TreeLSTMCell if cell_type == "nary" else ChildSumTreeLSTMCell
        self.cell = cell(x_size, h_size)

    def forward(self, batch, g, h, c):
        """Compute tree-lstm prediction given a batch.
        Parameters
        ----------
        batch : dgl.data.SSTBatch
            The data batch.
        g : dgl.DGLGraph
            Tree for computation.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.
        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        # feed embedding
        embeds = self.embedding(batch.wordid * batch.mask)
        g.ndata["iou"] = self.cell.W_iou(
            self.dropout(embeds)
        ) * batch.mask.float().unsqueeze(-1)
        g.ndata["h"] = h
        g.ndata["c"] = c
        # propagate
        dgl.prop_nodes_topo(
            g,
            self.cell.message_func,
            self.cell.reduce_func,
            apply_node_func=self.cell.apply_node_func,
        )
        # compute logits
        h = self.dropout(g.ndata.pop("h"))
        logits = self.linear(h)
        return logits
