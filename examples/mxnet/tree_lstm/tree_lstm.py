"""
Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks
https://arxiv.org/abs/1503.00075
"""
import itertools
import time

import dgl

import mxnet as mx
import networkx as nx
import numpy as np
from mxnet import gluon


class _TreeLSTMCellNodeFunc(gluon.HybridBlock):
    def hybrid_forward(self, F, iou, b_iou, c):
        iou = F.broadcast_add(iou, b_iou)
        i, o, u = iou.split(num_outputs=3, axis=1)
        i, o, u = i.sigmoid(), o.sigmoid(), u.tanh()
        c = i * u + c
        h = o * c.tanh()

        return h, c


class _TreeLSTMCellReduceFunc(gluon.HybridBlock):
    def __init__(self, U_iou, U_f):
        super(_TreeLSTMCellReduceFunc, self).__init__()
        self.U_iou = U_iou
        self.U_f = U_f

    def hybrid_forward(self, F, h, c):
        h_cat = h.reshape((0, -1))
        f = self.U_f(h_cat).sigmoid().reshape_like(h)
        c = (f * c).sum(axis=1)
        iou = self.U_iou(h_cat)
        return iou, c


class _TreeLSTMCell(gluon.HybridBlock):
    def __init__(self, h_size):
        super(_TreeLSTMCell, self).__init__()
        self._apply_node_func = _TreeLSTMCellNodeFunc()
        self.b_iou = self.params.get(
            "bias", shape=(1, 3 * h_size), init="zeros"
        )

    def message_func(self, edges):
        return {"h": edges.src["h"], "c": edges.src["c"]}

    def apply_node_func(self, nodes):
        iou = nodes.data["iou"]
        b_iou, c = self.b_iou.data(iou.context), nodes.data["c"]
        h, c = self._apply_node_func(iou, b_iou, c)
        return {"h": h, "c": c}


class TreeLSTMCell(_TreeLSTMCell):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__(h_size)
        self._reduce_func = _TreeLSTMCellReduceFunc(
            gluon.nn.Dense(3 * h_size, use_bias=False),
            gluon.nn.Dense(2 * h_size),
        )
        self.W_iou = gluon.nn.Dense(3 * h_size, use_bias=False)

    def reduce_func(self, nodes):
        h, c = nodes.mailbox["h"], nodes.mailbox["c"]
        iou, c = self._reduce_func(h, c)
        return {"iou": iou, "c": c}


class ChildSumTreeLSTMCell(_TreeLSTMCell):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = gluon.nn.Dense(3 * h_size, use_bias=False)
        self.U_iou = gluon.nn.Dense(3 * h_size, use_bias=False)
        self.U_f = gluon.nn.Dense(h_size)

    def reduce_func(self, nodes):
        h_tild = nodes.mailbox["h"].sum(axis=1)
        f = self.U_f(nodes.mailbox["h"]).sigmoid()
        c = (f * nodes.mailbox["c"]).sum(axis=1)
        return {"iou": self.U_iou(h_tild), "c": c}


class TreeLSTM(gluon.nn.Block):
    def __init__(
        self,
        num_vocabs,
        x_size,
        h_size,
        num_classes,
        dropout,
        cell_type="nary",
        pretrained_emb=None,
        ctx=None,
    ):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.embedding = gluon.nn.Embedding(num_vocabs, x_size)
        if pretrained_emb is not None:
            print("Using glove")
            self.embedding.initialize(ctx=ctx)
            self.embedding.weight.set_data(pretrained_emb)
        self.dropout = gluon.nn.Dropout(dropout)
        self.linear = gluon.nn.Dense(num_classes)
        cell = TreeLSTMCell if cell_type == "nary" else ChildSumTreeLSTMCell
        self.cell = cell(x_size, h_size)
        self.ctx = ctx

    def forward(self, batch, h, c):
        """Compute tree-lstm prediction given a batch.
        Parameters
        ----------
        batch : dgl.data.SSTBatch
            The data batch.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.
        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        g = batch.graph
        g = g.to(self.ctx)
        # feed embedding
        embeds = self.embedding(batch.wordid * batch.mask)
        wiou = self.cell.W_iou(self.dropout(embeds))
        g.ndata["iou"] = wiou * batch.mask.expand_dims(-1).astype(wiou.dtype)
        g.ndata["h"] = h
        g.ndata["c"] = c
        # propagate
        dgl.prop_nodes_topo(
            g,
            message_func=self.cell.message_func,
            reduce_func=self.cell.reduce_func,
            apply_node_func=self.cell.apply_node_func,
        )
        # compute logits
        h = self.dropout(g.ndata.pop("h"))
        logits = self.linear(h)
        return logits
