"""
Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks
https://arxiv.org/abs/1503.00075
"""
import time
import itertools
import networkx as nx
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

def topological_traverse(G):
    indegree_map = {v: d for v, d in G.in_degree() if d > 0}
    # These nodes have zero indegree and ready to be returned.
    zero_indegree = [v for v, d in G.in_degree() if d == 0]
    while True:
        yield zero_indegree
        next_zero_indegree = []
        while zero_indegree:
            node = zero_indegree.pop()
            for _, child in G.edges(node):
                indegree_map[child] -= 1
                if indegree_map[child] == 0:
                    next_zero_indegree.append(child)
                    del indegree_map[child]
        if len(next_zero_indegree) == 0:
            break
        zero_indegree = next_zero_indegree

class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size)
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)

    def message_func(self, src, edge):
        return src

    def reduce_func(self, node, msgs):
        # msgs['h'] (B, 2, H)
        h_cat = msgs['h'].view(msgs['h'].size(0), -1)
        f = th.sigmoid(self.U_f(h_cat)).view(*msgs['h'].size())
        c_f = th.sum(f * msgs['c'], 1)
        return {'h_cat' : h_cat, 'c_f' : c_f}

    def update_func(self, node, accum):
        if accum is None: #leaf node
            iou = self.W_iou(node['x'])
        else: #internal node
            iou = self.U_iou(accum['h_cat'])
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)

        if accum is None:
            c = i * u
        else:
            c = i * u + accum['c_f']
        h = o * th.tanh(c)
        return {'h' : h, 'c' : c} 

class TreeLSTM(nn.Module):
    def __init__(self,
                 num_vocabs,
                 x_size,
                 h_size,
                 num_classes,
                 dropout,
                 pretrained_emb=None):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.embedding = nn.Embedding(num_vocabs, x_size)
        if pretrained_emb is not None:
            print('Using glove')
            self.embedding.weight.data.copy_(pretrained_emb)
            self.embedding.weight.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, num_classes)

        self.cell = TreeLSTMCell(x_size, h_size)

    def forward(self, batch, x, h, c, iterator=None, train=True):
        """Compute tree-lstm prediction given a batch.

        Parameters
        ----------
        batch : dgl.data.SSTBatch
            The data batch.
        x : Tensor
            Initial node input.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.
        iterator : graph iterator
            External iterator on graph.

        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        g = batch.graph
        g.register_message_func(self.cell.message_func, batchable=True)
        g.register_reduce_func(self.cell.reduce_func, batchable=True)
        g.register_update_func(self.cell.update_func, batchable=True)
        # feed embedding
        embeds = self.embedding(batch.wordid)

        x = x.index_copy(0, batch.nid_with_word, embeds)
        g.set_n_repr({'x' : x, 'h' : h, 'c' : c})
        # TODO(minjie): potential bottleneck
        if iterator is None:
            for frontier in topological_traverse(g):
                #print('frontier', frontier)
                g.update_to(frontier)
        else:
            for frontier in iterator:
                g.update_to(frontier)
        # compute logits
        h = g.pop_n_repr('h')
        h = self.dropout(h)
        logits = self.linear(h)
        return logits
