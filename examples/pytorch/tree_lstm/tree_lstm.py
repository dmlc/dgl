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

class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size)
        self.U_iou = nn.Linear(h_size, 3 * h_size)
        self.W_f = nn.Linear(x_size, h_size)
        self.U_f = nn.Linear(h_size, h_size)
        self.rt = 0.
        self.ut = 0.

    def message_func(self, src, edge):
        return src

    def reduce_func(self, node, msgs):
        # equation (2)
        h_tild = th.sum(msgs['h'], 1)
        # equation (4)
        wx = self.W_f(node['x']).unsqueeze(1)  # shape: (B, 1, H)
        uh = self.U_f(msgs['h']) # shape: (B, deg, H)
        f = th.sigmoid(wx + uh)  # shape: (B, deg, H)
        # equation (7) second term
        c_tild = th.sum(f * msgs['c'], 1)
        return {'h_tild' : h_tild, 'c_tild' : c_tild}
    
    def update_func(self, node, accum):
        # equation (3), (5), (6)
        if accum is None:
            iou = self.W_iou(node['x'])
        else:
            iou = self.W_iou(node['x']) + self.U_iou(accum['h_tild'])
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        # equation (7)
        if accum is None:
            c = i * u
        else:
            c = i * u + accum['c_tild']
        # equation (8)
        h = o * th.tanh(c)
        return {'h' : h, 'c' : c}

class TreeLSTM(nn.Module):
    def __init__(self,
                 num_vocabs,
                 x_size,
                 h_size,
                 num_classes,
                 dropout,
                 cell_type='childsum'):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        # TODO(minjie): pre-trained embedding like GLoVe
        self.embedding = nn.Embedding(num_vocabs, x_size)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, num_classes)
        if cell_type == 'childsum':
            self.cell = ChildSumTreeLSTMCell(x_size, h_size)
        else:
            raise RuntimeError('Unknown cell type:', cell_type)

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

'''
class NAryTreeLSTM(TreeLSTM):
    def __init__(self, n_embeddings, x_size, h_size, n_ary, n_classes):
        super().__init__(n_embeddings, x_size, h_size, n_classes)

        # TODO initializer
        self.iou_w = nn.Parameter(th.randn(x_size, 3 * h_size))
        self.iou_u = [nn.Parameter(th.randn(1, h_size, 3 * h_size)) for i in range(n_ary)]
        self.iou_b = nn.Parameter(th.zeros(1, 3 * h_size))

        # TODO initializer
        self.f_x = nn.Parameter(th.randn(x_size, h_size))
        self.f_h = [[nn.Parameter(th.randn(1, h_size, h_size))
                     for i in range(n_ary)] for i in range(n_ary)]
        self.f_b = nn.Parameter(th.zeros(1, h_size))

    def internal_update_func(self, node_reprs, edge_reprs):
        assert len(edge_reprs) > 0
        assert all(msg['h'] is not None and msg['c'] is not None for msg in edge_reprs)

        x = node_reprs['x']
        n_children = len(edge_reprs)

        iou_wx = th.mm(x, self.iou_w) if x is not None else 0
        iou_u = th.cat(self.iou_u[:n_children], 0)
        iou_h = th.cat([msg['h'] for msg in edge_reprs], 0).unsqueeze(1)
        iou_uh = th.sum(th.bmm(iou_h, iou_u), 0)
        i, o, u = th.chunk(iou_wx + iou_uh + self.iou_b, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)

        f_wx = th.mm(x, self.f_x).repeat(n_children, 1) if x is not None else 0
        f_h = iou_h.repeat(n_children, 1, 1)
        f_u = th.cat(sum([self.f_h[i][:n_children] for i in range(n_children)], []), 0)
        f_uh = th.sum(th.bmm(f_h, f_u).view(n_children, n_children, -1), 0)
        f = th.sigmoid(f_wx + f_uh + self.f_b)

        c = th.cat([msg['c'] for msg in edge_reprs], 0)
        c = i * u + th.sum(f * c, 0, keepdim=True)
        h = o * th.tanh(c)

        return {'h' : h, 'c' : c}
'''
