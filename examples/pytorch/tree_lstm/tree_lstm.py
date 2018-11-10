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
import dgl

def tensor_topo_traverse(g, cuda):
    n = g.number_of_nodes()
    if cuda:
        adjmat = g._graph.adjacency_matrix().get(th.device('cuda:{}'.format(cuda)))
        mask = th.ones((n, 1)).cuda()
    else:
        adjmat = g._graph.adjacency_matrix().get(th.device('cpu'))
        mask = th.ones((n, 1))
    degree = th.spmm(adjmat, mask)
    while th.sum(mask) != 0.:
        v = (degree == 0.).float()
        v = v * mask
        mask = mask - v
        frontier = th.squeeze(th.squeeze(v).nonzero(), 1)
        yield frontier
        degree -= th.spmm(adjmat, v)

class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size)
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        f = th.sigmoid(self.U_f(h_cat)).view(*nodes.mailbox['h'].size())
        c_f = th.sum(f * nodes.mailbox['c'], 1)
        return {'x': self.U_iou(h_cat), 'c': c_f}

    def apply_node_func(self, nodes):
        iou = nodes.data['x']
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
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
#        print(g.number_of_nodes(), g.number_of_edges())
        g.register_message_func(self.cell.message_func)
        g.register_reduce_func(self.cell.reduce_func)
        g.register_apply_node_func(self.cell.apply_node_func)
        # feed embedding
        embeds = self.embedding(batch.wordid)
        iou = self.cell.W_iou(embeds)
        x = x.index_copy(0, batch.nid_with_word, iou)
        g.ndata['x'] = x
        g.ndata['h'] = h
        g.ndata['c'] = c
        # TODO(minjie): potential bottleneck

        g.prop_nodes(dgl.topological_nodes_generator(g))#tensor_topo_traverse(g, 0)) #dgl.topological_nodes_generator(g))
        # compute logits
        h = g.ndata.pop('h')
        h = self.dropout(h)
        logits = self.linear(h)
        return logits
