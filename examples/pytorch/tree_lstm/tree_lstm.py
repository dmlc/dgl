"""
Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks
https://arxiv.org/abs/1503.00075
"""
import itertools
import networkx as nx
import dgl.graph as G
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class ChildSumTreeLSTMCell(object):
    def __init__(self, x_size, h_size):
        self.W_iou = nn.Linear(x_size, 3 * h_size)
        self.U_iou = nn.Linear(h_size, 3 * h_size)
        self.W_f = nn.Linear(x_size, h_size)
        self.U_f = nn.Linear(h_size, h_size)

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
        iou = self.W_iou(node['x']) + self.U_iou(accum['h_tild'])
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        # equation (7)
        c = i * u + accum['c_tild']
        # equation (8)
        h = o * th.tanh(c)
        return {'h' : h, 'c' : c}

class TreeLSTM(nn.Module):
    def __init__(self, n_embeddings, x_size, h_size, n_classes, cell_type='childsum'):
        super(TreeLSTM, self).__init__()
        self.embedding = nn.Embedding(n_embeddings, x_size)
        self.linear = nn.Linear(h_size, n_classes)

    def message_func(self, src, edge):
        return src

    def update_func(self, node, accum):
        x = node_reprs['x']
        iou = th.mm(x, self.iou_w) + self.iou_b
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u
        h = o * th.tanh(c)
        return {'h' : h, 'c' : c}

    def internal_update_func(self, node_reprs, edge_reprs):
        raise NotImplementedError()

    def readout_func(self, g, train):
        if train:
            h = th.cat([d['h'] for d in g.nodes.values() if d['y'] is not None], 0)
            y = th.cat([d['y'] for d in g.nodes.values() if d['y'] is not None], 0)
            log_p = F.log_softmax(self.linear(h), 1)
            return -th.sum(y * log_p) / len(y)
        else:
            h = th.cat([reprs['h'] for reprs in g.nodes.values()], 0)
            y_bar = th.max(self.linear(h), 1)[1]
            # TODO
            for reprs, z in zip(labelled_reprs, y_bar):
                reprs['y_bar'] = z.item()

    def forward(self, g, train=False):
        """
        Parameters
        ----------
        g : networkx.DiGraph
        """
        assert any(d['y'] is not None for d in g.nodes.values()) # TODO

        g = G.DGLGraph(g)

        def update_func(node_reprs, edge_reprs):
            node_reprs = node_reprs.copy()
            if node_reprs['x'] is not None:
                node_reprs['x'] = self.embedding(node_reprs['x'])
            return node_reprs

        g.register_message_func(self.message_func)
        g.register_update_func(update_func, g.nodes)
        g.update_all()

        g.register_update_func(self.internal_update_func, g.nodes)
        leaves = list(filter(lambda x: g.in_degree(x) == 0, g.nodes))
        g.register_update_func(self.leaf_update_func, leaves)

        iterator = []
        frontier = [next(filter(lambda x: g.out_degree(x) == 0, g.nodes))]
        while frontier:
            src = sum([list(g.pred[x]) for x in frontier], [])
            trg = sum([[x] * len(g.pred[x]) for x in frontier], [])
            iterator.append((src, trg))
            frontier = src

        g.recv(leaves)
        g.propagate(reversed(iterator))

        return self.readout_func(g, train)


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
