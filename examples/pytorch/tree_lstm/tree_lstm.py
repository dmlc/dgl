"""
Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks
https://arxiv.org/abs/1503.00075
"""


import dgl.graph as G
import torch as th
import torch.nn as nn


class Preprocessor(nn.Module):
    def __init__(self, n_embeddings, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(n_embeddings, embedding_size)

    def forward(self, g):
        """
        Parameters
        ----------
        g : networkx.DiGraph
        """
        g = G.DGLGraph(g)


class TreeLSTM(nn.Module):
    @staticmethod
    def message_func(src, trg, _):
        return {'h' : src['h'], 'c' : src['c']}

    @staticmethod
    def leaf_update_func(node_reprs, edge_reprs):
        iou = th.mm(node_reprs['x'], self.iou_x) + self.iou_b
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u
        h = o * th.tanh(c)
        return h, c

    def forward(self, g):
        """
        Parameters
        ----------
        g : networkx.DiGraph
        """

        order = nx.topological_sort(g)
        frontier = itertools.takewhile(lambda x: not g.pred[x], order)
        order = order[len(frontier):]
        assert order

        g = self.preprocessor(g)

        g.register_update_func(self.update_func(leaf=True), frontier)
        g.update_all()

        while order:
            frontier = itertools.takewhile(lambda x: x not in order, order)
            edge_list = sum(([(y, x) for y in g.pred[x]] for x in frontier), [])
            g.register_message_func(self.message_func, edge_list)
            g.register_update_func(self.update_func(leaf=False, frontier)
            for x in frontier:
                g.update_to(x)
            order = order[len(frontier):]


class ChildSumTreeLSTM(TreeLSTM):
    def __init__(self, preprocessor, x_size, h_size):
        super().__init__()
        self.preprocessor = preprocessor

        self.iou_x = nn.Parameter(th.randn(x_size, 3 * h_size)) # TODO initializer
        self.iou_h = nn.Parameter(th.randn(h_size, 3 * h_size)) # TODO initializer
        self.iou_b = nn.Parameter(th.zeros(1, 3 * h_size))

        self.f_x = nn.Parameter(th.randn(x_size, h_size)) # TODO initializer
        self.f_h = nn.Parameter(th.randn(h_size, h_size)) # TODO initializer
        self.f_b = nn.Parameter(th.zeros(1, h_size))

    def update_func(self, leaf):
        if leaf:
            return self.leaf_update_func
        else:
            def u(node_reprs, edge_reprs):
                x = node_reprs['x']
                h_bar = sum(msg['h'] for msg in edge_reprs)
                iou = th.mm(x, self.iou_x) + th.mm(h_bar, self.iou_h) + self.iou_b
                i, o, u = th.chunk(iou, 3, 1)
                i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)

                batch_x = x.repeat(len(edge_reprs), 1)
                batch_h = 

                wx = th.mm(x, self.f_x)
                uh = th.mm(th.cat([msg['h'] for msg in edge_reprs], 0), self.f_h)
                f = wx.repeat(len(edge_reprs), 1) + uh + f_b

                c = th.cat([msg['c'] for msg in edge_reprs], 0)
                c = i * u + th.sum(f * c, 0)
                h = o * th.tanh(c)

                return h, c
            return u


class NAryTreeLSTM(TreeLSTM):
    def __init__(self, preprocessor, x_size, h_size):
        super().__init__()
        self.preprocessor = preprocessor

    def update_func(self, leaf):
        if leaf:
            return self.leaf_update_func
        else:
            def u(node_reprs, edge_reprs):
                x = node_reprs['x']
