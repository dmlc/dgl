"""
Supervised Community Detection with Hierarchical Graph Neural Networks
https://arxiv.org/abs/1705.08415

Deviations from paper:
- Addition of global aggregation operator.
- Message passing is equivalent to `A^j \cdot X`, instead of `\min(1, A^j) \cdot X`.
"""


# TODO self-loop?
# TODO in-place edit of node_reprs/edge_reprs in message_func/update_func?
# TODO batch-norm


import copy
import itertools
import dgl.graph as G
import networkx as nx
import torch as th
import torch.nn as nn


class GLGModule(nn.Module):
    __SHADOW__ = 'shadow'
    def __init__(self, in_feats, out_feats, radius):
        super().__init__()
        self.radius = radius

        new_linear = lambda: nn.Linear(in_feats, out_feats)
        new_module_list = lambda: nn.ModuleList([new_linear() for i in range(radius)])

        self.theta_x, self.theta_y, self.theta_deg, self.theta_global = \
            new_linear(), new_linear(), new_linear(), new_linear()
        self.theta_list = new_module_list()

        self.gamma_x, self.gamma_y, self.gamma_deg, self.gamma_global = \
            new_linear(), new_linear(), new_linear(), new_linear()
        self.gamma_list = new_module_list()

    @staticmethod
    def copy(which):
        if which == 'src':
            return lambda src, trg, _: src.copy()
        elif which == 'trg':
            return lambda src, trg, _: trg.copy()

    @staticmethod
    def aggregate(msg_fld, trg_fld, normalize=False):
        def a(node_reprs, edge_reprs):
            node_reprs = node_reprs.copy()
            node_reprs[trg_fld] = sum(msg[msg_fld] for msg in edge_reprs)
            if normalize:
                node_reprs[trg_fld] /= len(edge_reprs)
            return node_reprs
        return a

    @staticmethod
    def pull(msg_fld, trg_fld):
        def p(node_reprs, edge_reprs):
            node_reprs = node_reprs.copy()
            node_reprs[trg_fld] = edge_reprs[0][msg_fld]
            return node_reprs
        return p

    def local_aggregate(self, g):
        def step():
            g.register_message_func(self.copy('src'), g.edges)
            g.register_update_func(self.aggregate('x', 'x'), g.nodes)
            g.update_all()

        step()
        for reprs in g.nodes.values():
            reprs[0] = reprs['x']
        for i in range(1, self.radius):
            for j in range(2 ** (i - 1)):
                step()
            for reprs in g.nodes.values():
                reprs[i] = reprs['x']

    @staticmethod
    def global_aggregate(g):
        shadow = GLGModule.__SHADOW__
        copy, aggregate, pull = GLGModule.copy, GLGModule.aggregate, GLGModule.pull

        node_list = list(g.nodes)
        uv_list = [(node, shadow) for node in g.nodes]
        vu_list = [(shadow, node) for node in g.nodes]

        g.add_node(shadow) # TODO context manager

        tuple(itertools.starmap(g.add_edge, uv_list))
        g.register_message_func(copy('src'), uv_list)
        g.register_update_func(aggregate('x', 'global', normalize=True), (shadow,))
        g.update_to(shadow)

        tuple(itertools.starmap(g.add_edge, vu_list))
        g.register_message_func(copy('src'), vu_list)
        g.register_update_func(pull('global', 'global'), node_list)
        g.update_from(shadow)

        g.remove_node(shadow)

    @staticmethod
    def multiply_by_degree(g):
        g.register_message_func(lambda *args: None, g.edges)
        def update_func(node_reprs, _):
            node_reprs = node_reprs.copy()
            node_reprs['deg'] = node_reprs['x'] * node_reprs['degree']
            return node_reprs
        g.register_update_func(update_func, g.nodes)
        g.update_all()

    @staticmethod
    def message_func(src, trg, _):
        return {'y' : src['x']}

    def update_func(self, which):
        if which == 'node':
            linear_x, linear_y, linear_deg, linear_global = \
                self.theta_x, self.theta_y, self.theta_deg, self.theta_global
            linear_list = self.theta_list
        elif which == 'edge':
            linear_x, linear_y, linear_deg, linear_global = \
                self.gamma_x, self.gamma_y, self.gamma_deg, self.gamma_global
            linear_list = self.gamma_list

        def u(node_reprs, edge_reprs):
            edge_reprs = filter(lambda x: x is not None, edge_reprs)
            y = sum(x['y'] for x in edge_reprs)
            node_reprs = node_reprs.copy()
            node_reprs['x'] = linear_x(node_reprs['x']) \
                            + linear_y(y) \
                            + linear_deg(node_reprs['deg']) \
                            + linear_global(node_reprs['global']) \
                            + sum(linear(node_reprs[i]) \
                                  for i, linear in enumerate(linear_list))
            return node_reprs

        return u

    def forward(self, g, lg, glg):
        self.local_aggregate(g)
        self.local_aggregate(lg)

        self.global_aggregate(g)
        self.global_aggregate(lg)

        self.multiply_by_degree(g)
        self.multiply_by_degree(lg)

        # TODO efficiency
        for node, reprs in g.nodes.items():
            glg.nodes[node].update(reprs)
        for node, reprs in lg.nodes.items():
            glg.nodes[node].update(reprs)

        glg.register_message_func(self.message_func, glg.edges)
        glg.register_update_func(self.update_func('node'), g.nodes)
        glg.register_update_func(self.update_func('edge'), lg.nodes)
        glg.update_all()

        # TODO efficiency
        for node, reprs in g.nodes.items():
            reprs.update(glg.nodes[node])
        for node, reprs in lg.nodes.items():
            reprs.update(glg.nodes[node])


class GNNModule(nn.Module):
    def __init__(self, in_feats, out_feats, order, radius):
        super().__init__()
        self.module_list = nn.ModuleList([GLGModule(in_feats, out_feats, radius)
                                          for i in range(order)])

    def forward(self, pairs, fusions):
        for module, (g, lg), glg in zip(self.module_list, pairs, fusions):
            module(g, lg, glg)
        for lhs, rhs in zip(pairs[:-1], pairs[1:]):
            for node, reprs in lhs[1].nodes.items():
                x_rhs = reprs['x']
                reprs['x'] = x_rhs + rhs[0].nodes[node]['x']
                rhs[0].nodes[node]['x'] += x_rhs


class GNN(nn.Module):
    def __init__(self, feats, order, radius, n_classes):
        super().__init__()
        self.order = order
        self.linear = nn.Linear(feats[-1], n_classes)
        self.module_list = nn.ModuleList([GNNModule(in_feats, out_feats, order, radius)
                            for in_feats, out_feats in zip(feats[:-1], feats[1:])])

    @staticmethod
    def line_graph(g):
        lg = nx.line_graph(g)
        glg = nx.DiGraph()
        glg.add_nodes_from(g.nodes)
        glg.add_nodes_from(lg.nodes)
        for u, v in g.edges:
            glg.add_edge(u, (u, v))
            glg.add_edge((u, v), u)
            glg.add_edge(v, (u, v))
            glg.add_edge((u, v), v)
        return lg, glg

    @staticmethod
    def nx2dgl(g):
        deg_dict = dict(nx.degree(g))
        z = sum(deg_dict.values())
        dgl_g = G.DGLGraph(g)
        for node, reprs in dgl_g.nodes.items():
            reprs['degree'] = deg_dict[node]
            reprs['x'] = th.full((1, 1), reprs['degree'] / z)
            reprs.update(g.nodes[node])
        return dgl_g
    
    def forward(self, g):
        """
        Parameters
        ----------
        g : networkx.DiGraph
        """
        pair_list, glg_list = [], []
        dgl_g = self.nx2dgl(g)
        origin = dgl_g
        for i in range(self.order):
            lg, glg = self.line_graph(g)
            dgl_lg = self.nx2dgl(lg)
            pair_list.append((dgl_g, copy.deepcopy(dgl_lg)))
            glg_list.append(G.DGLGraph(glg))

            g = lg
            dgl_g = dgl_lg

        for module in self.module_list:
            module(pair_list, glg_list)

        return self.linear(th.cat([reprs['x'] for reprs in origin.nodes.values()], 0))
