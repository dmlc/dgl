from collections import defaultdict
import dgl
import networkx as nx
import numpy as np
import scipy.sparse as ssp
import backend as F

case_registry = defaultdict(list)

def register_case(labels):
    def wrapper(fn):
        for lbl in labels:
            case_registry[lbl].append(fn)
        fn.__labels__ = labels
        return fn
    return wrapper

def get_cases(labels=None, exclude=[]):
    """Get all graph instances of the given labels."""
    cases = set()
    if labels is None:
        # get all the cases
        labels = case_registry.keys()
    for lbl in labels:
        for case in case_registry[lbl]:
            if not any([l in exclude for l in case.__labels__]):
                cases.add(case)
    return [fn() for fn in cases]

@register_case(['dglgraph', 'path', 'small'])
def dglgraph_path():
    return dgl.DGLGraph(nx.path_graph(5))

@register_case(['bipartite', 'small', 'hetero', 'zero-degree'])
def bipartite1():
    return dgl.bipartite([(0, 0), (0, 1), (0, 4), (2, 1), (2, 4), (3, 3)])

@register_case(['bipartite', 'small', 'hetero'])
def bipartite_full():
    return dgl.bipartite([(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)])

@register_case(['homo', 'small'])
def graph0():
    return dgl.graph(([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 6, 6, 7, 8, 9],
                      [4, 5, 1, 2, 4, 7, 9, 8 ,6, 4, 1, 0, 1, 0, 2, 3, 5]))

@register_case(['homo', 'small', 'has_feature'])
def graph1():
    g = dgl.graph(([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 6, 6, 7, 8, 9],
                   [4, 5, 1, 2, 4, 7, 9, 8 ,6, 4, 1, 0, 1, 0, 2, 3, 5]))
    g.ndata['h'] = F.randn((g.number_of_nodes(), 2))
    g.edata['w'] = F.randn((g.number_of_edges(), 3))
    return g

@register_case(['hetero', 'small', 'has_feature'])
def heterograph0():
    g = dgl.heterograph({
        ('user', 'plays', 'game'): (F.tensor([0, 1, 1, 2]), F.tensor([0, 0, 1, 1])),
        ('developer', 'develops', 'game'): (F.tensor([0, 1]), F.tensor([0, 1]))})
    g.nodes['user'].data['h'] = F.randn((g.number_of_nodes('user'), 3))
    g.nodes['game'].data['h'] = F.randn((g.number_of_nodes('game'), 2))
    g.nodes['developer'].data['h'] = F.randn((g.number_of_nodes('developer'), 3))
    g.edges['plays'].data['h'] = F.randn((g.number_of_edges('plays'), 1))
    g.edges['develops'].data['h'] = F.randn((g.number_of_edges('develops'), 5))
    return g


@register_case(['batched', 'homo', 'small'])
def batched_graph0():
    g1 = dgl.graph(([0, 1, 2], [1, 2, 3]))
    g2 = dgl.graph(([1, 1], [2, 0]))
    g3 = dgl.graph(([0], [1]))
    return dgl.batch([g1, g2, g3])

@register_case(['block'])
def block_graph0():
    g = dgl.graph(([2, 3, 4], [5, 6, 7]), num_nodes=100)
    return dgl.to_block(g)

@register_case(['block'])
def block_graph1():
    g = dgl.heterograph({
            ('user', 'plays', 'game') : ([0, 1, 2], [1, 1, 0]),
            ('user', 'likes', 'game') : ([1, 2, 3], [0, 0, 2]),
            ('store', 'sells', 'game') : ([0, 1, 1], [0, 1, 2]),
        })
    return dgl.to_block(g)

def random_dglgraph(size):
    return dgl.DGLGraph(nx.erdos_renyi_graph(size, 0.3))

def random_graph(size):
    return dgl.graph(nx.erdos_renyi_graph(size, 0.3))

def random_bipartite(size_src, size_dst):
    return dgl.bipartite(ssp.random(size_src, size_dst, 0.1))
