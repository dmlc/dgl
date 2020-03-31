from collections import defaultdict
import dgl
import networkx as nx
import scipy.sparse as ssp

case_registry = defaultdict(list)

def register_case(labels):
    def wrapper(fn):
        for lbl in labels:
            case_registry[lbl].append(fn)
        return fn
    return wrapper

def get_cases(labels=None, exclude=None):
    cases = set()
    if labels is None:
        # get all the cases
        labels = case_registry.keys()
    for lbl in labels:
        if exclude is not None and lbl in exclude:
            continue
        cases.update(case_registry[lbl])
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

def random_dglgraph(size):
    return dgl.DGLGraph(nx.erdos_renyi_graph(size, 0.3))

def random_graph(size):
    return dgl.graph(nx.erdos_renyi_graph(size, 0.3))

def random_bipartite(size_src, size_dst):
    return dgl.bipartite(ssp.random(size_src, size_dst, 0.1))
