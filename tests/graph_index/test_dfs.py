import random
import sys
import time

import dgl
import dgl.backend as F
import dgl.utils as utils
import igraph
import networkx as nx
import numpy as np
import scipy.sparse as sp

'''
g_nx = nx.DiGraph([(0, 1), (1, 2), (2, 1)])
g = dgl.DGLGraph()
g.from_networkx(g_nx)
src = [0]
'''

n = int(sys.argv[1])
g_nx = nx.generators.trees.random_tree(n)
g = dgl.DGLGraph()
g.from_networkx(g_nx)
src = [random.choice(range(g.number_of_nodes()))]

src_tuple, dst_tuple, type_tuple = list(zip(*g._graph.dfs_labeled_edges(src, out=True, reverse_edge=True, nontree_edge=True)))
src_dgl = F.pack(src_tuple)
dst_dgl = F.pack(dst_tuple)
type_dgl = F.pack(type_tuple)

src_nx, dst_nx, type_nx = zip(*nx.dfs_labeled_edges(g_nx, src[0]))

dict_dgl = dict(zip(zip(src_dgl.numpy(), dst_dgl.numpy()), type_dgl.numpy()))
dict_nx = dict(zip(zip(src_nx, dst_nx), type_nx))
dict_nx = {k : v for k, v in dict_nx.items() if k[0] != k[1]}

assert len(dict_dgl) == len(dict_nx)
gi = g._graph
mapping = {gi.FORWARD : 'forward', gi.REVERSE : 'reverse', gi.NONTREE : 'nontree'}
assert all(mapping[v] == dict_nx[k] for k, v in dict_dgl.items())
