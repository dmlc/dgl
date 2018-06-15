from __future__ import division

import networkx as nx
from dgl.graph import DGLGraph

DAMP = 0.85
N = 100
K = 10

def message_func(src, dst, edge):
    return src['pv'] / src['deg']

def update_func(node, msgs):
    pv = (1 - DAMP) / N + DAMP * sum(msgs)
    return {'pv' : pv}

def compute_pagerank(g):
    g = DGLGraph(g)
    print(g.number_of_edges(), g.number_of_nodes())
    g.register_message_func(message_func)
    g.register_update_func(update_func)
    # init pv value
    for n in g.nodes():
        g.node[n]['pv'] = 1 / N
        g.node[n]['deg'] = g.out_degree(n)
    # pagerank
    for k in range(K):
        g.update_all()
    return [g.node[n]['pv'] for n in g.nodes()]

if __name__ == '__main__':
    g = nx.erdos_renyi_graph(N, 0.05)
    pv = compute_pagerank(g)
    print(pv)
