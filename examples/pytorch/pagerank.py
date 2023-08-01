import dgl
import dgl.function as fn
import networkx as nx
import torch

N = 100
network = nx.erdos_renyi_graph(N, 0.05)
g = dgl.from_networkx(network)

DAMP = 0.85
K = 10


def compute_pagerank(g):
    g.ndata["pv"] = torch.ones(N) / N
    degrees = g.out_degrees(g.nodes()).type(torch.float32)
    for k in range(K):
        g.ndata["pv"] = g.ndata["pv"] / degrees
        g.update_all(
            message_func=fn.copy_u(u="pv", out="m"),
            reduce_func=fn.sum(msg="m", out="pv"),
        )
        g.ndata["pv"] = (1 - DAMP) / N + DAMP * g.ndata["pv"]
    return g.ndata["pv"]


pv = compute_pagerank(g)
print(pv)
