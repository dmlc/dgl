import networkx as nx
import torch

import dgl.sparse as dglsp

N = 100
DAMP = 0.85
K = 10


if __name__ == "__main__":
    g = nx.erdos_renyi_graph(N, 0.05, seed=10086)

    # Create the adjacency matrix of graph.
    edges = list(g.to_directed().edges())
    indices = torch.tensor(edges).transpose(0, 1)
    A = dglsp.spmatrix(indices, shape=(N, N))

    D = A.sum(0)
    PV = torch.ones(N) / N
    for _ in range(K):
        ########################################################################
        # (HIGHLIGHT) Take the advantage of DGL sparse APIs to calculate the
        # page rank.
        ########################################################################
        PV = (1 - DAMP) / N + DAMP * A @ (PV / D)

    print(PV)
