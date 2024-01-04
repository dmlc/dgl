import dgl
import dgl.sparse as dglsp
import networkx as nx
import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, in_feats, n_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_feats, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, x):
        return self.fc(x)


def symmetric_normalize_adjacency(graph):
    """Symmetric normalize graph adjacency matrix."""
    indices = torch.stack(graph.edges())
    n = graph.num_nodes()
    adj = dglsp.spmatrix(indices, shape=(n, n))
    deg_invsqrt = dglsp.diag(adj.sum(0)) ** -0.5
    return deg_invsqrt @ adj @ deg_invsqrt


def inverse_graph_convolution(edge_num, node_num, I_N):
    graph = dgl.from_networkx(nx.random_regular_graph(edge_num, node_num))
    indices = torch.stack(graph.edges())
    adj = dglsp.spmatrix(indices, shape=(node_num, node_num)).coalesce()

    # re-normalization trick
    adj_sym_nor = dglsp.sub(2 * I_N, adj) / (edge_num + 2)
    return adj_sym_nor


def lazy_random_walk(adj, beta, I_N):
    return dglsp.add((1 - beta) * I_N, beta * adj)
