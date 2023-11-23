import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx


class LinearNeuralNetwork(nn.Module):
    def __init__(self, nfeat, nclass, bias=True):
        super(LinearNeuralNetwork, self).__init__()
        self.W = nn.Linear(nfeat, nclass, bias=bias)

    def forward(self, x):
        return self.W(x)


def symmetric_normalize_adjacency(graph):
    """Symmetric normalize graph adjacency matrix."""
    indices = torch.stack(graph.edges())
    n = graph.num_nodes()
    adj = dglsp.spmatrix(indices, shape=(n, n))
    deg_invsqrt = dglsp.diag(adj.sum(0)) ** -0.5
    return deg_invsqrt @ adj @ deg_invsqrt


def model_test(model, embedds):
    model.eval()
    with torch.no_grad():
        output = model(embedds)
        pred = output.argmax(dim=-1)
        test_mask, val_mask = model.test_mask, model.val_mask
        loss = F.cross_entropy(output[val_mask], model.label[val_mask])
    accs = []
    for mask in [val_mask, test_mask]:
        accs.append(float((pred[mask] == model.label[mask]).sum()/mask.sum()))
    return loss.item(), accs[0], accs[1]


def inverse_graph_convolution(k, n, device):
    adj = nx.adjacency_matrix(nx.random_regular_graph(k, n)).tocoo()
    indices = torch.tensor([adj.row.tolist(), adj.col.tolist()])
    values = torch.tensor(adj.data.tolist())
    adj_sym_nor = dglsp.spmatrix(indices, values, adj.shape).coalesce().to(device)
    I_N = dglsp.identity((n, n)).to(dtype=torch.int64)
    # re-normalization trick
    adj_sym_nor = dglsp.sub(2 * I_N, adj_sym_nor) / (k + 2)
    return adj_sym_nor


def lazy_random_walk(adj, beta, I_N):
    return dglsp.add((1 - beta) * I_N, beta * adj)
