"""
Hypergraph Convolution and Hypergraph Attention
(https://arxiv.org/pdf/1901.08150.pdf).
"""
import argparse

import dgl.sparse as dglsp

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.data import CoraGraphDataset
from torchmetrics.functional import accuracy


def hypergraph_laplacian(H):
    ###########################################################
    # (HIGHLIGHT) Compute the Laplacian with Sparse Matrix API
    ###########################################################
    d_V = H.sum(1)  # node degree
    d_E = H.sum(0)  # edge degree
    n_edges = d_E.shape[0]
    D_V_invsqrt = dglsp.diag(d_V**-0.5)  # D_V ** (-1/2)
    D_E_inv = dglsp.diag(d_E**-1)  # D_E ** (-1)
    W = dglsp.identity((n_edges, n_edges))
    return D_V_invsqrt @ H @ W @ D_E_inv @ H.T @ D_V_invsqrt


class HypergraphAttention(nn.Module):
    """Hypergraph Attention module as in the paper
    `Hypergraph Convolution and Hypergraph Attention
    <https://arxiv.org/pdf/1901.08150.pdf>`_.
    """

    def __init__(self, in_size, out_size):
        super().__init__()

        self.P = nn.Linear(in_size, out_size)
        self.a = nn.Linear(2 * out_size, 1)

    def forward(self, H, X, X_edges):
        Z = self.P(X)
        Z_edges = self.P(X_edges)
        sim = self.a(torch.cat([Z[H.row], Z_edges[H.col]], 1))
        sim = F.leaky_relu(sim, 0.2).squeeze(1)
        # Reassign the hypergraph new weights.
        H_att = dglsp.val_like(H, sim)
        H_att = H_att.softmax()
        return hypergraph_laplacian(H_att) @ Z


class Net(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=16):
        super().__init__()

        self.layer1 = HypergraphAttention(in_size, hidden_size)
        self.layer2 = HypergraphAttention(hidden_size, out_size)

    def forward(self, H, X):
        Z = self.layer1(H, X, X)
        Z = F.elu(Z)
        Z = self.layer2(H, Z, Z)
        return Z


def train(model, optimizer, H, X, Y, train_mask):
    model.train()
    Y_hat = model(H, X)
    loss = F.cross_entropy(Y_hat[train_mask], Y[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, H, X, Y, val_mask, test_mask, num_classes):
    model.eval()
    Y_hat = model(H, X)
    val_acc = accuracy(
        Y_hat[val_mask], Y[val_mask], task="multiclass", num_classes=num_classes
    )
    test_acc = accuracy(
        Y_hat[test_mask],
        Y[test_mask],
        task="multiclass",
        num_classes=num_classes,
    )
    return val_acc, test_acc


def load_data():
    dataset = CoraGraphDataset()

    graph = dataset[0]
    # The paper created a hypergraph from the original graph. For each node in
    # the original graph, a hyperedge in the hypergraph is created to connect
    # its neighbors and itself. In this case, the incidence matrix of the
    # hypergraph is the same as the adjacency matrix of the original graph (with
    # self-loops).
    # We follow the paper and assume that the rows of the incidence matrix
    # are for nodes and the columns are for edges.
    indices = torch.stack(graph.edges())
    H = dglsp.spmatrix(indices)
    H = H + dglsp.identity(H.shape)

    X = graph.ndata["feat"]
    Y = graph.ndata["label"]
    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]
    return H, X, Y, dataset.num_classes, train_mask, val_mask, test_mask


def main(args):
    H, X, Y, num_classes, train_mask, val_mask, test_mask = load_data()
    model = Net(X.shape[1], num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    with tqdm.trange(args.epochs) as tq:
        for epoch in tq:
            loss = train(model, optimizer, H, X, Y, train_mask)
            val_acc, test_acc = evaluate(
                model, H, X, Y, val_mask, test_mask, num_classes
            )
            tq.set_postfix(
                {
                    "Loss": f"{loss:.5f}",
                    "Val acc": f"{val_acc:.5f}",
                    "Test acc": f"{test_acc:.5f}",
                },
                refresh=False,
            )

    print(f"Test acc: {test_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hypergraph Attention Example")
    parser.add_argument(
        "--epochs", type=int, default=500, help="Number of training epochs."
    )
    args = parser.parse_args()
    main(args)
