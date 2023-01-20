"""
Hypergraph Neural Networks (https://arxiv.org/pdf/1809.09401.pdf)
"""
import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.data import CoraGraphDataset
from torchmetrics.functional import accuracy


class HGNN(nn.Module):
    def __init__(self, H, in_size, out_size, hidden_dims=16):
        super().__init__()

        self.Theta1 = nn.Linear(in_size, hidden_dims)
        self.Theta2 = nn.Linear(hidden_dims, out_size)
        self.dropout = nn.Dropout(0.5)

        ###########################################################
        # (HIGHLIGHT) Compute the Laplacian with Sparse Matrix API
        ###########################################################
        d_V = H.sum(1)  # node degree
        d_E = H.sum(0)  # edge degree
        n_edges = d_E.shape[0]
        D_V_invsqrt = dglsp.diag(d_V**-0.5)  # D_V ** (-1/2)
        D_E_inv = dglsp.diag(d_E**-1)  # D_E ** (-1)
        W = dglsp.identity((n_edges, n_edges))
        self.laplacian = D_V_invsqrt @ H @ W @ D_E_inv @ H.T @ D_V_invsqrt

    def forward(self, X):
        X = self.laplacian @ self.Theta1(self.dropout(X))
        X = F.relu(X)
        X = self.laplacian @ self.Theta2(self.dropout(X))
        return X


def train(model, optimizer, X, Y, train_mask):
    model.train()
    Y_hat = model(X)
    loss = F.cross_entropy(Y_hat[train_mask], Y[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def evaluate(model, X, Y, val_mask, test_mask, num_classes):
    model.eval()
    Y_hat = model(X)
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


def main():
    H, X, Y, num_classes, train_mask, val_mask, test_mask = load_data()
    model = HGNN(H, X.shape[1], num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    with tqdm.trange(500) as tq:
        for epoch in tq:
            train(model, optimizer, X, Y, train_mask)
            val_acc, test_acc = evaluate(
                model, X, Y, val_mask, test_mask, num_classes
            )
            tq.set_postfix(
                {
                    "Val acc": f"{val_acc:.5f}",
                    "Test acc": f"{test_acc:.5f}",
                },
                refresh=False,
            )

    print(f"Test acc: {test_acc:.3f}")


if __name__ == "__main__":
    main()
