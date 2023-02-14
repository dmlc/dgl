"""
[Semi-Supervised Classification with Graph Convolutional Networks]
(https://arxiv.org/abs/1609.02907)
"""

import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import CoraGraphDataset
from torch.optim import Adam


class GCN(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=16):
        super().__init__()

        # Two-layer GCN.
        self.W1 = nn.Linear(in_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, out_size)

    ############################################################################
    # (HIGHLIGHT) Take the advantage of DGL sparse APIs to implement the GCN
    # forward process.
    ############################################################################
    def forward(self, A_norm, X):
        X = A_norm @ self.W1(X)
        X = F.relu(X)
        X = A_norm @ self.W2(X)
        return X


def evaluate(g, pred):
    label = g.ndata["label"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    # Compute accuracy on validation/test set.
    val_acc = (pred[val_mask] == label[val_mask]).float().mean()
    test_acc = (pred[test_mask] == label[test_mask]).float().mean()
    return val_acc, test_acc


def train(model, g, A_norm, X):
    label = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    optimizer = Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    loss_fcn = nn.CrossEntropyLoss()

    for epoch in range(200):
        model.train()

        # Forward.
        logits = model(A_norm, X)

        # Compute loss with nodes in the training set.
        loss = loss_fcn(logits[train_mask], label[train_mask])

        # Backward.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute prediction.
        pred = logits.argmax(dim=1)

        # Evaluate the prediction.
        val_acc, test_acc = evaluate(g, pred)
        if epoch % 20 == 0:
            print(
                f"In epoch {epoch}, loss: {loss:.3f}, val acc: {val_acc:.3f}"
                f", test acc: {test_acc:.3f}"
            )


if __name__ == "__main__":
    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load graph from the existing dataset.
    dataset = CoraGraphDataset()
    g = dataset[0].to(dev)
    num_classes = dataset.num_classes
    X = g.ndata["feat"]

    # Create the adjacency matrix of graph.
    indices = torch.stack(g.edges())
    N = g.num_nodes()
    A = dglsp.spmatrix(indices, shape=(N, N))

    ############################################################################
    # (HIGHLIGHT) Compute the symmetrically normalized adjacency matrix with
    # Sparse Matrix API
    ############################################################################
    I = dglsp.identity(A.shape, device=dev)
    A_hat = A + I
    D_hat = dglsp.diag(A_hat.sum(1)) ** -0.5
    A_norm = D_hat @ A_hat @ D_hat

    # Create model.
    in_size = X.shape[1]
    out_size = num_classes
    model = GCN(in_size, out_size).to(dev)

    # Kick off training.
    train(model, g, A_norm, X)
