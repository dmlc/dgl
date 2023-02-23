"""
[Simplifying Graph Convolutional Networks]
(https://arxiv.org/abs/1902.07153)
"""

import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import CoraGraphDataset
from torch.optim import Adam


################################################################################
# (HIGHLIGHT) Take the advantage of DGL sparse APIs to implement the feature
# pre-computation.
################################################################################
def pre_compute(A, X, k):
    for _ in range(k):
        X = A @ X
    return X


def evaluate(g, pred):
    label = g.ndata["label"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    # Compute accuracy on validation/test set.
    val_acc = (pred[val_mask] == label[val_mask]).float().mean()
    test_acc = (pred[test_mask] == label[test_mask]).float().mean()
    return val_acc, test_acc


def train(model, g, X_sgc):
    label = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    optimizer = Adam(model.parameters(), lr=2e-1, weight_decay=5e-6)

    for epoch in range(20):
        # Forward.
        logits = model(X_sgc)

        # Compute loss with nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], label[train_mask])

        # Backward.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute prediction.
        pred = logits.argmax(dim=1)

        # Evaluate the prediction.
        val_acc, test_acc = evaluate(g, pred)
        print(
            f"In epoch {epoch}, loss: {loss:.3f}, val acc: {val_acc:.3f}, test"
            f" acc: {test_acc:.3f}"
        )


if __name__ == "__main__":
    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load graph from the existing dataset.
    dataset = CoraGraphDataset()
    g = dataset[0].to(dev)

    # Create the sparse adjacency matrix A
    indices = torch.stack(g.edges())
    N = g.num_nodes()
    A = dglsp.spmatrix(indices, shape=(N, N))

    # Calculate the symmetrically normalized adjacency matrix.
    I = dglsp.identity(A.shape, device=dev)
    A_hat = A + I
    D_hat = dglsp.diag(A_hat.sum(dim=1)) ** -0.5
    A_hat = D_hat @ A_hat @ D_hat

    # 2-hop diffusion.
    k = 2
    X = g.ndata["feat"]
    X_sgc = pre_compute(A_hat, X, k)

    # Create model.
    in_size = X.shape[1]
    out_size = dataset.num_classes
    model = nn.Linear(in_size, out_size).to(dev)

    # Kick off training.
    train(model, g, X_sgc)
