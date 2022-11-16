"""
[SIGN: Scalable Inception Graph Neural Networks]
(https://arxiv.org/abs/2004.11198)

This example shows a simplified version of SIGN: a precomputed 2-hops diffusion
operators on top of symmetrically normalized adjacency matrix A.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from dgl.data import CoraGraphDataset
from dgl.mock_sparse import create_from_coo, diag, identity

################################################################################
# (HIGHLIGHT) Take the advantage of DGL sparse API to significantly simplify the
# code for sign diffusion operator.
################################################################################
def sign_diffusion(A, X, r):
    # Perform the r-hops diffusion operation.
    X_sign = [X]
    for _ in range(r):
        X = A @ X
        X_sign.append(X)
    return X_sign


class Sign(nn.Module):
    def __init__(self, in_size, out_size, r, hidden_size=64):
        super().__init__()
        self.linear = nn.ModuleList(
            [nn.Linear(in_size, hidden_size) for _ in range(r + 1)]
        )
        self.pred = nn.Linear(hidden_size * (r + 1), out_size)

    def forward(self, X_sign):
        results = []
        for i in range(len(X_sign)):
            results.append(self.linear[i](X_sign[i]))
        Z = F.relu(torch.cat(results, dim=1))
        return F.sigmoid(self.pred(Z))


if __name__ == "__main__":
    # If CUDA is available, use GPU to accelerate the training, otherwise, use
    # CPU for the training.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create graph from the existing dataset.
    dataset = CoraGraphDataset()
    g = dataset[0].to(dev)

    # Create the sparse adjacent matrix A (note that W was used as notion for
    # adjacent matrix in the original paper).
    src, dst = g.edges()
    N = g.num_nodes()
    A = create_from_coo(dst, src, shape=(N, N))

    # 2-hops diffusion operators.
    r = 2
    X = g.ndata["feat"]
    X_sign = sign_diffusion(A, X, r)

    # Create SIGN model.
    in_size = X.shape[1]
    out_size = dataset.num_classes
    model = Sign(in_size, out_size, r)

    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    loss_func = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=3e-3)
    best_val_acc = 0
    best_test_acc = 0

    for e in range(100):
        # Forward.
        logits = model(X_sign).to(dev)

        # Compute prediction.
        pred = logits.argmax(1)

        # Compute loss with nodes in training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test.
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test
        # accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print(
                f"In epoch {e}, loss: {loss:.3f}, val acc: {val_acc:.3f} (best "
                f"{best_val_acc:.3f}), test acc: {test_acc:.3f} (best "
                f"{best_test_acc:.3f})"
            )
