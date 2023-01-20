"""
[Graph Neural Networks Inspired by Classical Iterative Algorithms]
(https://arxiv.org/pdf/2103.06064.pdf)

This example shows a simplified version of the TWIRLS model proposed
in the paper. It implements two variants. One is the basic iterative
graph diffusion algorithm. The other is an advanced implementation
with attention.
"""

import argparse

import dgl.sparse as dglsp

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import CoraGraphDataset
from torch.optim import Adam


class MLP(nn.Module):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.8)

    def forward(self, X):
        H = self.linear_1(X)
        H = F.relu(H)
        H = self.dropout(H)
        H = self.linear_2(H)
        return H


################################################################################
# (HIGHLIGHT) Use DGL sparse API to implement the iterative graph diffusion
# algorithm.
################################################################################
class TWIRLS(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        hidden_size=128,
        num_steps=16,
        lam=1.0,
        alpha=0.5,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.lam = lam
        self.alpha = alpha
        self.mlp = MLP(in_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, out_size)

    def forward(self, A, X):
        # Compute Y = Y0 = f(X; W) using a two-layer MLP.
        Y = Y0 = self.mlp(X)

        # Compute diagonal matrix D_tild.
        I = dglsp.identity(A.shape, device=A.device)
        D_tild = self.lam * dglsp.diag(A.sum(1)) + I

        # Iteratively compute new Y by equation (6) in the paper.
        for k in range(self.num_steps):
            Y_hat = self.lam * A @ Y + Y0
            # The inverse of a diagonal matrix inverses its diagonal values.
            Y = (1 - self.alpha) * Y + self.alpha * (D_tild**-1) @ Y_hat

        # Apply a linear layer on the final output.
        return self.linear_out(Y)


################################################################################
# (HIGHLIGHT) Implementation of the advanced TWIRLS model with attention
# to show the usage of differentiable weighted sparse matrix.
################################################################################
class TWIRLSWithAttention(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        hidden_size=128,
        num_steps=16,
        lam=1.0,
        alpha=0.5,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.lam = lam
        self.alpha = alpha
        self.mlp = MLP(in_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, out_size)

    def forward(self, A, X):
        # Compute Y = Y0 = f(X; W) using a two-layer MLP.
        Y = Y0 = self.mlp(X)

        # Compute diagonal matrix D_tild.
        I = dglsp.identity(A.shape, device=A.device)
        D_tild = self.lam * dglsp.diag(A.sum(1)) + I

        # Conduct half of the diffusion steps.
        for k in range(self.num_steps // 2):
            Y_hat = self.lam * A @ Y + Y0
            Y = (1 - self.alpha) * Y + self.alpha * (D_tild**-1) @ Y_hat

        # Calculate attention weight by equation (25) in the paper.
        Y_i = Y[A.row]
        Y_j = Y[A.col]
        norm_ij = torch.linalg.vector_norm(Y_i - Y_j, dim=1)
        # Bound the attention value within [0.0, 1.0).
        gamma_ij = torch.clamp(0.5 / (norm_ij + 1e-7), min=0.0, max=1.0)
        # Create a new adjacency matrix with the new weight.
        A = dglsp.val_like(A, gamma_ij)
        # Recompute D_tild.
        D_tild = self.lam * dglsp.diag(A.sum(1)) + I

        # Conduct the other half of the diffusion steps.
        for k in range(self.num_steps // 2):
            Y_hat = self.lam * A @ Y + Y0
            Y = (1 - self.alpha) * Y + self.alpha * (D_tild**-1) @ Y_hat

        # Apply a linear layer on the final output.
        return self.linear_out(Y)


def evaluate(g, pred):
    model.eval()
    label = g.ndata["label"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    # Compute accuracy on validation/test set.
    val_acc = (pred[val_mask] == label[val_mask]).float().mean()
    test_acc = (pred[test_mask] == label[test_mask]).float().mean()
    return val_acc, test_acc


def train(g, model, A, X):
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    optimizer = Adam(model.parameters(), lr=5e-4)

    for epoch in range(300):
        model.train()
        # Forward.
        logits = model(A, X)

        # Compute loss with nodes in training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Backward.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute prediction.
        pred = logits.argmax(1)

        # Evaluate the prediction.
        val_acc, test_acc = evaluate(g, pred)
        print(
            f"In epoch {epoch}, loss: {loss:.3f}, val acc: {val_acc:.3f}, test"
            f" acc: {test_acc:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TWIRLS example in DGL Sparse.")
    parser.add_argument(
        "--attention", action="store_true", help="Use TWIRLS with attention."
    )
    args = parser.parse_args()
    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load graph from the existing dataset.
    dataset = CoraGraphDataset()
    g = dataset[0].to(dev)
    X = g.ndata["feat"]

    # Create the sparse adjacency matrix A.
    indices = torch.stack(g.edges())
    N = g.num_nodes()
    A = dglsp.spmatrix(indices, shape=(N, N))

    # Create the TWIRLS model.
    in_size = X.shape[1]
    out_size = dataset.num_classes
    if args.attention:
        model = TWIRLSWithAttention(in_size, out_size).to(dev)
    else:
        model = TWIRLS(in_size, out_size).to(dev)

    # Kick off training.
    train(g, model, A, X)
