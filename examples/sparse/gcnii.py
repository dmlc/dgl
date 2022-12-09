"""
[Simple and Deep Graph Convolutional Networks]
(https://arxiv.org/abs/2007.02133)
"""

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import CoraGraphDataset
from dgl.mock_sparse import create_from_coo, diag, identity, spmm
from torch.nn.parameter import Parameter
from torch.optim import Adam


class GCNIIConvolution(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.out_size = out_size
        self.weight = Parameter(torch.FloatTensor(in_size, out_size))
        self.reset_parameters()

    ############################################################################
    # (HIGHLIGHT) Take the advantage of DGL sparse APIs to implement the GCNII 
    # forward process.
    ############################################################################
    def forward(self, A_norm, H, H0, lamda, alpha, l):
        beta = math.log(lamda / l + 1)
        H = spmm(A_norm, H)
        support = (1 - alpha) * H + alpha * H0
        H = (1 - beta) * support + beta * support @ self.weight
        return H

    def reset_parameters(self):
        std = 1 / math.sqrt(self.out_size)
        self.weight.data.uniform_(-std, std)


class GCNII(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        hidden_size,
        n_layers,
        lamda,
        alpha,
        dropout=0.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lamda = lamda
        self.alpha = alpha

        self.FC_layers = nn.ModuleList()
        self.FC_layers.append(nn.Linear(in_size, hidden_size))
        self.FC_layers.append(nn.Linear(hidden_size, out_size))

        self.CONV_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.CONV_layers.append(GCNIIConvolution(hidden_size, hidden_size))
        self.activation = nn.ReLU()
        self.dropout = dropout

    def forward(self, A_norm, feature):
        H = feature
        H = F.dropout(H, self.dropout)
        H = self.FC_layers[0](H)
        H = self.activation(H)
        H0 = H
        for i, conv in enumerate(self.CONV_layers):
            H = F.dropout(H, self.dropout)
            H = conv(A_norm, H, H0, self.lamda, self.alpha, i + 1)
            H = self.activation(H)
        H = F.dropout(H, self.dropout)
        H = self.FC_layers[-1](H)

        return F.log_softmax(H, dim=1)


def evaluate(g, pred):
    label = g.ndata["label"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    # Compute accuracy on validation/test set.
    val_acc = (pred[val_mask] == label[val_mask]).float().mean()
    test_acc = (pred[test_mask] == label[test_mask]).float().mean()
    return val_acc, test_acc


def train(args, model, g, A_norm, H):
    label = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    optimizer = Adam(model.parameters(), args.lr, weight_decay=args.wd)

    loss_fcn = nn.NLLLoss()

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        # Forward.
        logits = model(A_norm, H)

        # Compute loss with nodes in the training set.
        loss = loss_fcn(logits[train_mask], label[train_mask])

        # Backward.
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
    # Training settings
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--epochs", type=int, default=1500)
    argparse.add_argument("--lr", type=float, default=0.01)
    argparse.add_argument("--wd", type=float, default=5e-4)
    argparse.add_argument("--n_layers", type=int, default=64)
    argparse.add_argument("--hidden_size", type=int, default=64)
    argparse.add_argument("--dropout", type=float, default=0.5)
    argparse.add_argument("--alpha", type=float, default=0.2)
    argparse.add_argument("--lamda", type=float, default=0.5)
    args = argparse.parse_args()

    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load graph from the existing dataset.
    dataset = CoraGraphDataset()
    g = dataset[0].to(dev)
    num_classes = dataset.num_classes
    H = g.ndata["feat"]

    # Create the adjacency matrix of graph.
    src, dst = g.edges()
    N = g.num_nodes()
    A = create_from_coo(dst, src, shape=(N, N))

    ############################################################################
    # (HIGHLIGHT) Compute the symmetrically normalized adjacency matrix with 
    # Sparse Matrix API
    ############################################################################
    I = identity(A.shape, device=dev)
    A_hat = A + I
    D_hat = diag(A_hat.sum(1)) ** -0.5
    A_norm = D_hat @ A_hat @ D_hat

    # Create model.
    in_size = H.shape[1]
    out_size = num_classes
    model = GCNII(
        in_size,
        out_size,
        args.hidden_size,
        args.n_layers,
        args.lamda,
        args.alpha,
        args.dropout,
    ).to(dev)

    # Kick off training.
    train(args, model, g, A_norm, H)
