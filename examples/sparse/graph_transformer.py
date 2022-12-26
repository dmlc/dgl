"""
[A Generalization of Transformer Networks to Graphs]
(https://arxiv.org/abs/2012.09699)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import CoraGraphDataset
from dgl.mock_sparse import create_from_coo, bspmm, mock_bsddmm as bsddmm
from torch.optim import Adam


class SparseMHA(nn.Module):
    """Sparse Multi-head Attention Module"""
    def __init__(self, hidden_size=80, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, A, h):
        N = len(h)
        q = self.q_proj(h).reshape(N, self.num_heads, self.head_dim) \
            * self.scaling
        k = self.k_proj(h).reshape(N, self.num_heads, self.head_dim)
        v = self.v_proj(h).reshape(N, self.num_heads, self.head_dim)

        ######################################################################
        # (HIGHLIGHT) Compute the multi-head attention with Sparse Matrix API
        ######################################################################
        attn = bsddmm(A, q.permute(1, 0, 2), k.permute(1, 2, 0))  # [N, N, nh]
        attn = attn.softmax()
        out = bspmm(attn, v.permute(0, 2, 1)).permute(0, 2, 1)

        return self.out_proj(out.reshape(N, -1))


class GTLayer(nn.Module):
    """Graph Transformer Layer"""
    def __init__(self, hidden_size=80, num_heads=8) -> None:
        super().__init__()
        self.MHA = SparseMHA(hidden_size=hidden_size, num_heads=num_heads)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.FFN1 = nn.Linear(hidden_size, hidden_size * 2)
        self.FFN2 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, A, h):
        h1 = h
        h = self.MHA(A, h)
        h = self.batchnorm1(h + h1)

        h2 = h
        h = self.FFN2(F.relu(self.FFN1(h)))
        h = h2 + h

        return self.batchnorm2(h)


class GTModel(nn.Module):
    def __init__(
        self, in_size, out_size, hidden_size=80, pos_enc_size=2,
        num_layers=4, num_heads=8
    ):
        super().__init__()
        self.h_linear = nn.Linear(in_size, hidden_size)
        self.pos_linear = nn.Linear(pos_enc_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, out_size)
        self.layers = nn.ModuleList([
            GTLayer(hidden_size, num_heads) for _ in range(num_layers)
        ])

    def forward(self, A, X, pos_enc):
        h = self.h_linear(X) + self.pos_linear(pos_enc)
        for layer in self.layers:
            h = layer(A, h)
        
        return self.out_linear(h)


def evaluate(g, pred):
    label = g.ndata["label"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    # Compute accuracy on validation/test set.
    val_acc = (pred[val_mask] == label[val_mask]).float().mean()
    test_acc = (pred[test_mask] == label[test_mask]).float().mean()
    return val_acc, test_acc


def train(model, g, A, X, pos_enc):
    label = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    optimizer = Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    loss_fcn = nn.CrossEntropyLoss()

    for epoch in range(200):
        model.train()

        # Forward.
        logits = model(A, X, pos_enc)

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
    # laplacian positional encoding
    pos_enc = dgl.laplacian_pe(g, 2).to(dev)

    # Create the adjacency matrix of graph.
    src, dst = g.edges()
    N = g.num_nodes()
    A = create_from_coo(dst, src, shape=(N, N))

    # Create model.
    in_size = X.shape[1]
    out_size = num_classes
    model = GTModel(in_size, out_size).to(dev)

    # Kick off training.
    train(model, g, A, X, pos_enc)
