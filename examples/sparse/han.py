"""
[Heterogeneous Graph Attention Network]
(https://arxiv.org/abs/1903.07293)
"""

import pickle

import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.data.utils import _get_dgl_url, download, get_download_dir
from torch.optim import Adam


class GATConv(nn.Module):
    def __init__(self, in_size, out_size, num_heads, dropout):
        super().__init__()

        self.out_size = out_size
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.W = nn.Linear(in_size, out_size * num_heads)
        self.a_l = nn.Parameter(torch.zeros(1, out_size, num_heads))
        self.a_r = nn.Parameter(torch.zeros(1, out_size, num_heads))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.W.weight, gain=gain)
        nn.init.xavier_normal_(self.a_l, gain=gain)
        nn.init.xavier_normal_(self.a_r, gain=gain)

    ###########################################################################
    # (HIGHLIGHT) Take the advantage of DGL sparse APIs to implement
    # multihead attention.
    ###########################################################################
    def forward(self, A_hat, Z):
        Z = self.dropout(Z)
        Z = self.W(Z).view(Z.shape[0], self.out_size, self.num_heads)

        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        e_l = (Z * self.a_l).sum(dim=1)
        e_r = (Z * self.a_r).sum(dim=1)
        e = e_l[A_hat.row] + e_r[A_hat.col]

        a = F.leaky_relu(e)
        A_atten = dglsp.val_like(A_hat, a).softmax()
        a_drop = self.dropout(A_atten.val)
        A_atten = dglsp.val_like(A_atten, a_drop)
        return dglsp.bspmm(A_atten, Z)


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super().__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)

        return (beta * z).sum(1)


class HAN(nn.Module):
    def __init__(
        self,
        num_meta_paths,
        in_size,
        out_size,
        hidden_size=8,
        num_heads=8,
        dropout=0.6,
    ):
        super().__init__()

        self.gat_layers = nn.ModuleList()
        for _ in range(num_meta_paths):
            self.gat_layers.append(
                GATConv(in_size, hidden_size, num_heads, dropout)
            )

        in_size = hidden_size * num_heads
        self.semantic_attention = SemanticAttention(in_size)
        self.predict = nn.Linear(in_size, out_size)

    def forward(self, A_list, X):
        meta_path_Z_list = []
        for i, A in enumerate(A_list):
            meta_path_Z_list.append(self.gat_layers[i](A, X).flatten(1))

        # (num_nodes, num_meta_paths, hidden_size * num_heads)
        meta_path_Z = torch.stack(meta_path_Z_list, dim=1)

        Z = self.semantic_attention(meta_path_Z)
        Z = self.predict(Z)

        return Z


def evaluate(label, val_idx, test_idx, pred):
    # Compute accuracy on validation/test set.
    val_acc = (pred[val_idx] == label[val_idx]).float().mean()
    test_acc = (pred[test_idx] == label[test_idx]).float().mean()
    return val_acc, test_acc


def train(model, data, A_list, X, label):
    dev = X.device
    train_idx = torch.from_numpy(data["train_idx"]).long().squeeze(0).to(dev)
    val_idx = torch.from_numpy(data["val_idx"]).long().squeeze(0).to(dev)
    test_idx = torch.from_numpy(data["test_idx"]).long().squeeze(0).to(dev)
    optimizer = Adam(model.parameters(), lr=0.005, weight_decay=0.001)

    for epoch in range(70):
        # Forward.
        model.train()
        logits = model(A_list, X)

        # Compute loss with nodes in training set.
        loss = F.cross_entropy(logits[train_idx], label[train_idx])

        # Backward.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute prediction.
        model.eval()
        logits = model(A_list, X)
        pred = logits.argmax(dim=1)

        # Evaluate the prediction.
        val_acc, test_acc = evaluate(label, val_idx, test_idx, pred)
        print(
            f"In epoch {epoch}, loss: {loss:.3f}, val acc: {val_acc:.3f}, test"
            f" acc: {test_acc:.3f}"
        )


if __name__ == "__main__":
    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # (TODO): Move the logic to a built-in dataset.
    # Load the data.
    url = "dataset/ACM3025.pkl"
    data_path = get_download_dir() + "/ACM3025.pkl"
    download(_get_dgl_url(url), path=data_path)

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # Create sparse adjacency matrices corresponding to two meta paths.
    # Self-loops already added.
    PAP_dst, PAP_src = data["PAP"].nonzero()
    PAP_indices = torch.stack(
        [torch.from_numpy(PAP_src).long(), torch.from_numpy(PAP_dst).long()]
    ).to(dev)
    PAP_A = dglsp.spmatrix(PAP_indices)

    PLP_dst, PLP_src = data["PLP"].nonzero()
    PLP_indices = torch.stack(
        [torch.from_numpy(PLP_src).long(), torch.from_numpy(PLP_src).long()]
    ).to(dev)
    PLP_A = dglsp.spmatrix(PLP_indices)
    A_list = [PAP_A, PLP_A]

    # Create HAN model.
    X = torch.from_numpy(data["feature"].todense()).float().to(dev)
    label = torch.from_numpy(data["label"].todense())
    out_size = label.shape[1]
    label = label.nonzero()[:, 1].to(dev)
    in_size = X.shape[1]
    model = HAN(len(A_list), in_size, out_size).to(dev)

    # Kick off training.
    train(model, data, A_list, X, label)
