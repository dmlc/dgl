import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl import AddSelfLoop
import argparse

from dgl.mock_sparse import create_from_coo, softmax, bspmm
from torch.nn import init


class GATConv(nn.Module):
    def __init__(self, in_size, out_size, n_heads):
        super(GATConv, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.n_heads = n_heads
        self.W = nn.Parameter(torch.Tensor(in_size, out_size * n_heads))
        self.a_l = nn.Parameter(torch.Tensor(1, n_heads, out_size))
        self.a_r = nn.Parameter(torch.Tensor(1, n_heads, out_size))
        self.leaky_relu = nn.LeakyReLU(0.2)
        init.xavier_uniform_(self.W)
        init.xavier_uniform_(self.a_l)
        init.xavier_uniform_(self.a_r)

    def forward(self, A, h):
        Wh = (h @ self.W).view(
            -1, self.n_heads, self.out_size
        )  # |V| x N_h x D_o
        Wh1 = (Wh * self.a_l).sum(2)  # |V| x N_h
        Wh2 = (Wh * self.a_r).sum(2)  # |V| x N_h
        Wh1 = Wh1[A.row, :]  # |E| x N_h
        Wh2 = Wh2[A.col, :]  # |E| x N_h
        e = Wh1 + Wh2  # |E| x N_h
        e = self.leaky_relu(e)  # |E| x N_h
        A = create_from_coo(
            A.row, A.col, e, A.shape
        )  # |V| x |V| x N_h SparseMatrix
        A_hat = softmax(A)  # |V| x |V| x N_h SparseMatrix
        Wh = Wh.reshape(-1, self.out_size, self.n_heads)  # |V| x D_o x N_h
        h_prime = bspmm(A_hat, Wh)  # |V| x D_o x N_h

        return torch.relu(h_prime)


class GAT(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_heads):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_size, hidden_size, n_heads))
        self.layers.append(GATConv(hidden_size * n_heads, out_size, n_heads))

    def forward(self, A, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(A, h)
            if i == 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h


def evaluate(A, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(A, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(A, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training loop
    for epoch in range(50):
        model.train()
        logits = model(A, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(A, features, labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'citeseer', 'pubmed').",
    )
    args = parser.parse_args()
    print(f"Training with DGL SparseMatrix GATConv module.")

    # load and preprocess dataset
    transform = (
        AddSelfLoop()
    )  # by default, it will first remove self-loops to prevent duplication
    if args.dataset == "cora":
        data = CoraGraphDataset(transform=transform)
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset(transform=transform)
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset(transform=transform)
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    g = data[0]
    g = g.int()
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]

    row, col = g.adj_sparse("coo")
    A = create_from_coo(
        row, col, shape=(g.number_of_nodes(), g.number_of_nodes())
    )

    # create GAT model
    in_size = features.shape[1]
    out_size = data.num_classes
    model = GAT(in_size, 8, out_size, 8)

    # model training
    print("Training...")
    train(A, features, labels, masks, model)

    # test the model
    print("Testing...")
    acc = evaluate(A, features, labels, masks[2], model)
    print("Test accuracy {:.4f}".format(acc))
