import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl import AddSelfLoop
from torch.nn import init

from dgl.mock_sparse import create_from_coo, diag, identity


class GraphConv(nn.Module):
    def __init__(self, in_size, out_size, activation=None):
        super(GraphConv, self).__init__()
        self.W = nn.Parameter(torch.Tensor(in_size, out_size))
        self.activation = activation
        self.bias = nn.Parameter(torch.Tensor(out_size))

        self.reset_parameters()

    def forward(self, A, x):
        h = x @ self.W  # Dense mm, pytorch op
        h = A @ h       # SpMM
        h += self.bias

        if self.activation:
            h = self.activation(h)
        return h

    def reset_parameters(self):
        init.xavier_uniform_(self.W)
        init.zeros_(self.bias)


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(GraphConv(in_size, hid_size, activation=F.relu))
        self.layers.append(GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, A, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(A, h)
        return h


def gcn_norm(A):
    # normalization
    I = identity(A.shape)  # create an identity matrix
    A_hat = A + I  # add self-loop to A
    D = diag(A_hat.sum(0))  # diagonal degree matrix of A_hat
    D_hat = D
    D_hat = pow(D_hat, -0.5)
    A_hat = D_hat @ A_hat @ D_hat

    return A_hat


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
    for epoch in range(200):
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
        help="Dataset name ('cora', 'citeseer', 'pubmed', 'synthetic).",
    )
    args = parser.parse_args()
    print(f"Training with DGL SparseMatrix GraphConv module.")

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
    g = data[0].int()
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]

    row, col = g.adj_sparse("coo")
    A = create_from_coo(
        row, col, shape=(g.number_of_nodes(), g.number_of_nodes())
    )
    A = gcn_norm(A)

    # create GCN model
    in_size = features.shape[1]
    out_size = data.num_classes
    model = GCN(in_size, 16, out_size)

    # model training
    print("Training...")
    train(A, features, labels, masks, model)

    # test the model
    print("Testing...")
    acc = evaluate(A, features, labels, masks[2], model)
    print("Test accuracy {:.4f}".format(acc))
