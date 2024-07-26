"""
This script trains and tests a GraphSAGE model based on the information of 
a full graph.

This flowchart describes the main functional sequence of the provided example.
main
│
├───> Load and preprocess full dataset
│
├───> Instantiate SAGE model
│
├───> train
│     │
│     └───> Training loop
│           │
│           └───> SAGE.forward
└───> test
      │
      └───> Evaluate the model
"""
import argparse
import time

import dgl.nn as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset


class SAGE(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # Two-layer GraphSAGE-gcn.
        self.layers.append(dglnn.SAGEConv(in_size, hidden_size, "gcn"))
        self.layers.append(dglnn.SAGEConv(hidden_size, out_size, "gcn"))
        self.dropout = nn.Dropout(0.5)

    def forward(self, graph, x):
        hidden_x = x
        for layer_idx, layer in enumerate(self.layers):
            hidden_x = layer(graph, hidden_x)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = F.relu(hidden_x)
                hidden_x = self.dropout(hidden_x)
        return hidden_x


def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g, features, labels, masks, model):
    # Define train/val samples, loss function and optimizer.
    train_mask, val_mask = masks
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # Training loop.
    for epoch in range(200):
        t0 = time.time()
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t1 = time.time()
        acc = evaluate(g, features, labels, val_mask, model)
        print(
            f"Epoch {epoch:05d} | Loss {loss.item():.4f} | Accuracy {acc:.4f} | "
            f"Time {t1 - t0:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphSAGE")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'citeseer', 'pubmed')",
    )
    args = parser.parse_args()
    print(f"Training with DGL built-in GraphSage module")

    #####################################################################
    # (HIGHLIGHT) Node classification task is a supervise learning task
    # in which the model try to predict the label of a certain node.
    # In this example, graph sage algorithm is applied to this task.
    # A good accuracy can be achieved after a few steps of training.
    #
    # First, the whole graph is loaded and transformed. Then the training
    # process is performed on a model which is composed of 2 GraphSAGE-gcn
    # layer. Finally, the performance of the model is evaluated on test set.
    #####################################################################

    # Load and preprocess dataset.
    transform = (
        AddSelfLoop()
    )  # By default, it will first remove self-loops to prevent duplication.
    if args.dataset == "cora":
        data = CoraGraphDataset(transform=transform)
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset(transform=transform)
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset(transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    g = data[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = (g.ndata["train_mask"], g.ndata["val_mask"])

    # Create GraphSAGE model.
    in_size = features.shape[1]
    out_size = data.num_classes
    model = SAGE(in_size, 16, out_size).to(device)

    # Model training.
    print("Training...")
    train(g, features, labels, masks, model)

    # Test the model.
    print("Testing...")
    acc = evaluate(g, features, labels, g.ndata["test_mask"], model)
    print(f"Test accuracy {acc:.4f}")
