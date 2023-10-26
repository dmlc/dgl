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

import dgl.sparse as dglsp

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset


class SAGEConv(nn.Module):
    r"""GraphSAGE layer from `Inductive Representation Learning on
    Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`__
    """

    def __init__(
        self,
        in_feats,
        out_feats,
    ):
        super(SAGEConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = in_feats, in_feats
        self._out_feats = out_feats

        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, A, feat):
        feat_src = feat_dst = feat
        feat_dst = feat_src[: A.shape[0]]

        # Aggregator type: mean
        h_self = feat_dst
        srcdata = self.fc_neigh(feat_src)
        # Divided by degree.
        D_hat = dglsp.diag(A.sum(0)) ** -1
        A_div = A @ D_hat
        dstdata = A_div.T @ srcdata

        rst = self.fc_self(h_self) + dstdata
        return rst


class SAGE(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # Two-layer GraphSAGE-gcn.
        self.layers.append(SAGEConv(in_size, hidden_size))
        self.layers.append(SAGEConv(hidden_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, A, x):
        hidden_x = x
        for layer_idx, layer in enumerate(self.layers):
            hidden_x = layer(A, hidden_x)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = F.relu(hidden_x)
                hidden_x = self.dropout(hidden_x)
        return hidden_x


def evaluate(A, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(A, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(A, data, model):
    train_idx = data.train_idx.to(device)
    val_idx = data.val_idx.to(device)
    train_dataloader = torch.utils.data.DataLoader(train_idx, batchsize=10)

    # Define train/val samples, loss function and optimizer.
    train_mask, val_mask = masks
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # Training loop.
    for epoch in range(50):
        model.train()
        logits = model(A, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(A, features, labels, val_mask, model)
        print(
            f"Epoch {epoch:05d} | Loss {loss.item():.4f} | Accuracy {acc:.4f} "
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphSAGE")
    parser.add_argument(
        "--mode",
        default="puregpu",
        choices=["cpu", "puregpu"],
        help="Training mode. 'cpu' for CPU training, "
        "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

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
    print("Loading data")
    data = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
    g = data[0]
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    g = g.long().to(device)
    # features = g.ndata["feat"]
    # labels = g.ndata["label"]
    # masks = (g.ndata["train_mask"], g.ndata["val_mask"])

    # Load and preprocess dataset.
    # transform = (
    #     AddSelfLoop()
    # )  # By default, it will first remove self-loops to prevent duplication.
    # if args.dataset == "cora":
    #     data = CoraGraphDataset(transform=transform)
    # elif args.dataset == "citeseer":
    #     data = CiteseerGraphDataset(transform=transform)
    # elif args.dataset == "pubmed":
    #     data = PubmedGraphDataset(transform=transform)
    # else:
    #     raise ValueError(f"Unknown dataset: {args.dataset}")
    # g = data[0]
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # g = g.long().to(device)
    # features = g.ndata["feat"]
    # labels = g.ndata["label"]
    # masks = (g.ndata["train_mask"], g.ndata["val_mask"])

    # Create GraphSAGE model.
    in_size = g.ndata["feat"].shape[1]
    out_size = data.num_classes
    model = SAGE(in_size, 16, out_size).to(device)

    # Create sparse.
    indices = torch.stack(g.edges())
    N = g.num_nodes()
    A = dglsp.spmatrix(indices, shape=(N, N))

    # Model training.
    print("Training...")
    train(A, data, model)

    # Test the model.
    print("Testing...")
    acc = evaluate(A, features, labels, g.ndata["test_mask"], model)
    print(f"Test accuracy {acc:.4f}")
