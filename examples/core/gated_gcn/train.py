"""
Gated Graph Convolutional Network module for graph classification tasks
"""

import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import GatedGCNConv
from dgl.nn.pytorch.glob import AvgPooling
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class GatedGCN(nn.Module):
    def __init__(
        self,
        hid_dim,
        out_dim,
        num_layers,
        dropout=0.2,
        batch_norm=True,
        residual=True,
        activation=F.relu,
    ):
        super(GatedGCN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.node_encoder = AtomEncoder(hid_dim)
        self.edge_encoder = BondEncoder(hid_dim)

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            layer = GatedGCNConv(
                input_feats=hid_dim,
                edge_feats=hid_dim,
                output_feats=hid_dim,
                dropout=dropout,
                batch_norm=batch_norm,
                residual=residual,
                activation=activation,
            )
            self.layers.append(layer)

        self.pooling = AvgPooling()
        self.output = nn.Linear(hid_dim, out_dim)

    def forward(self, g, node_feat, edge_feat):
        # Encode node and edge feature.
        hv = self.node_encoder(node_feat)
        he = self.edge_encoder(edge_feat)

        # GatedGCNConv layers.
        for layer in self.layers:
            hv, he = layer(g, hv, he)

        # Output project.
        h_g = self.pooling(g, hv)

        return self.output(h_g)


def train(model, device, data_loader, opt, loss_fn):
    model.train()
    train_loss = []

    for g, labels in data_loader:
        g = g.to(device)
        labels = labels.to(torch.float32).to(device)
        logits = model(g, g.ndata["feat"], g.edata["feat"])
        loss = loss_fn(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss.append(loss.item())

    return sum(train_loss) / len(train_loss)


@torch.no_grad()
def evaluate(model, device, data_loader, evaluator):
    model.eval()
    y_true, y_pred = [], []

    for g, labels in data_loader:
        g = g.to(device)
        logits = model(g, g.ndata["feat"], g.edata["feat"])
        y_true.append(labels.detach().cpu())
        y_pred.append(logits.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    return evaluator.eval({"y_true": y_true, "y_pred": y_pred})["rocauc"]


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load ogb dataset & evaluator.
    dataset = DglGraphPropPredDataset(name="ogbg-molhiv")
    evaluator = Evaluator(name="ogbg-molhiv")

    n_classes = dataset.num_tasks

    split_idx = dataset.get_idx_split()
    train_loader = GraphDataLoader(
        dataset[split_idx["train"]],
        batch_size=32,
        shuffle=True,
    )
    valid_loader = GraphDataLoader(dataset[split_idx["valid"]], batch_size=32)
    test_loader = GraphDataLoader(dataset[split_idx["test"]], batch_size=32)

    # Load model.
    model = GatedGCN(hid_dim=256, out_dim=n_classes, num_layers=8).to(device)

    print(model)

    opt = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    print("---------- Training ----------")
    for epoch in range(50):
        # Kick off training.
        loss = train(model, device, train_loader, opt, loss_fn)

        # Evaluate the prediction.
        valid_acc = evaluate(model, device, valid_loader, evaluator)
        test_acc = evaluate(model, device, test_loader, evaluator)
        print(
            f"In epoch {epoch}, loss: {loss:.3f}, val acc: {valid_acc:.3f}, test"
            f" acc: {test_acc:.3f}"
        )


if __name__ == "__main__":
    main()
