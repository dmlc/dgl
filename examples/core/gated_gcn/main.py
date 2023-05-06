import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from model import GatedGCN
from ogb.graphproppred import collate_dgl, DglGraphPropPredDataset, Evaluator
from torch.utils.data import DataLoader


def train(model, device, data_loader, opt, loss_fn):
    model.train()
    train_loss = []

    for g, labels in data_loader:
        g = g.to(device)
        labels = labels.to(torch.float32).to(device)
        logits = model(g, g.edata["feat"], g.ndata["feat"])
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
        logits = model(g, g.edata["feat"], g.ndata["feat"])
        y_true.append(labels.detach().cpu())
        y_pred.append(logits.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    return evaluator.eval({"y_true": y_true, "y_pred": y_pred})["rocauc"]


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load ogb dataset & evaluator.
    dataset = DglGraphPropPredDataset(name="ogbg-molhiv")
    evaluator = Evaluator(name="ogbg-molhiv")

    n_classes = dataset.num_tasks

    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(
        dataset[split_idx["train"]],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_dgl,
    )
    valid_loader = DataLoader(
        dataset[split_idx["valid"]],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_dgl,
    )

    test_loader = DataLoader(
        dataset[split_idx["test"]],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_dgl,
    )

    # Load model.
    model = GatedGCN(
        hid_dim=args.hid_dim,
        out_dim=n_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    print(model)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    print("---------- Training ----------")
    for epoch in range(args.epochs):
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
    parser = argparse.ArgumentParser(description="GatedGCN")
    # Parameters.
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs to train."
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="Dropout rate."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size."
    )
    parser.add_argument(
        "--num-layers", type=int, default=7, help="Number of GNN layers."
    )
    parser.add_argument(
        "--hid-dim", type=int, default=256, help="Hidden channel size."
    )

    args = parser.parse_args()

    main(args)
