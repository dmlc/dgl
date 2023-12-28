import argparse

import numpy as np
import torch as th
import torch.optim as optim

from dgl.data import PPIDataset
from dgl.dataloading import GraphDataLoader
from model import GeniePath, GeniePathLazy
from sklearn.metrics import f1_score


def evaluate(model, loss_fn, dataloader, device="cpu"):
    loss = 0
    f1 = 0
    num_blocks = 0
    for subgraph in dataloader:
        subgraph = subgraph.to(device)
        label = subgraph.ndata["label"].to(device)
        feat = subgraph.ndata["feat"]
        logits = model(subgraph, feat)

        # compute loss
        loss += loss_fn(logits, label).item()
        predict = np.where(logits.data.cpu().numpy() >= 0.0, 1, 0)
        f1 += f1_score(label.cpu(), predict, average="micro")
        num_blocks += 1

    return f1 / num_blocks, loss / num_blocks


def main(args):
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # Load dataset
    train_dataset = PPIDataset(mode="train")
    valid_dataset = PPIDataset(mode="valid")
    test_dataset = PPIDataset(mode="test")
    train_dataloader = GraphDataLoader(
        train_dataset, batch_size=args.batch_size
    )
    valid_dataloader = GraphDataLoader(
        valid_dataset, batch_size=args.batch_size
    )
    test_dataloader = GraphDataLoader(test_dataset, batch_size=args.batch_size)

    # check cuda
    if args.gpu >= 0 and th.cuda.is_available():
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    num_classes = train_dataset.num_classes

    # Extract node features
    graph = train_dataset[0]
    feat = graph.ndata["feat"]

    # Step 2: Create model =================================================================== #
    if args.lazy:
        model = GeniePathLazy(
            in_dim=feat.shape[-1],
            out_dim=num_classes,
            hid_dim=args.hid_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            residual=args.residual,
        )
    else:
        model = GeniePath(
            in_dim=feat.shape[-1],
            out_dim=num_classes,
            hid_dim=args.hid_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            residual=args.residual,
        )

    model = model.to(device)

    # Step 3: Create training components ===================================================== #
    loss_fn = th.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Step 4: training epochs =============================================================== #
    for epoch in range(args.max_epoch):
        model.train()
        tr_loss = 0
        tr_f1 = 0
        num_blocks = 0
        for subgraph in train_dataloader:
            subgraph = subgraph.to(device)
            label = subgraph.ndata["label"]
            feat = subgraph.ndata["feat"]
            logits = model(subgraph, feat)

            # compute loss
            batch_loss = loss_fn(logits, label)
            tr_loss += batch_loss.item()
            tr_predict = np.where(logits.data.cpu().numpy() >= 0.0, 1, 0)
            tr_f1 += f1_score(label.cpu(), tr_predict, average="micro")
            num_blocks += 1

            # backward
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # validation
        model.eval()
        val_f1, val_loss = evaluate(model, loss_fn, valid_dataloader, device)

        print(
            "In epoch {}, Train F1: {:.4f} | Train Loss: {:.4f}; Valid F1: {:.4f} | Valid loss: {:.4f}".format(
                epoch,
                tr_f1 / num_blocks,
                tr_loss / num_blocks,
                val_f1,
                val_loss,
            )
        )

    # Test after all epoch
    model.eval()
    test_f1, test_loss = evaluate(model, loss_fn, test_dataloader, device)

    print("Test F1: {:.4f} | Test loss: {:.4f}".format(test_f1, test_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeniePath")
    parser.add_argument(
        "--gpu", type=int, default=-1, help="GPU Index. Default: -1, using CPU."
    )
    parser.add_argument(
        "--hid_dim", type=int, default=256, help="Hidden layer dimension"
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="Number of GeniePath layers"
    )
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=1000,
        help="The max number of epochs. Default: 1000",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0004,
        help="Learning rate. Default: 0.0004",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=1,
        help="Number of head in breadth function. Default: 1",
    )
    parser.add_argument(
        "--residual", type=bool, default=False, help="Residual in GAT or not"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size of graph dataloader",
    )
    parser.add_argument(
        "--lazy", type=bool, default=False, help="Variant GeniePath-Lazy"
    )

    args = parser.parse_args()
    print(args)
    th.manual_seed(16)
    main(args)
