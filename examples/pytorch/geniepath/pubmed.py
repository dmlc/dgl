import argparse

import torch as th
import torch.optim as optim

from dgl.data import PubmedGraphDataset
from model import GeniePath, GeniePathLazy
from sklearn.metrics import accuracy_score


def main(args):
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # Load dataset
    dataset = PubmedGraphDataset()
    graph = dataset[0]

    # check cuda
    if args.gpu >= 0 and th.cuda.is_available():
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    num_classes = dataset.num_classes

    # retrieve label of ground truth
    label = graph.ndata["label"].to(device)

    # Extract node features
    feat = graph.ndata["feat"].to(device)

    # retrieve masks for train/validation/test
    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]

    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze(1).to(device)
    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze(1).to(device)
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze(1).to(device)

    graph = graph.to(device)

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
    loss_fn = th.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Step 4: training epochs =============================================================== #
    for epoch in range(args.max_epoch):
        # Training and validation
        model.train()
        logits = model(graph, feat)

        # compute loss
        tr_loss = loss_fn(logits[train_idx], label[train_idx])
        tr_acc = accuracy_score(
            label[train_idx].cpu(), logits[train_idx].argmax(dim=1).cpu()
        )

        # validation
        valid_loss = loss_fn(logits[val_idx], label[val_idx])
        valid_acc = accuracy_score(
            label[val_idx].cpu(), logits[val_idx].argmax(dim=1).cpu()
        )

        # backward
        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()

        # Print out performance
        print(
            "In epoch {}, Train ACC: {:.4f} | Train Loss: {:.4f}; Valid ACC: {:.4f} | Valid loss: {:.4f}".format(
                epoch, tr_acc, tr_loss.item(), valid_acc, valid_loss.item()
            )
        )

    # Test after all epoch
    model.eval()

    # forward
    logits = model(graph, feat)

    # compute loss
    test_loss = loss_fn(logits[test_idx], label[test_idx])
    test_acc = accuracy_score(
        label[test_idx].cpu(), logits[test_idx].argmax(dim=1).cpu()
    )

    print(
        "Test ACC: {:.4f} | Test loss: {:.4f}".format(
            test_acc, test_loss.item()
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeniePath")
    parser.add_argument(
        "--gpu", type=int, default=-1, help="GPU Index. Default: -1, using CPU."
    )
    parser.add_argument(
        "--hid_dim", type=int, default=16, help="Hidden layer dimension"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of GeniePath layers"
    )
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=300,
        help="The max number of epochs. Default: 300",
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
        "--lazy", type=bool, default=False, help="Variant GeniePath-Lazy"
    )

    args = parser.parse_args()
    th.manual_seed(16)
    print(args)
    main(args)
