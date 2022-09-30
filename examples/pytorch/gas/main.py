import argparse

import torch as th
import torch.nn.functional as F
import torch.optim as optim
from dataloader import GASDataset
from model import GAS
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score


def main(args):
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # Load dataset
    dataset = GASDataset(args.dataset)
    graph = dataset[0]

    # check cuda
    if args.gpu >= 0 and th.cuda.is_available():
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    # binary classification
    num_classes = dataset.num_classes

    # retrieve labels of ground truth
    labels = graph.edges["forward"].data["label"].to(device).long()

    # Extract node features
    e_feat = graph.edges["forward"].data["feat"].to(device)
    u_feat = graph.nodes["u"].data["feat"].to(device)
    v_feat = graph.nodes["v"].data["feat"].to(device)

    # retrieve masks for train/validation/test
    train_mask = graph.edges["forward"].data["train_mask"]
    val_mask = graph.edges["forward"].data["val_mask"]
    test_mask = graph.edges["forward"].data["test_mask"]

    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze(1).to(device)
    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze(1).to(device)
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze(1).to(device)

    graph = graph.to(device)

    # Step 2: Create model =================================================================== #
    model = GAS(
        e_in_dim=e_feat.shape[-1],
        u_in_dim=u_feat.shape[-1],
        v_in_dim=v_feat.shape[-1],
        e_hid_dim=args.e_hid_dim,
        u_hid_dim=args.u_hid_dim,
        v_hid_dim=args.v_hid_dim,
        out_dim=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        activation=F.relu,
    )

    model = model.to(device)

    # Step 3: Create training components ===================================================== #
    loss_fn = th.nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Step 4: training epochs =============================================================== #
    for epoch in range(args.max_epoch):
        # Training and validation using a full graph
        model.train()
        logits = model(graph, e_feat, u_feat, v_feat)

        # compute loss
        tr_loss = loss_fn(logits[train_idx], labels[train_idx])
        tr_f1 = f1_score(
            labels[train_idx].cpu(), logits[train_idx].argmax(dim=1).cpu()
        )
        tr_auc = roc_auc_score(
            labels[train_idx].cpu(), logits[train_idx][:, 1].detach().cpu()
        )
        tr_pre, tr_re, _ = precision_recall_curve(
            labels[train_idx].cpu(), logits[train_idx][:, 1].detach().cpu()
        )
        tr_rap = tr_re[tr_pre > args.precision].max()

        # validation
        valid_loss = loss_fn(logits[val_idx], labels[val_idx])
        valid_f1 = f1_score(
            labels[val_idx].cpu(), logits[val_idx].argmax(dim=1).cpu()
        )
        valid_auc = roc_auc_score(
            labels[val_idx].cpu(), logits[val_idx][:, 1].detach().cpu()
        )
        valid_pre, valid_re, _ = precision_recall_curve(
            labels[val_idx].cpu(), logits[val_idx][:, 1].detach().cpu()
        )
        valid_rap = valid_re[valid_pre > args.precision].max()

        # backward
        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()

        # Print out performance
        print(
            "In epoch {}, Train R@P: {:.4f} | Train F1: {:.4f} | Train AUC: {:.4f} | Train Loss: {:.4f}; "
            "Valid R@P: {:.4f} | Valid F1: {:.4f} | Valid AUC: {:.4f} | Valid loss: {:.4f}".format(
                epoch,
                tr_rap,
                tr_f1,
                tr_auc,
                tr_loss.item(),
                valid_rap,
                valid_f1,
                valid_auc,
                valid_loss.item(),
            )
        )

    # Test after all epoch
    model.eval()

    # forward
    logits = model(graph, e_feat, u_feat, v_feat)

    # compute loss
    test_loss = loss_fn(logits[test_idx], labels[test_idx])
    test_f1 = f1_score(
        labels[test_idx].cpu(), logits[test_idx].argmax(dim=1).cpu()
    )
    test_auc = roc_auc_score(
        labels[test_idx].cpu(), logits[test_idx][:, 1].detach().cpu()
    )
    test_pre, test_re, _ = precision_recall_curve(
        labels[test_idx].cpu(), logits[test_idx][:, 1].detach().cpu()
    )
    test_rap = test_re[test_pre > args.precision].max()

    print(
        "Test R@P: {:.4f} | Test F1: {:.4f} | Test AUC: {:.4f} | Test loss: {:.4f}".format(
            test_rap, test_f1, test_auc, test_loss.item()
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN-based Anti-Spam Model")
    parser.add_argument(
        "--dataset", type=str, default="pol", help="'pol', or 'gos'"
    )
    parser.add_argument(
        "--gpu", type=int, default=-1, help="GPU Index. Default: -1, using CPU."
    )
    parser.add_argument(
        "--e_hid_dim",
        type=int,
        default=128,
        help="Hidden layer dimension for edges",
    )
    parser.add_argument(
        "--u_hid_dim",
        type=int,
        default=128,
        help="Hidden layer dimension for source nodes",
    )
    parser.add_argument(
        "--v_hid_dim",
        type=int,
        default=128,
        help="Hidden layer dimension for destination nodes",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of GCN layers"
    )
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=100,
        help="The max number of epochs. Default: 100",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate. Default: 1e-3"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout rate. Default: 0.0"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight Decay. Default: 0.0005",
    )
    parser.add_argument(
        "--precision",
        type=float,
        default=0.9,
        help="The value p in recall@p precision. Default: 0.9",
    )

    args = parser.parse_args()
    print(args)
    main(args)
