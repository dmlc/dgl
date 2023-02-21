import argparse
import copy
import os

import dgl

import torch
import torch.nn.functional as F
import torch.optim as optim
from model import CorrectAndSmooth, MLP, MLPLinear
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator


def evaluate(y_pred, y_true, idx, evaluator):
    return evaluator.eval({"y_true": y_true[idx], "y_pred": y_pred[idx]})["acc"]


def main():
    # check cuda
    device = (
        f"cuda:{args.gpu}"
        if torch.cuda.is_available() and args.gpu >= 0
        else "cpu"
    )
    # load data
    dataset = DglNodePropPredDataset(name=args.dataset)
    evaluator = Evaluator(name=args.dataset)

    split_idx = dataset.get_idx_split()
    g, labels = dataset[
        0
    ]  # graph: DGLGraph object, label: torch tensor of shape (num_nodes, num_tasks)

    if args.dataset == "ogbn-arxiv":
        g = dgl.to_bidirected(g, copy_ndata=True)

        feat = g.ndata["feat"]
        feat = (feat - feat.mean(0)) / feat.std(0)
        g.ndata["feat"] = feat

    g = g.to(device)
    feats = g.ndata["feat"]
    labels = labels.to(device)

    # load masks for train / validation / test
    train_idx = split_idx["train"].to(device)
    valid_idx = split_idx["valid"].to(device)
    test_idx = split_idx["test"].to(device)

    n_features = feats.size()[-1]
    n_classes = dataset.num_classes

    # load model
    if args.model == "mlp":
        model = MLP(
            n_features, args.hid_dim, n_classes, args.num_layers, args.dropout
        )
    elif args.model == "linear":
        model = MLPLinear(n_features, n_classes)
    else:
        raise NotImplementedError(f"Model {args.model} is not supported.")

    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    if args.pretrain:
        print("---------- Before ----------")
        model.load_state_dict(
            torch.load(f"base/{args.dataset}-{args.model}.pt")
        )
        model.eval()

        y_soft = model(feats).exp()

        y_pred = y_soft.argmax(dim=-1, keepdim=True)
        valid_acc = evaluate(y_pred, labels, valid_idx, evaluator)
        test_acc = evaluate(y_pred, labels, test_idx, evaluator)
        print(f"Valid acc: {valid_acc:.4f} | Test acc: {test_acc:.4f}")

        print("---------- Correct & Smoothing ----------")
        cs = CorrectAndSmooth(
            num_correction_layers=args.num_correction_layers,
            correction_alpha=args.correction_alpha,
            correction_adj=args.correction_adj,
            num_smoothing_layers=args.num_smoothing_layers,
            smoothing_alpha=args.smoothing_alpha,
            smoothing_adj=args.smoothing_adj,
            autoscale=args.autoscale,
            scale=args.scale,
        )

        y_soft = cs.correct(g, y_soft, labels[train_idx], train_idx)
        y_soft = cs.smooth(g, y_soft, labels[train_idx], train_idx)
        y_pred = y_soft.argmax(dim=-1, keepdim=True)
        valid_acc = evaluate(y_pred, labels, valid_idx, evaluator)
        test_acc = evaluate(y_pred, labels, test_idx, evaluator)
        print(f"Valid acc: {valid_acc:.4f} | Test acc: {test_acc:.4f}")
    else:
        opt = optim.Adam(model.parameters(), lr=args.lr)

        best_acc = 0
        best_model = copy.deepcopy(model)

        # training
        print("---------- Training ----------")
        for i in range(args.epochs):
            model.train()
            opt.zero_grad()

            logits = model(feats)

            train_loss = F.nll_loss(
                logits[train_idx], labels.squeeze(1)[train_idx]
            )
            train_loss.backward()

            opt.step()

            model.eval()
            with torch.no_grad():
                logits = model(feats)

                y_pred = logits.argmax(dim=-1, keepdim=True)

                train_acc = evaluate(y_pred, labels, train_idx, evaluator)
                valid_acc = evaluate(y_pred, labels, valid_idx, evaluator)

                print(
                    f"Epoch {i} | Train loss: {train_loss.item():.4f} | Train acc: {train_acc:.4f} | Valid acc {valid_acc:.4f}"
                )

                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_model = copy.deepcopy(model)

        # testing & saving model
        print("---------- Testing ----------")
        best_model.eval()

        logits = best_model(feats)

        y_pred = logits.argmax(dim=-1, keepdim=True)
        test_acc = evaluate(y_pred, labels, test_idx, evaluator)
        print(f"Test acc: {test_acc:.4f}")

        if not os.path.exists("base"):
            os.makedirs("base")

        torch.save(
            best_model.state_dict(), f"base/{args.dataset}-{args.model}.pt"
        )


if __name__ == "__main__":
    """
    Correct & Smoothing Hyperparameters
    """
    parser = argparse.ArgumentParser(description="Base predictor(C&S)")

    # Dataset
    parser.add_argument("--gpu", type=int, default=0, help="-1 for cpu")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-arxiv",
        choices=["ogbn-arxiv", "ogbn-products"],
    )
    # Base predictor
    parser.add_argument(
        "--model", type=str, default="mlp", choices=["mlp", "linear"]
    )
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--hid-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=300)
    # extra options for gat
    parser.add_argument("--n-heads", type=int, default=3)
    parser.add_argument("--attn_drop", type=float, default=0.05)
    # C & S
    parser.add_argument(
        "--pretrain", action="store_true", help="Whether to perform C & S"
    )
    parser.add_argument("--num-correction-layers", type=int, default=50)
    parser.add_argument("--correction-alpha", type=float, default=0.979)
    parser.add_argument("--correction-adj", type=str, default="DAD")
    parser.add_argument("--num-smoothing-layers", type=int, default=50)
    parser.add_argument("--smoothing-alpha", type=float, default=0.756)
    parser.add_argument("--smoothing-adj", type=str, default="DAD")
    parser.add_argument("--autoscale", action="store_true")
    parser.add_argument("--scale", type=float, default=20.0)

    args = parser.parse_args()
    print(args)

    main()
