import argparse
import warnings

import dgl

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from model import GRAND

warnings.filterwarnings("ignore")


def argument():
    parser = argparse.ArgumentParser(description="GRAND")

    # data source params
    parser.add_argument(
        "--dataname", type=str, default="cora", help="Name of dataset."
    )
    # cuda params
    parser.add_argument(
        "--gpu", type=int, default=-1, help="GPU index. Default: -1, using CPU."
    )
    # training params
    parser.add_argument(
        "--epochs", type=int, default=200, help="Training epochs."
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=200,
        help="Patient epochs to wait before early stopping.",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="L2 reg."
    )
    # model params
    parser.add_argument(
        "--hid_dim", type=int, default=32, help="Hidden layer dimensionalities."
    )
    parser.add_argument(
        "--dropnode_rate",
        type=float,
        default=0.5,
        help="Dropnode rate (1 - keep probability).",
    )
    parser.add_argument(
        "--input_droprate",
        type=float,
        default=0.0,
        help="dropout rate of input layer",
    )
    parser.add_argument(
        "--hidden_droprate",
        type=float,
        default=0.0,
        help="dropout rate of hidden layer",
    )
    parser.add_argument("--order", type=int, default=8, help="Propagation step")
    parser.add_argument(
        "--sample", type=int, default=4, help="Sampling times of dropnode"
    )
    parser.add_argument(
        "--tem", type=float, default=0.5, help="Sharpening temperature"
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=1.0,
        help="Coefficient of consistency regularization",
    )
    parser.add_argument(
        "--use_bn",
        action="store_true",
        default=False,
        help="Using Batch Normalization",
    )

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and th.cuda.is_available():
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"

    return args


def consis_loss(logps, temp, lam):
    ps = [th.exp(p) for p in logps]
    ps = th.stack(ps, dim=2)

    avg_p = th.mean(ps, dim=2)
    sharp_p = (
        th.pow(avg_p, 1.0 / temp)
        / th.sum(th.pow(avg_p, 1.0 / temp), dim=1, keepdim=True)
    ).detach()

    sharp_p = sharp_p.unsqueeze(2)
    loss = th.mean(th.sum(th.pow(ps - sharp_p, 2), dim=1, keepdim=True))

    loss = lam * loss
    return loss


if __name__ == "__main__":
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # Load from DGL dataset
    args = argument()
    print(args)

    if args.dataname == "cora":
        dataset = CoraGraphDataset()
    elif args.dataname == "citeseer":
        dataset = CiteseerGraphDataset()
    elif args.dataname == "pubmed":
        dataset = PubmedGraphDataset()

    graph = dataset[0]

    graph = dgl.add_self_loop(graph)
    device = args.device

    # retrieve the number of classes
    n_classes = dataset.num_classes

    # retrieve labels of ground truth
    labels = graph.ndata.pop("label").to(device).long()

    # Extract node features
    feats = graph.ndata.pop("feat").to(device)
    n_features = feats.shape[-1]

    # retrieve masks for train/validation/test
    train_mask = graph.ndata.pop("train_mask")
    val_mask = graph.ndata.pop("val_mask")
    test_mask = graph.ndata.pop("test_mask")

    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze().to(device)
    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze().to(device)
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze().to(device)

    # Step 2: Create model =================================================================== #
    model = GRAND(
        n_features,
        args.hid_dim,
        n_classes,
        args.sample,
        args.order,
        args.dropnode_rate,
        args.input_droprate,
        args.hidden_droprate,
        args.use_bn,
    )

    model = model.to(args.device)
    graph = graph.to(args.device)

    # Step 3: Create training components ===================================================== #
    loss_fn = nn.NLLLoss()
    opt = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    loss_best = np.inf
    acc_best = 0

    # Step 4: training epoches =============================================================== #
    for epoch in range(args.epochs):
        """Training"""
        model.train()

        loss_sup = 0
        logits = model(graph, feats, True)

        # calculate supervised loss
        for k in range(args.sample):
            loss_sup += F.nll_loss(logits[k][train_idx], labels[train_idx])

        loss_sup = loss_sup / args.sample

        # calculate consistency loss
        loss_consis = consis_loss(logits, args.tem, args.lam)

        loss_train = loss_sup + loss_consis
        acc_train = th.sum(
            logits[0][train_idx].argmax(dim=1) == labels[train_idx]
        ).item() / len(train_idx)

        # backward
        opt.zero_grad()
        loss_train.backward()
        opt.step()

        """ Validating """
        model.eval()
        with th.no_grad():
            val_logits = model(graph, feats, False)

            loss_val = F.nll_loss(val_logits[val_idx], labels[val_idx])
            acc_val = th.sum(
                val_logits[val_idx].argmax(dim=1) == labels[val_idx]
            ).item() / len(val_idx)

            # Print out performance
            print(
                "In epoch {}, Train Acc: {:.4f} | Train Loss: {:.4f} ,Val Acc: {:.4f} | Val Loss: {:.4f}".format(
                    epoch,
                    acc_train,
                    loss_train.item(),
                    acc_val,
                    loss_val.item(),
                )
            )

            # set early stopping counter
            if loss_val < loss_best or acc_val > acc_best:
                if loss_val < loss_best:
                    best_epoch = epoch
                    th.save(model.state_dict(), args.dataname + ".pkl")
                no_improvement = 0
                loss_best = min(loss_val, loss_best)
                acc_best = max(acc_val, acc_best)
            else:
                no_improvement += 1
                if no_improvement == args.early_stopping:
                    print("Early stopping.")
                    break

    print("Optimization Finished!")

    print("Loading {}th epoch".format(best_epoch))
    model.load_state_dict(th.load(args.dataname + ".pkl"))

    """ Testing """
    model.eval()

    test_logits = model(graph, feats, False)
    test_acc = th.sum(
        test_logits[test_idx].argmax(dim=1) == labels[test_idx]
    ).item() / len(test_idx)

    print("Test Acc: {:.4f}".format(test_acc))
