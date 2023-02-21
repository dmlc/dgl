import argparse
import os
import random

import dgl

import numpy as np
import torch
import torch.optim as optim
from gnn import GNN
from ogb.lsc import DglPCQM4MDataset, PCQM4MEvaluator
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

reg_criterion = torch.nn.L1Loss()


def collate_dgl(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    labels = torch.stack(labels)

    return batched_graph, labels


def train(model, device, loader, optimizer):
    model.train()
    loss_accum = 0

    for step, (bg, labels) in enumerate(tqdm(loader, desc="Iteration")):
        bg = bg.to(device)
        x = bg.ndata.pop("feat")
        edge_attr = bg.edata.pop("feat")
        labels = labels.to(device)

        pred = model(bg, x, edge_attr).view(
            -1,
        )
        optimizer.zero_grad()
        loss = reg_criterion(pred, labels)
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, (bg, labels) in enumerate(tqdm(loader, desc="Iteration")):
        bg = bg.to(device)
        x = bg.ndata.pop("feat")
        edge_attr = bg.edata.pop("feat")
        labels = labels.to(device)

        with torch.no_grad():
            pred = model(bg, x, edge_attr).view(
                -1,
            )

        y_true.append(labels.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]


def test(model, device, loader):
    model.eval()
    y_pred = []

    for step, (bg, _) in enumerate(tqdm(loader, desc="Iteration")):
        bg = bg.to(device)
        x = bg.ndata.pop("feat")
        edge_attr = bg.edata.pop("feat")

        with torch.no_grad():
            pred = model(bg, x, edge_attr).view(
                -1,
            )

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)

    return y_pred


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description="GNN baselines on pcqm4m with DGL"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed to use (default: 42)"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="which gpu to use if any (default: 0)",
    )
    parser.add_argument(
        "--gnn",
        type=str,
        default="gin-virtual",
        help="GNN to use, which can be from "
        "[gin, gin-virtual, gcn, gcn-virtual] (default: gin-virtual)",
    )
    parser.add_argument(
        "--graph_pooling",
        type=str,
        default="sum",
        help="graph pooling strategy mean or sum (default: sum)",
    )
    parser.add_argument(
        "--drop_ratio", type=float, default=0, help="dropout ratio (default: 0)"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=5,
        help="number of GNN message passing layers (default: 5)",
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=600,
        help="dimensionality of hidden units in GNNs (default: 600)",
    )
    parser.add_argument(
        "--train_subset",
        action="store_true",
        help="use 10% of the training set for training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="input batch size for training (default: 256)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="number of workers (default: 0)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="tensorboard log directory. If not specified, "
        "tensorboard will not be used.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="directory to save checkpoint",
    )
    parser.add_argument(
        "--save_test_dir",
        type=str,
        default="",
        help="directory to save test submission file",
    )
    args = parser.parse_args()

    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = DglPCQM4MDataset(root="dataset/")

    # split_idx['train'], split_idx['valid'], split_idx['test']
    # separately gives a 1D int64 tensor
    split_idx = dataset.get_idx_split()

    ### automatic evaluator.
    evaluator = PCQM4MEvaluator()

    if args.train_subset:
        subset_ratio = 0.1
        subset_idx = torch.randperm(len(split_idx["train"]))[
            : int(subset_ratio * len(split_idx["train"]))
        ]
        train_loader = DataLoader(
            dataset[split_idx["train"][subset_idx]],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_dgl,
        )
    else:
        train_loader = DataLoader(
            dataset[split_idx["train"]],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_dgl,
        )

    valid_loader = DataLoader(
        dataset[split_idx["valid"]],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_dgl,
    )

    if args.save_test_dir != "":
        test_loader = DataLoader(
            dataset[split_idx["test"]],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_dgl,
        )

    if args.checkpoint_dir != "":
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    shared_params = {
        "num_layers": args.num_layers,
        "emb_dim": args.emb_dim,
        "drop_ratio": args.drop_ratio,
        "graph_pooling": args.graph_pooling,
    }

    if args.gnn == "gin":
        model = GNN(gnn_type="gin", virtual_node=False, **shared_params).to(
            device
        )
    elif args.gnn == "gin-virtual":
        model = GNN(gnn_type="gin", virtual_node=True, **shared_params).to(
            device
        )
    elif args.gnn == "gcn":
        model = GNN(gnn_type="gcn", virtual_node=False, **shared_params).to(
            device
        )
    elif args.gnn == "gcn-virtual":
        model = GNN(gnn_type="gcn", virtual_node=True, **shared_params).to(
            device
        )
    else:
        raise ValueError("Invalid GNN type")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"#Params: {num_params}")

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.log_dir != "":
        writer = SummaryWriter(log_dir=args.log_dir)

    best_valid_mae = 1000

    if args.train_subset:
        scheduler = StepLR(optimizer, step_size=300, gamma=0.25)
        args.epochs = 1000
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.25)

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print("Training...")
        train_mae = train(model, device, train_loader, optimizer)

        print("Evaluating...")
        valid_mae = eval(model, device, valid_loader, evaluator)

        print({"Train": train_mae, "Validation": valid_mae})

        if args.log_dir != "":
            writer.add_scalar("valid/mae", valid_mae, epoch)
            writer.add_scalar("train/mae", train_mae, epoch)

        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            if args.checkpoint_dir != "":
                print("Saving checkpoint...")
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_mae": best_valid_mae,
                    "num_params": num_params,
                }
                torch.save(
                    checkpoint,
                    os.path.join(args.checkpoint_dir, "checkpoint.pt"),
                )

            if args.save_test_dir != "":
                print("Predicting on test data...")
                y_pred = test(model, device, test_loader)
                print("Saving test submission file...")
                evaluator.save_test_submission(
                    {"y_pred": y_pred}, args.save_test_dir
                )

        scheduler.step()

        print(f"Best validation MAE so far: {best_valid_mae}")

    if args.log_dir != "":
        writer.close()


if __name__ == "__main__":
    main()
