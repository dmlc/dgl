import os
import time

import dgl.function as fn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator
from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from utils import load_model, set_random_seed


def normalize_edge_weights(graph, device, num_ew_channels):
    degs = graph.in_degrees().float()
    degs = torch.clamp(degs, min=1)
    norm = torch.pow(degs, 0.5)
    norm = norm.to(args["device"])
    graph.ndata["norm"] = norm.unsqueeze(1)
    graph.apply_edges(fn.e_div_u("feat", "norm", "feat"))
    graph.apply_edges(fn.e_div_v("feat", "norm", "feat"))
    for channel in range(num_ew_channels):
        graph.edata["feat_" + str(channel)] = graph.edata["feat"][
            :, channel : channel + 1
        ]


def run_a_train_epoch(graph, node_idx, model, criterion, optimizer, evaluator):
    model.train()
    logits = model(graph)[node_idx]
    labels = graph.ndata["labels"][node_idx]
    loss = criterion(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss = loss.data.item()
    labels = labels.cpu().numpy()
    preds = logits.cpu().detach().numpy()

    return loss, evaluator.eval({"y_true": labels, "y_pred": preds})["rocauc"]


def run_an_eval_epoch(graph, splitted_idx, model, evaluator):
    model.eval()
    with torch.no_grad():
        logits = model(graph)
    labels = graph.ndata["labels"].cpu().numpy()
    preds = logits.cpu().detach().numpy()

    train_score = evaluator.eval(
        {
            "y_true": labels[splitted_idx["train"]],
            "y_pred": preds[splitted_idx["train"]],
        }
    )
    val_score = evaluator.eval(
        {
            "y_true": labels[splitted_idx["valid"]],
            "y_pred": preds[splitted_idx["valid"]],
        }
    )
    test_score = evaluator.eval(
        {
            "y_true": labels[splitted_idx["test"]],
            "y_pred": preds[splitted_idx["test"]],
        }
    )

    return train_score["rocauc"], val_score["rocauc"], test_score["rocauc"]


def main(args):
    print(args)
    if args["rand_seed"] > -1:
        set_random_seed(args["rand_seed"])

    dataset = DglNodePropPredDataset(name=args["dataset"])
    print(dataset.meta_info)
    splitted_idx = dataset.get_idx_split()
    graph = dataset.graph[0]
    graph.ndata["labels"] = dataset.labels.float().to(args["device"])
    graph.edata["feat"] = graph.edata["feat"].float().to(args["device"])

    if args["ewnorm"] == "both":
        print("Symmetric normalization of edge weights by degree")
        normalize_edge_weights(graph, args["device"], args["num_ew_channels"])
    elif args["ewnorm"] == "none":
        print("Not normalizing edge weights")
        for channel in range(args["num_ew_channels"]):
            graph.edata["feat_" + str(channel)] = graph.edata["feat"][
                :, channel : channel + 1
            ]

    model = load_model(args).to(args["device"])
    optimizer = Adam(
        model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
    )
    min_lr = 1e-3
    scheduler = ReduceLROnPlateau(
        optimizer, "max", factor=0.7, patience=100, verbose=True, min_lr=min_lr
    )
    print("scheduler min_lr", min_lr)

    criterion = nn.BCEWithLogitsLoss()
    evaluator = Evaluator(args["dataset"])

    print("model", args["model"])
    print("n_layers", args["n_layers"])
    print("hidden dim", args["hidden_feats"])
    print("lr", args["lr"])

    dur = []
    best_val_score = 0.0
    num_patient_epochs = 0
    model_folder = "./saved_models/"
    model_path = (
        model_folder + str(args["exp_name"]) + "_" + str(args["postfix"])
    )

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    for epoch in range(1, args["num_epochs"] + 1):
        if epoch >= 3:
            t0 = time.time()

        loss, train_score = run_a_train_epoch(
            graph, splitted_idx["train"], model, criterion, optimizer, evaluator
        )

        if epoch >= 3:
            dur.append(time.time() - t0)
            avg_time = np.mean(dur)
        else:
            avg_time = None

        train_score, val_score, test_score = run_an_eval_epoch(
            graph, splitted_idx, model, evaluator
        )

        scheduler.step(val_score)

        # Early stop
        if val_score > best_val_score:
            torch.save(model.state_dict(), model_path)
            best_val_score = val_score
            num_patient_epochs = 0
        else:
            num_patient_epochs += 1

        print(
            "Epoch {:d}, loss {:.4f}, train score {:.4f}, "
            "val score {:.4f}, avg time {}, num patient epochs {:d}".format(
                epoch,
                loss,
                train_score,
                val_score,
                avg_time,
                num_patient_epochs,
            )
        )

        if num_patient_epochs == args["patience"]:
            break

    model.load_state_dict(torch.load(model_path))
    train_score, val_score, test_score = run_an_eval_epoch(
        graph, splitted_idx, model, evaluator
    )
    print("Train score {:.4f}".format(train_score))
    print("Valid score {:.4f}".format(val_score))
    print("Test score {:.4f}".format(test_score))

    with open("results.txt", "w") as f:
        f.write("loss {:.4f}\n".format(loss))
        f.write("Best validation rocauc {:.4f}\n".format(best_val_score))
        f.write("Test rocauc {:.4f}\n".format(test_score))

    print(args)


if __name__ == "__main__":
    import argparse

    from configure import get_exp_configure

    parser = argparse.ArgumentParser(
        description="OGB node property prediction with DGL using full graph training"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["MWE-GCN", "MWE-DGCN"],
        default="MWE-DGCN",
        help="Model to use",
    )
    parser.add_argument("-c", "--cuda", type=str, default="none")
    parser.add_argument(
        "--postfix",
        type=str,
        default="",
        help="a string appended to the file name of the saved model",
    )
    parser.add_argument(
        "--rand_seed",
        type=int,
        default=-1,
        help="random seed for torch and numpy",
    )
    parser.add_argument("--residual", action="store_true")
    parser.add_argument(
        "--ewnorm", type=str, default="none", choices=["none", "both"]
    )
    args = parser.parse_args().__dict__

    # Get experiment configuration
    args["dataset"] = "ogbn-proteins"
    args["exp_name"] = "_".join([args["model"], args["dataset"]])
    args.update(get_exp_configure(args))

    if not (args["cuda"] == "none"):
        args["device"] = torch.device("cuda: " + str(args["cuda"]))
    else:
        args["device"] = torch.device("cpu")

    main(args)
