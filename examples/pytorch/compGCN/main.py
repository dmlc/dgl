import argparse
from time import time

import numpy as np
import torch as th
import torch.optim as optim
from data_loader import Data
from models import CompGCN_ConvE
from utils import in_out_norm


# predict the tail for (head, rel, -1) or head for (-1, rel, tail)
def predict(model, graph, device, data_iter, split="valid", mode="tail"):
    model.eval()
    with th.no_grad():
        results = {}
        train_iter = iter(data_iter["{}_{}".format(split, mode)])

        for step, batch in enumerate(train_iter):
            triple, label = batch[0].to(device), batch[1].to(device)
            sub, rel, obj, label = (
                triple[:, 0],
                triple[:, 1],
                triple[:, 2],
                label,
            )
            pred = model(graph, sub, rel)
            b_range = th.arange(pred.size()[0], device=device)
            target_pred = pred[b_range, obj]
            pred = th.where(label.bool(), -th.ones_like(pred) * 10000000, pred)
            pred[b_range, obj] = target_pred

            # compute metrics
            ranks = (
                1
                + th.argsort(
                    th.argsort(pred, dim=1, descending=True),
                    dim=1,
                    descending=False,
                )[b_range, obj]
            )
            ranks = ranks.float()
            results["count"] = th.numel(ranks) + results.get("count", 0.0)
            results["mr"] = th.sum(ranks).item() + results.get("mr", 0.0)
            results["mrr"] = th.sum(1.0 / ranks).item() + results.get(
                "mrr", 0.0
            )
            for k in [1, 3, 10]:
                results["hits@{}".format(k)] = th.numel(
                    ranks[ranks <= (k)]
                ) + results.get("hits@{}".format(k), 0.0)

    return results


# evaluation function, evaluate the head and tail prediction and then combine the results
def evaluate(model, graph, device, data_iter, split="valid"):
    # predict for head and tail
    left_results = predict(model, graph, device, data_iter, split, mode="tail")
    right_results = predict(model, graph, device, data_iter, split, mode="head")
    results = {}
    count = float(left_results["count"])

    # combine the head and tail prediction results
    # Metrics: MRR, MR, and Hit@k
    results["left_mr"] = round(left_results["mr"] / count, 5)
    results["left_mrr"] = round(left_results["mrr"] / count, 5)
    results["right_mr"] = round(right_results["mr"] / count, 5)
    results["right_mrr"] = round(right_results["mrr"] / count, 5)
    results["mr"] = round(
        (left_results["mr"] + right_results["mr"]) / (2 * count), 5
    )
    results["mrr"] = round(
        (left_results["mrr"] + right_results["mrr"]) / (2 * count), 5
    )
    for k in [1, 3, 10]:
        results["left_hits@{}".format(k)] = round(
            left_results["hits@{}".format(k)] / count, 5
        )
        results["right_hits@{}".format(k)] = round(
            right_results["hits@{}".format(k)] / count, 5
        )
        results["hits@{}".format(k)] = round(
            (
                left_results["hits@{}".format(k)]
                + right_results["hits@{}".format(k)]
            )
            / (2 * count),
            5,
        )
    return results


def main(args):
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # check cuda
    if args.gpu >= 0 and th.cuda.is_available():
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    # construct graph, split in/out edges and prepare train/validation/test data_loader
    data = Data(
        args.dataset, args.lbl_smooth, args.num_workers, args.batch_size
    )
    data_iter = data.data_iter  # train/validation/test data_loader
    graph = data.g.to(device)
    num_rel = th.max(graph.edata["etype"]).item() + 1

    # Compute in/out edge norms and store in edata
    graph = in_out_norm(graph)

    # Step 2: Create model =================================================================== #
    compgcn_model = CompGCN_ConvE(
        num_bases=args.num_bases,
        num_rel=num_rel,
        num_ent=graph.num_nodes(),
        in_dim=args.init_dim,
        layer_size=args.layer_size,
        comp_fn=args.opn,
        batchnorm=True,
        dropout=args.dropout,
        layer_dropout=args.layer_dropout,
        num_filt=args.num_filt,
        hid_drop=args.hid_drop,
        feat_drop=args.feat_drop,
        ker_sz=args.ker_sz,
        k_w=args.k_w,
        k_h=args.k_h,
    )
    compgcn_model = compgcn_model.to(device)

    # Step 3: Create training components ===================================================== #
    loss_fn = th.nn.BCELoss()
    optimizer = optim.Adam(
        compgcn_model.parameters(), lr=args.lr, weight_decay=args.l2
    )

    # Step 4: training epoches =============================================================== #
    best_mrr = 0.0
    kill_cnt = 0
    for epoch in range(args.max_epochs):
        # Training and validation using a full graph
        compgcn_model.train()
        train_loss = []
        t0 = time()
        for step, batch in enumerate(data_iter["train"]):
            triple, label = batch[0].to(device), batch[1].to(device)
            sub, rel, obj, label = (
                triple[:, 0],
                triple[:, 1],
                triple[:, 2],
                label,
            )
            logits = compgcn_model(graph, sub, rel)

            # compute loss
            tr_loss = loss_fn(logits, label)
            train_loss.append(tr_loss.item())

            # backward
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()

        train_loss = np.sum(train_loss)

        t1 = time()
        val_results = evaluate(
            compgcn_model, graph, device, data_iter, split="valid"
        )
        t2 = time()

        # validate
        if val_results["mrr"] > best_mrr:
            best_mrr = val_results["mrr"]
            th.save(
                compgcn_model.state_dict(), "comp_link" + "_" + args.dataset
            )
            kill_cnt = 0
            print("saving model...")
        else:
            kill_cnt += 1
            if kill_cnt > 100:
                print("early stop.")
                break
        print(
            "In epoch {}, Train Loss: {:.4f}, Valid MRR: {:.5}, Train time: {}, Valid time: {}".format(
                epoch, train_loss, val_results["mrr"], t1 - t0, t2 - t1
            )
        )

    # test use the best model
    compgcn_model.eval()
    compgcn_model.load_state_dict(th.load("comp_link" + "_" + args.dataset))
    test_results = evaluate(
        compgcn_model, graph, device, data_iter, split="test"
    )
    print(
        "Test MRR: {:.5}\n, MR: {:.10}\n, H@10: {:.5}\n, H@3: {:.5}\n, H@1: {:.5}\n".format(
            test_results["mrr"],
            test_results["mr"],
            test_results["hits@10"],
            test_results["hits@3"],
            test_results["hits@1"],
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parser For Arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data",
        dest="dataset",
        default="FB15k-237",
        help="Dataset to use, default: FB15k-237",
    )
    parser.add_argument(
        "--model", dest="model", default="compgcn", help="Model Name"
    )
    parser.add_argument(
        "--score_func",
        dest="score_func",
        default="conve",
        help="Score Function for Link prediction",
    )
    parser.add_argument(
        "--opn",
        dest="opn",
        default="ccorr",
        help="Composition Operation to be used in CompGCN",
    )

    parser.add_argument(
        "--batch", dest="batch_size", default=1024, type=int, help="Batch size"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default="0",
        help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0",
    )
    parser.add_argument(
        "--epoch",
        dest="max_epochs",
        type=int,
        default=500,
        help="Number of epochs",
    )
    parser.add_argument(
        "--l2", type=float, default=0.0, help="L2 Regularization for Optimizer"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Starting Learning Rate"
    )
    parser.add_argument(
        "--lbl_smooth",
        dest="lbl_smooth",
        type=float,
        default=0.1,
        help="Label Smoothing",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of processes to construct batches",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        default=41504,
        type=int,
        help="Seed for randomization",
    )

    parser.add_argument(
        "--num_bases",
        dest="num_bases",
        default=-1,
        type=int,
        help="Number of basis relation vectors to use",
    )
    parser.add_argument(
        "--init_dim",
        dest="init_dim",
        default=100,
        type=int,
        help="Initial dimension size for entities and relations",
    )
    parser.add_argument(
        "--layer_size",
        nargs="?",
        default="[200]",
        help="List of output size for each compGCN layer",
    )
    parser.add_argument(
        "--gcn_drop",
        dest="dropout",
        default=0.1,
        type=float,
        help="Dropout to use in GCN Layer",
    )
    parser.add_argument(
        "--layer_dropout",
        nargs="?",
        default="[0.3]",
        help="List of dropout value after each compGCN layer",
    )

    # ConvE specific hyperparameters
    parser.add_argument(
        "--hid_drop",
        dest="hid_drop",
        default=0.3,
        type=float,
        help="ConvE: Hidden dropout",
    )
    parser.add_argument(
        "--feat_drop",
        dest="feat_drop",
        default=0.3,
        type=float,
        help="ConvE: Feature Dropout",
    )
    parser.add_argument(
        "--k_w", dest="k_w", default=10, type=int, help="ConvE: k_w"
    )
    parser.add_argument(
        "--k_h", dest="k_h", default=20, type=int, help="ConvE: k_h"
    )
    parser.add_argument(
        "--num_filt",
        dest="num_filt",
        default=200,
        type=int,
        help="ConvE: Number of filters in convolution",
    )
    parser.add_argument(
        "--ker_sz",
        dest="ker_sz",
        default=7,
        type=int,
        help="ConvE: Kernel size to use",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    print(args)

    args.layer_size = eval(args.layer_size)
    args.layer_dropout = eval(args.layer_dropout)

    main(args)
