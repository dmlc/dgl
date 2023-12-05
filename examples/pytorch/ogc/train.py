import argparse
import time

import dgl.sparse as dglsp

import torch.nn.functional as F
import torch.optim as optim
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset

from ogc import OGC
from utils import model_test, symmetric_normalize_adjacency


def train(model, embeds, lazy_adj, args):
    patience = 0
    _, _, last_acc, last_output = model_test(model, embeds)

    tv_mask = model.tv_mask
    optimizer = optim.SGD(model.parameters(), lr=args.lr_clf)

    for i in range(64):
        model.train()
        output = model(embeds)
        loss_tv = F.mse_loss(
            output[tv_mask], model.label_one_hot[tv_mask], reduction="sum"
        )
        optimizer.zero_grad()
        loss_tv.backward()
        optimizer.step()

        # Updating node embeds by LGC and SEB jointly.
        embeds = model.update_embeds(embeds, lazy_adj, args)

        loss_tv, acc_tv, acc_test, pred = model_test(model, embeds)
        print(
            "epoch {} loss_tv {:.4f} acc_tv {:.4f} acc_test {:.4f}".format(
                i + 1, loss_tv, acc_tv, acc_test
            )
        )

        sim_rate = float(int((pred == last_output).sum()) / int(pred.shape[0]))
        if sim_rate > args.max_sim_rate:
            patience += 1
            if patience > args.max_patience:
                break
        last_acc = acc_test
        last_output = pred
    return last_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="citeseer",
        choices=["cora", "citeseer", "pubmed"],
        help="dataset to use",
    )
    parser.add_argument(
        "--decline", type=float, default=0.9, help="decline rate"
    )
    parser.add_argument(
        "--lr_sup",
        type=float,
        default=0.001,
        help="learning rate for supervised loss",
    )
    parser.add_argument(
        "--lr_clf",
        type=float,
        default=0.5,
        help="learning rate for the used linear classifier",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="moving probability that a node moves to its neighbors",
    )
    parser.add_argument(
        "--max_sim_rate",
        type=float,
        default=0.995,
        help="max label prediction similarity between iterations",
    )
    parser.add_argument(
        "--max_patience",
        type=int,
        default=2,
        help="tolerance for consecutively similar test predictions",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="device to use",
    )
    args, _ = parser.parse_known_args()

    # Load and preprocess dataset.
    transform = AddSelfLoop()
    if args.dataset == "cora":
        data = CoraGraphDataset(transform=transform)
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset(transform=transform)
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset(transform=transform)
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    graph = data[0].to(args.device)
    features = graph.ndata["feat"]
    adj = symmetric_normalize_adjacency(graph)
    I_N = dglsp.identity((features.shape[0], features.shape[0]))
    # Lazy random walk (also known as lazy graph convolution).
    lazy_adj = dglsp.add((1 - args.beta) * I_N, args.beta * adj).to(args.device)

    model = OGC(graph).to(args.device)
    start_time = time.time()
    res = train(model, features, lazy_adj, args)
    time_tot = time.time() - start_time

    print(f"Test Acc:{res:.4f}")
    print(f"Total Time:{time_tot:.4f}")
