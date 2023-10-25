import argparse
import time

import scipy.sparse as sp

import torch
import torch.nn.functional as F

from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from utils import (
    LinearNeuralNetwork,
    sparse_mx_to_torch_sparse_tensor,
    symmetric_normalize_adjacency,
)


# Training settings.
decline = 0.9  # the decline rate
lr_sup = 0.001  # the learning rate for supervised loss
lr_clf = 0.5  # the learning rate for the used linear classifier
beta = 0.1  # the moving probability that a node moves to its neighbors
max_sim_rate = 0.995  # the max label prediction similarity between iterations
max_patience = 2  # the tolerance for consecutively similar test predictions


def update_embeds(embeds, graph, label_idx_mat):
    global lr_sup
    # Update classifier's weight by training a linear supervised model.
    pred_labels, clf_weight = linear_clf.update_weight(embeds, graph, lr_clf)
    labels = F.one_hot(graph.ndata["label"]).float()

    # Update the smoothness loss via LGC.
    embeds = torch.spmm(lazy_adj.to(device), embeds)

    # Update the supervised loss via SEB.
    deriv_sup = 2 * torch.mm(
        torch.sparse.mm(label_idx_mat, -labels + pred_labels), clf_weight
    )
    embeds = embeds - lr_sup * deriv_sup

    lr_sup = lr_sup * decline
    return embeds


def OGC(linear_clf, embeds, graph, label_idx_mat):
    patience = 0
    _, _, last_acc, last_output = linear_clf.test(embeds, graph)
    for i in range(64):
        # Updating node embeds by LGC and SEB jointly.
        embeds = update_embeds(embeds, graph, label_idx_mat)

        loss_tv, acc_tv, acc_test, pred = linear_clf.test(embeds, graph)
        print(
            "epoch {} loss_tv {:.4f} acc_tv {:.4f} acc_test {:.4f}".format(
                i + 1, loss_tv, acc_tv, acc_test
            )
        )

        sim_rate = float(int((pred == last_output).sum()) / int(pred.shape[0]))
        if sim_rate > max_sim_rate:
            patience += 1
            if patience > max_patience:
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
        help="Dataset to use.",
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
    graph = data[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = graph.int().to(device)
    features = graph.ndata["feat"]

    adj = symmetric_normalize_adjacency(graph)
    I_N = sp.eye(features.shape[0])
    # Lazy random walk (also known as lazy graph convolution).
    lazy_adj = (1 - beta) * I_N + beta * adj
    lazy_adj = sparse_mx_to_torch_sparse_tensor(lazy_adj)
    # LIM track, else use both train and val set to construct this matrix.
    label_idx_mat = torch.diag(graph.ndata["train_mask"]).float().to_sparse()

    linear_clf = LinearNeuralNetwork(
        nfeat=graph.ndata["feat"].size(1),
        nclass=graph.ndata["label"].max().item() + 1,
        bias=False,
    ).to(device)

    start_time = time.time()
    res = OGC(linear_clf, features, graph, label_idx_mat)
    time_tot = time.time() - start_time

    print(f"Test Acc:{res:.4f}")
    print(f"Total Time:{time_tot:.4f}")
