import time
import argparse
import scipy.sparse as sp

import torch
import torch.nn.functional as F

from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from utils import sparse_mx_to_torch_sparse_tensor, symmetric_normalize_adjacency, LinearNeuralNetwork


# Training settings
decline = 0.9            # the dcline rate
eta_sup = 0.001          # the learning rate for supervised loss
eta_W = 0.5              # the learning rate for updating W
beta = 0.1               # in [0,1], the moving probability that a node moves to its neighbors
max_similar_tol = 0.995  # the max_tol test set label prediction similarity between two iterations
max_patience = 2         # the tolreance for consecutively getting very similar test prediction


def update_U(U, Y, predY, W):
    global eta_sup
    # ------ update the smoothness loss via LGC ------
    U = torch.spmm(lazy_adj.to(device), U)

    # ------ update the supervised loss via SEB ------
    dU_sup = 2*torch.mm(torch.sparse.mm(S, -Y + predY), W)
    U = U - eta_sup * dU_sup

    eta_sup = eta_sup * decline
    return U


def OGC(linear_clf, U, g):
    patience = 0
    _, _, last_acc, last_outp = linear_clf.test(U, g)
    for i in range(64):
        # updating W by training a simple linear supervised model Y=W*X
        predY, W = linear_clf.update_W(U, g, eta_W)

        # updating U by LGC and SEB jointly
        U = update_U(U, F.one_hot(g.ndata["label"]).float(), predY, W)

        loss_tv, acc_tv, acc_test, pred = linear_clf.test(U, g)
        print('epoch {} loss_tv {:.4f} acc_train_val {:.4f} acc_test {:.4f}'.format(
               i + 1, loss_tv, acc_tv, acc_test))      

        sim_rate = float(int((pred == last_outp).sum()) / int(pred.shape[0]))
        if (sim_rate > max_similar_tol):
            patience += 1
            if (patience > max_patience):
                break

        last_acc = acc_test
        last_outp = pred
    return last_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default="citeseer",
        choices=["cora", "citeseer", "pubmed"],
        help='Dataset to use.')
    args, _ = parser.parse_known_args()

    # load and preprocess dataset
    transform = (AddSelfLoop())
    if args.dataset == "cora":
        data = CoraGraphDataset(transform=transform)
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset(transform=transform)
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset(transform=transform)
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    g = data[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)
    features = g.ndata["feat"]

    adj = symmetric_normalize_adjacency(g)
    print(g.num_edges)
    I_N = sp.eye(features.shape[0])
    # lazy random walk (also known as lazy graph convolution)
    lazy_adj = (1 - beta) * I_N + beta * adj
    lazy_adj = sparse_mx_to_torch_sparse_tensor(lazy_adj)
    # LIM track, else use both train and validation set to construct S
    S = torch.diag(g.ndata["train_mask"]).float().to_sparse()
    
    linear_clf = LinearNeuralNetwork(nfeat=g.ndata["feat"].size(1),
                                     nclass=g.ndata["label"].max().item()+1,
                                     bias=False).to(device)

    start_time = time.time()
    res = OGC(linear_clf, features, g)
    time_tot = time.time() - start_time

    print(f'Test Acc:{res:.4f}')
    print(f'Total Time:{time_tot:.4f}')