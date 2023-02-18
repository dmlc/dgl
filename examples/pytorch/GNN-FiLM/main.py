import argparse
import os

import dgl
import dgl.function as fn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader import load_PPI
from utils import evaluate_f1_score


class GNNFiLMLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes, dropout=0.1):
        super(GNNFiLMLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        # weights for different types of edges
        self.W = nn.ModuleDict(
            {name: nn.Linear(in_size, out_size, bias=False) for name in etypes}
        )

        # hypernets to learn the affine functions for different types of edges
        self.film = nn.ModuleDict(
            {
                name: nn.Linear(in_size, 2 * out_size, bias=False)
                for name in etypes
            }
        )

        # layernorm before each propogation
        self.layernorm = nn.LayerNorm(out_size)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, feat_dict):
        # the input graph is a multi-relational graph, so treated as hetero-graph.

        funcs = {}  # message and reduce functions dict
        # for each type of edges, compute messages and reduce them all
        for srctype, etype, dsttype in g.canonical_etypes:
            messages = self.W[etype](
                feat_dict[srctype]
            )  # apply W_l on src feature
            film_weights = self.film[etype](
                feat_dict[dsttype]
            )  # use dst feature to compute affine function paras
            gamma = film_weights[
                :, : self.out_size
            ]  # "gamma" for the affine function
            beta = film_weights[
                :, self.out_size :
            ]  # "beta" for the affine function
            messages = gamma * messages + beta  # compute messages
            messages = F.relu_(messages)
            g.nodes[srctype].data[etype] = messages  # store in ndata
            funcs[etype] = (
                fn.copy_u(etype, "m"),
                fn.sum("m", "h"),
            )  # define message and reduce functions
        g.multi_update_all(
            funcs, "sum"
        )  # update all, reduce by first type-wisely then across different types
        feat_dict = {}
        for ntype in g.ntypes:
            feat_dict[ntype] = self.dropout(
                self.layernorm(g.nodes[ntype].data["h"])
            )  # apply layernorm and dropout
        return feat_dict


class GNNFiLM(nn.Module):
    def __init__(
        self, etypes, in_size, hidden_size, out_size, num_layers, dropout=0.1
    ):
        super(GNNFiLM, self).__init__()
        self.film_layers = nn.ModuleList()
        self.film_layers.append(
            GNNFiLMLayer(in_size, hidden_size, etypes, dropout)
        )
        for i in range(num_layers - 1):
            self.film_layers.append(
                GNNFiLMLayer(hidden_size, hidden_size, etypes, dropout)
            )
        self.predict = nn.Linear(hidden_size, out_size, bias=True)

    def forward(self, g, out_key):
        h_dict = {
            ntype: g.nodes[ntype].data["feat"] for ntype in g.ntypes
        }  # prepare input feature dict
        for layer in self.film_layers:
            h_dict = layer(g, h_dict)
        h = self.predict(
            h_dict[out_key]
        )  # use the final embed to predict, out_size = num_classes
        h = torch.sigmoid(h)
        return h


def main(args):
    # Step 1: Prepare graph data and retrieve train/validation/test dataloader ============================= #
    if args.gpu >= 0 and torch.cuda.is_available():
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    if args.dataset == "PPI":
        train_set, valid_set, test_set, etypes, in_size, out_size = load_PPI(
            args.batch_size, device
        )

    # Step 2: Create model and training components=========================================================== #
    model = GNNFiLM(
        etypes, in_size, args.hidden_size, out_size, args.num_layers
    ).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.step_size, gamma=args.gamma
    )

    # Step 4: training epoches ============================================================================== #
    lastf1 = 0
    cnt = 0
    best_val_f1 = 0
    for epoch in range(args.max_epoch):
        train_loss = []
        train_f1 = []
        val_loss = []
        val_f1 = []
        model.train()
        for batch in train_set:
            g = batch.graph
            g = g.to(device)
            logits = model.forward(g, "_N")
            labels = batch.label
            loss = criterion(logits, labels)
            f1 = evaluate_f1_score(
                logits.detach().cpu().numpy(), labels.detach().cpu().numpy()
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_f1.append(f1)

        train_loss = np.mean(train_loss)
        train_f1 = np.mean(train_f1)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            for batch in valid_set:
                g = batch.graph
                g = g.to(device)
                logits = model.forward(g, "_N")
                labels = batch.label
                loss = criterion(logits, labels)
                f1 = evaluate_f1_score(
                    logits.detach().cpu().numpy(), labels.detach().cpu().numpy()
                )
                val_loss.append(loss.item())
                val_f1.append(f1)

        val_loss = np.mean(val_loss)
        val_f1 = np.mean(val_f1)
        print(
            "Epoch {:d} | Train Loss {:.4f} | Train F1 {:.4f} | Val Loss {:.4f} | Val F1 {:.4f} |".format(
                epoch + 1, train_loss, train_f1, val_loss, val_f1
            )
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                model.state_dict(), os.path.join(args.save_dir, args.name)
            )

        if val_f1 < lastf1:
            cnt += 1
            if cnt == args.early_stopping:
                print("Early stop.")
                break
        else:
            cnt = 0
            lastf1 = val_f1

    model.eval()
    test_loss = []
    test_f1 = []
    model.load_state_dict(torch.load(os.path.join(args.save_dir, args.name)))
    with torch.no_grad():
        for batch in test_set:
            g = batch.graph
            g = g.to(device)
            logits = model.forward(g, "_N")
            labels = batch.label
            loss = criterion(logits, labels)
            f1 = evaluate_f1_score(
                logits.detach().cpu().numpy(), labels.detach().cpu().numpy()
            )
            test_loss.append(loss.item())
            test_f1.append(f1)
    test_loss = np.mean(test_loss)
    test_f1 = np.mean(test_f1)

    print("Test F1: {:.4f} | Test loss: {:.4f}".format(test_f1, test_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN-FiLM")
    parser.add_argument(
        "--dataset",
        type=str,
        default="PPI",
        help="DGL dataset for this GNN-FiLM",
    )
    parser.add_argument(
        "--gpu", type=int, default=-1, help="GPU Index. Default: -1, using CPU."
    )
    parser.add_argument(
        "--in_size", type=int, default=50, help="Input dimensionalities"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=320,
        help="Hidden layer dimensionalities",
    )
    parser.add_argument(
        "--out_size", type=int, default=121, help="Output dimensionalities"
    )
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Number of GNN layers"
    )
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=1500,
        help="The max number of epoches. Default: 500",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=80,
        help="Early stopping. Default: 50",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate. Default: 3e-1"
    )
    parser.add_argument(
        "--wd", type=float, default=0.0009, help="Weight decay. Default: 3e-1"
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=40,
        help="Period of learning rate decay.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.8,
        help="Multiplicative factor of learning rate decay.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate. Default: 0.9"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./out", help="Path to save the model."
    )
    parser.add_argument(
        "--name", type=str, default="GNN-FiLM", help="Saved model name."
    )

    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    main(args)
