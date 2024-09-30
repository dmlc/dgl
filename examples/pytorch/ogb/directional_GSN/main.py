import argparse
import random

import dgl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from preprocessing import prepare_dataset
from torch.utils.data import Dataset
from tqdm import tqdm


def aggregate_mean(h, vector_field, h_in):
    return torch.mean(h, dim=1)


def aggregate_max(h, vector_field, h_in):
    return torch.max(h, dim=1)[0]


def aggregate_sum(h, vector_field, h_in):
    return torch.sum(h, dim=1)


def aggregate_dir_dx(h, vector_field, h_in, eig_idx=1):
    eig_w = (
        (vector_field[:, :, eig_idx])
        / (
            torch.sum(
                torch.abs(vector_field[:, :, eig_idx]), keepdim=True, dim=1
            )
            + 1e-8
        )
    ).unsqueeze(-1)
    h_mod = torch.mul(h, eig_w)
    return torch.abs(torch.sum(h_mod, dim=1) - torch.sum(eig_w, dim=1) * h_in)


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size):
        super(FCLayer, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.linear = nn.Linear(in_size, out_size, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight, 1 / self.in_size)
        self.linear.bias.data.zero_()

    def forward(self, x):
        h = self.linear(x)
        return h


class MLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(MLP, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.fc = FCLayer(in_size, out_size)

    def forward(self, x):
        x = self.fc(x)
        return x


class DGNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, aggregators):
        super().__init__()

        self.dropout = dropout

        self.aggregators = aggregators

        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.pretrans = MLP(in_size=2 * in_dim, out_size=in_dim)
        self.posttrans = MLP(
            in_size=(len(aggregators) * 1 + 1) * in_dim, out_size=out_dim
        )

    def pretrans_edges(self, edges):
        z2 = torch.cat([edges.src["h"], edges.dst["h"]], dim=1)
        vector_field = edges.data["eig"]
        return {"e": self.pretrans(z2), "vector_field": vector_field}

    def message_func(self, edges):
        return {
            "e": edges.data["e"],
            "vector_field": edges.data["vector_field"],
        }

    def reduce_func(self, nodes):
        h_in = nodes.data["h"]
        h = nodes.mailbox["e"]

        vector_field = nodes.mailbox["vector_field"]

        h = torch.cat(
            [
                aggregate(h, vector_field, h_in)
                for aggregate in self.aggregators
            ],
            dim=1,
        )

        return {"h": h}

    def forward(self, g, h, snorm_n):
        g.ndata["h"] = h

        # pretransformation
        g.apply_edges(self.pretrans_edges)

        # aggregation
        g.update_all(self.message_func, self.reduce_func)
        h = torch.cat([h, g.ndata["h"]], dim=1)

        # posttransformation
        h = self.posttrans(h)

        # graph and batch normalization
        h = h * snorm_n
        h = self.batchnorm_h(h)
        h = F.relu(h)

        h = F.dropout(h, self.dropout, training=self.training)

        return h


class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [
            nn.Linear(input_dim // 2**l, input_dim // 2 ** (l + 1), bias=True)
            for l in range(L)
        ]
        list_FC_layers.append(
            nn.Linear(input_dim // 2**L, output_dim, bias=True)
        )
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y


class DGNNet(nn.Module):
    def __init__(self, hidden_dim=420, out_dim=420, dropout=0.2, n_layers=4):
        super().__init__()

        self.embedding_h = AtomEncoder(emb_dim=hidden_dim)
        self.aggregators = [
            aggregate_mean,
            aggregate_sum,
            aggregate_max,
            aggregate_dir_dx,
        ]

        self.layers = nn.ModuleList(
            [
                DGNLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    dropout=dropout,
                    aggregators=self.aggregators,
                )
                for _ in range(n_layers - 1)
            ]
        )
        self.layers.append(
            DGNLayer(
                in_dim=hidden_dim,
                out_dim=out_dim,
                dropout=dropout,
                aggregators=self.aggregators,
            )
        )

        # 128 out dim since ogbg-molpcba has 128 tasks
        self.MLP_layer = MLPReadout(out_dim, 128)

    def forward(self, g, h, snorm_n):
        h = self.embedding_h(h)

        for i, conv in enumerate(self.layers):
            h_t = conv(g, h, snorm_n)
            h = h_t

        g.ndata["h"] = h

        hg = dgl.mean_nodes(g, "h")

        return self.MLP_layer(hg)

    def loss(self, scores, labels):
        is_labeled = labels == labels
        loss = nn.BCEWithLogitsLoss()(
            scores[is_labeled], labels[is_labeled].float()
        )
        return loss


def train_epoch(model, optimizer, device, data_loader):
    model.train()
    epoch_loss = 0
    epoch_train_AP = 0
    list_scores = []
    list_labels = []
    for iter, (batch_graphs, batch_labels, batch_snorm_n) in enumerate(
        data_loader
    ):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata["feat"]  # num x feat
        batch_snorm_n = batch_snorm_n.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()

        batch_scores = model(batch_graphs, batch_x, batch_snorm_n)

        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        list_scores.append(batch_scores)
        list_labels.append(batch_labels)

    epoch_loss /= iter + 1

    evaluator = Evaluator(name="ogbg-molpcba")
    epoch_train_AP = evaluator.eval(
        {"y_pred": torch.cat(list_scores), "y_true": torch.cat(list_labels)}
    )["ap"]

    return epoch_loss, epoch_train_AP


def evaluate_network(model, device, data_loader):
    model.eval()
    epoch_test_loss = 0
    epoch_test_AP = 0
    with torch.no_grad():
        list_scores = []
        list_labels = []
        for iter, (batch_graphs, batch_labels, batch_snorm_n) in enumerate(
            data_loader
        ):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata["feat"]
            batch_snorm_n = batch_snorm_n.to(device)
            batch_labels = batch_labels.to(device)

            batch_scores = model(batch_graphs, batch_x, batch_snorm_n)

            loss = model.loss(batch_scores, batch_labels)
            epoch_test_loss += loss.item()
            list_scores.append(batch_scores)
            list_labels.append(batch_labels)

        epoch_test_loss /= iter + 1

        evaluator = Evaluator(name="ogbg-molpcba")
        epoch_test_AP = evaluator.eval(
            {"y_pred": torch.cat(list_scores), "y_true": torch.cat(list_labels)}
        )["ap"]

    return epoch_test_loss, epoch_test_AP


def train(dataset, params):
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
    device = params.device

    print("Training Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))

    model = DGNNet()
    model = model.to(device)

    # view model parameters
    total_param = 0
    print("MODEL DETAILS:\n")
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print("DGN Total parameters:", total_param)

    optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=8
    )

    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_APs, epoch_val_APs, epoch_test_APs = [], [], []

    train_loader = GraphDataLoader(
        trainset,
        batch_size=params.batch_size,
        shuffle=True,
        collate_fn=dataset.collate,
        pin_memory=True,
    )
    val_loader = GraphDataLoader(
        valset,
        batch_size=params.batch_size,
        shuffle=False,
        collate_fn=dataset.collate,
        pin_memory=True,
    )
    test_loader = GraphDataLoader(
        testset,
        batch_size=params.batch_size,
        shuffle=False,
        collate_fn=dataset.collate,
        pin_memory=True,
    )

    with tqdm(range(450), unit="epoch") as t:
        for epoch in t:
            t.set_description("Epoch %d" % epoch)

            epoch_train_loss, epoch_train_ap = train_epoch(
                model, optimizer, device, train_loader
            )
            epoch_val_loss, epoch_val_ap = evaluate_network(
                model, device, val_loader
            )

            epoch_train_losses.append(epoch_train_loss)
            epoch_val_losses.append(epoch_val_loss)
            epoch_train_APs.append(epoch_train_ap.item())
            epoch_val_APs.append(epoch_val_ap.item())

            _, epoch_test_ap = evaluate_network(model, device, test_loader)

            epoch_test_APs.append(epoch_test_ap.item())

            t.set_postfix(
                train_loss=epoch_train_loss,
                train_AP=epoch_train_ap.item(),
                val_AP=epoch_val_ap.item(),
                refresh=False,
            )

            scheduler.step(-epoch_val_ap.item())

            if optimizer.param_groups[0]["lr"] < 1e-5:
                print("\n!! LR EQUAL TO MIN LR SET.")
                break

            print("")

    best_val_epoch = np.argmax(np.array(epoch_val_APs))
    best_train_epoch = np.argmax(np.array(epoch_train_APs))
    best_val_ap = epoch_val_APs[best_val_epoch]
    best_val_test_ap = epoch_test_APs[best_val_epoch]
    best_val_train_ap = epoch_train_APs[best_val_epoch]
    best_train_ap = epoch_train_APs[best_train_epoch]

    print("Best Train AP: {:.4f}".format(best_train_ap))
    print("Best Val AP: {:.4f}".format(best_val_ap))
    print("Test AP of Best Val: {:.4f}".format(best_val_test_ap))
    print("Train AP of Best Val: {:.4f}".format(best_val_train_ap))


class Subset(object):
    def __init__(self, dataset, labels, indices):
        dataset = [dataset[idx] for idx in indices]
        labels = [labels[idx] for idx in indices]
        self.dataset, self.labels = [], []
        for i, g in enumerate(dataset):
            if g.num_nodes() > 5:
                self.dataset.append(g)
                self.labels.append(labels[i])
        self.len = len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]

    def __len__(self):
        return self.len


class PCBADataset(Dataset):
    def __init__(self, name):
        print("[I] Loading dataset %s..." % (name))
        self.name = name

        self.dataset, self.split_idx = prepare_dataset(name)
        print("One hot encoding substructure counts... ", end="")
        self.d_id = [1] * self.dataset[0].edata["subgraph_counts"].shape[1]

        for g in self.dataset:
            g.edata["eig"] = g.edata["subgraph_counts"].float()

        self.train = Subset(
            self.dataset, self.split_idx["label"], self.split_idx["train"]
        )
        self.val = Subset(
            self.dataset, self.split_idx["label"], self.split_idx["valid"]
        )
        self.test = Subset(
            self.dataset, self.split_idx["label"], self.split_idx["test"]
        )

        print(
            "train, test, val sizes :",
            len(self.train),
            len(self.test),
            len(self.val),
        )
        print("[I] Finished loading.")

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.stack(labels)

        tab_sizes_n = [g.num_nodes() for g in graphs]
        tab_snorm_n = [
            torch.FloatTensor(size, 1).fill_(1.0 / size) for size in tab_sizes_n
        ]
        snorm_n = torch.cat(tab_snorm_n).sqrt()
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels, snorm_n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu_id", default=0, type=int, help="Please give a value for gpu id"
    )
    parser.add_argument(
        "--seed", default=41, type=int, help="Please give a value for seed"
    )
    parser.add_argument(
        "--batch_size",
        default=2048,
        type=int,
        help="Please give a value for batch_size",
    )
    args = parser.parse_args()

    # device
    args.device = torch.device(
        "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu"
    )

    # setting seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    dataset = PCBADataset("ogbg-molpcba")
    train(dataset, args)
