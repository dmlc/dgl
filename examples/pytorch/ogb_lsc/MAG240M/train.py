#!/usr/bin/env python
# coding: utf-8

import argparse
import time

import dgl
import dgl.function as fn
import dgl.nn as dglnn

import numpy as np
import ogb
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from ogb.lsc import MAG240MDataset, MAG240MEvaluator


class RGAT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        num_etypes,
        num_layers,
        num_heads,
        dropout,
        pred_ntype,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()

        self.convs.append(
            nn.ModuleList(
                [
                    dglnn.GATConv(
                        in_channels,
                        hidden_channels // num_heads,
                        num_heads,
                        allow_zero_in_degree=True,
                    )
                    for _ in range(num_etypes)
                ]
            )
        )
        self.norms.append(nn.BatchNorm1d(hidden_channels))
        self.skips.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(
                nn.ModuleList(
                    [
                        dglnn.GATConv(
                            hidden_channels,
                            hidden_channels // num_heads,
                            num_heads,
                            allow_zero_in_degree=True,
                        )
                        for _ in range(num_etypes)
                    ]
                )
            )
            self.norms.append(nn.BatchNorm1d(hidden_channels))
            self.skips.append(nn.Linear(hidden_channels, hidden_channels))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )
        self.dropout = nn.Dropout(dropout)

        self.hidden_channels = hidden_channels
        self.pred_ntype = pred_ntype
        self.num_etypes = num_etypes

    def forward(self, mfgs, x):
        for i in range(len(mfgs)):
            mfg = mfgs[i]
            x_dst = x[: mfg.num_dst_nodes()]
            n_src = mfg.num_src_nodes()
            n_dst = mfg.num_dst_nodes()
            mfg = dgl.block_to_graph(mfg)
            x_skip = self.skips[i](x_dst)
            for j in range(self.num_etypes):
                subg = mfg.edge_subgraph(
                    mfg.edata["etype"] == j, relabel_nodes=False
                )
                x_skip += self.convs[i][j](subg, (x, x_dst)).view(
                    -1, self.hidden_channels
                )
            x = self.norms[i](x_skip)
            x = F.elu(x)
            x = self.dropout(x)
        return self.mlp(x)


class ExternalNodeCollator(dgl.dataloading.NodeCollator):
    def __init__(self, g, idx, sampler, offset, feats, label):
        super().__init__(g, idx, sampler)
        self.offset = offset
        self.feats = feats
        self.label = label

    def collate(self, items):
        input_nodes, output_nodes, mfgs = super().collate(items)
        # Copy input features
        mfgs[0].srcdata["x"] = torch.FloatTensor(self.feats[input_nodes])
        mfgs[-1].dstdata["y"] = torch.LongTensor(
            self.label[output_nodes - self.offset]
        )
        return input_nodes, output_nodes, mfgs


def train(args, dataset, g, feats, paper_offset):
    print("Loading masks and labels")
    train_idx = torch.LongTensor(dataset.get_idx_split("train")) + paper_offset
    valid_idx = torch.LongTensor(dataset.get_idx_split("valid")) + paper_offset
    label = dataset.paper_label

    print("Initializing dataloader...")
    sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 25])
    train_collator = ExternalNodeCollator(
        g, train_idx, sampler, paper_offset, feats, label
    )
    valid_collator = ExternalNodeCollator(
        g, valid_idx, sampler, paper_offset, feats, label
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_collator.dataset,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        collate_fn=train_collator.collate,
        num_workers=4,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_collator.dataset,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        collate_fn=valid_collator.collate,
        num_workers=2,
    )

    print("Initializing model...")
    model = RGAT(
        dataset.num_paper_features,
        dataset.num_classes,
        1024,
        5,
        2,
        4,
        0.5,
        "paper",
    ).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.25)

    best_acc = 0

    for _ in range(args.epochs):
        model.train()
        with tqdm.tqdm(train_dataloader) as tq:
            for i, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                mfgs = [g.to("cuda") for g in mfgs]
                x = mfgs[0].srcdata["x"]
                y = mfgs[-1].dstdata["y"]
                y_hat = model(mfgs, x)
                loss = F.cross_entropy(y_hat, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                acc = (y_hat.argmax(1) == y).float().mean()
                tq.set_postfix(
                    {"loss": "%.4f" % loss.item(), "acc": "%.4f" % acc.item()},
                    refresh=False,
                )

        model.eval()
        correct = total = 0
        for i, (input_nodes, output_nodes, mfgs) in enumerate(
            tqdm.tqdm(valid_dataloader)
        ):
            with torch.no_grad():
                mfgs = [g.to("cuda") for g in mfgs]
                x = mfgs[0].srcdata["x"]
                y = mfgs[-1].dstdata["y"]
                y_hat = model(mfgs, x)
                correct += (y_hat.argmax(1) == y).sum().item()
                total += y_hat.shape[0]
        acc = correct / total
        print("Validation accuracy:", acc)

        sched.step()

        if best_acc < acc:
            best_acc = acc
            print("Updating best model...")
            torch.save(model.state_dict(), args.model_path)


def test(args, dataset, g, feats, paper_offset):
    print("Loading masks and labels...")
    valid_idx = torch.LongTensor(dataset.get_idx_split("valid")) + paper_offset
    test_idx = torch.LongTensor(dataset.get_idx_split("test")) + paper_offset
    label = dataset.paper_label

    print("Initializing data loader...")
    sampler = dgl.dataloading.MultiLayerNeighborSampler([160, 160])
    valid_collator = ExternalNodeCollator(
        g, valid_idx, sampler, paper_offset, feats, label
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_collator.dataset,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        collate_fn=valid_collator.collate,
        num_workers=2,
    )
    test_collator = ExternalNodeCollator(
        g, test_idx, sampler, paper_offset, feats, label
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_collator.dataset,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        collate_fn=test_collator.collate,
        num_workers=4,
    )

    print("Loading model...")
    model = RGAT(
        dataset.num_paper_features,
        dataset.num_classes,
        1024,
        5,
        2,
        4,
        0.5,
        "paper",
    ).cuda()
    model.load_state_dict(torch.load(args.model_path))

    model.eval()
    correct = total = 0
    for i, (input_nodes, output_nodes, mfgs) in enumerate(
        tqdm.tqdm(valid_dataloader)
    ):
        with torch.no_grad():
            mfgs = [g.to("cuda") for g in mfgs]
            x = mfgs[0].srcdata["x"]
            y = mfgs[-1].dstdata["y"]
            y_hat = model(mfgs, x)
            correct += (y_hat.argmax(1) == y).sum().item()
            total += y_hat.shape[0]
    acc = correct / total
    print("Validation accuracy:", acc)
    evaluator = MAG240MEvaluator()
    y_preds = []
    for i, (input_nodes, output_nodes, mfgs) in enumerate(
        tqdm.tqdm(test_dataloader)
    ):
        with torch.no_grad():
            mfgs = [g.to("cuda") for g in mfgs]
            x = mfgs[0].srcdata["x"]
            y = mfgs[-1].dstdata["y"]
            y_hat = model(mfgs, x)
            y_preds.append(y_hat.argmax(1).cpu())
    evaluator.save_test_submission(
        {"y_pred": torch.cat(y_preds)}, args.submission_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rootdir",
        type=str,
        default=".",
        help="Directory to download the OGB dataset.",
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        default="./graph.dgl",
        help="Path to the graph.",
    )
    parser.add_argument(
        "--full-feature-path",
        type=str,
        default="./full.npy",
        help="Path to the features of all nodes.",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./model.pt",
        help="Path to store the best model.",
    )
    parser.add_argument(
        "--submission-path",
        type=str,
        default="./results",
        help="Submission directory.",
    )
    args = parser.parse_args()

    dataset = MAG240MDataset(root=args.rootdir)

    print("Loading graph")
    (g,), _ = dgl.load_graphs(args.graph_path)
    g = g.formats(["csc"])

    print("Loading features")
    paper_offset = dataset.num_authors + dataset.num_institutions
    num_nodes = paper_offset + dataset.num_papers
    num_features = dataset.num_paper_features
    feats = np.memmap(
        args.full_feature_path,
        mode="r",
        dtype="float16",
        shape=(num_nodes, num_features),
    )

    if args.epochs != 0:
        train(args, dataset, g, feats, paper_offset)
    test(args, dataset, g, feats, paper_offset)
