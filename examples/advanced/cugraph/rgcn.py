"""
[RGCN: Relational Graph Convolutional Networks]
(https://arxiv.org/abs/1703.06103)

This example showcases the usage of `CuGraphRelGraphConv` via the entity
classification problem in the RGCN paper with mini-batch training. It offers
a 1.5~2x speed-up over `RelGraphConv` on cuda devices and only requires minimal
code changes from the current `entity_sample.py` example.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import dgl
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from dgl.dataloading import MultiLayerNeighborSampler, DataLoader
from dgl.nn import CuGraphRelGraphConv
import argparse


class RGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases, fanouts):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, h_dim)
        # two-layer RGCN
        self.conv1 = CuGraphRelGraphConv(
            h_dim,
            h_dim,
            num_rels,
            regularizer="basis",
            num_bases=num_bases,
            self_loop=False,
            max_in_degree=fanouts[0]
        )
        self.conv2 = CuGraphRelGraphConv(
            h_dim,
            out_dim,
            num_rels,
            regularizer="basis",
            num_bases=num_bases,
            self_loop=False,
            max_in_degree=fanouts[1]
        )

    def forward(self, g):
        x = self.emb(g[0].srcdata[dgl.NID])
        h = F.relu(self.conv1(g[0], x, g[0].edata[dgl.ETYPE],
                   norm=g[0].edata["norm"]))
        h = self.conv2(g[1], h, g[1].edata[dgl.ETYPE], norm=g[1].edata["norm"])
        return h

    def update_max_in_degree(self, fanouts):
        self.conv1.max_in_degree = fanouts[0]
        self.conv2.max_in_degree = fanouts[1]


def evaluate(model, labels, dataloader, inv_target):
    model.eval()
    eval_logits = []
    eval_seeds = []
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in dataloader:
            output_nodes = inv_target[output_nodes.type(torch.int64)]
            for block in blocks:
                block.edata["norm"] = dgl.norm_by_dst(block).unsqueeze(1)
            logits = model(blocks)
            eval_logits.append(logits.cpu().detach())
            eval_seeds.append(output_nodes.cpu().detach())
    eval_logits = torch.cat(eval_logits)
    eval_seeds = torch.cat(eval_seeds)
    return accuracy(eval_logits.argmax(dim=1), labels[eval_seeds].cpu()).item()


def train(device, g, target_idx, labels, train_mask, model, fanouts):
    # Define train idx, loss function and optimizer.
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    # Construct sampler and dataloader.
    sampler = MultiLayerNeighborSampler(fanouts)
    train_loader = DataLoader(
        g,
        target_idx[train_idx].type(g.idtype),
        sampler,
        device=device,
        batch_size=100,
        shuffle=True,
    )
    # No separate validation subset, use train index instead for validation.
    val_loader = DataLoader(
        g,
        target_idx[train_idx].type(g.idtype),
        sampler,
        device=device,
        batch_size=100,
        shuffle=False,
    )
    for epoch in range(100):
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
            output_nodes = inv_target[output_nodes.type(torch.int64)]
            for block in blocks:
                block.edata["norm"] = dgl.norm_by_dst(block).unsqueeze(1)
            logits = model(blocks)
            loss = loss_fcn(logits, labels[output_nodes])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        acc = evaluate(model, labels, val_loader, inv_target)
        print(
            f"Epoch {epoch:05d} | Loss {total_loss / (it+1):.4f} | "
            f"Val. Accuracy {acc:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RGCN for entity classification with sampling"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="aifb",
        choices=['aifb', 'mutag', 'bgs', 'am'],
    )
    args = parser.parse_args()
    device = torch.device("cuda")
    print(f"Training with DGL CuGraphRelGraphConv module with sampling.")

    # Load and preprocess dataset.
    if args.dataset == "aifb":
        data = AIFBDataset()
    elif args.dataset == "mutag":
        data = MUTAGDataset()
    elif args.dataset == "bgs":
        data = BGSDataset()
    elif args.dataset == "am":
        data = AMDataset()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    hg = data[0].to(device)
    num_rels = len(hg.canonical_etypes)
    category = data.predict_category

    labels = hg.nodes[category].data.pop("labels")
    train_mask = hg.nodes[category].data.pop("train_mask")
    test_mask = hg.nodes[category].data.pop("test_mask")

    # Find target category and node id.
    category_id = hg.ntypes.index(category)
    g = dgl.to_homogeneous(hg)
    node_ids = torch.arange(g.num_nodes()).to(device)
    target_idx = node_ids[g.ndata[dgl.NTYPE] == category_id]
    g.ndata["ntype"] = g.ndata.pop(dgl.NTYPE)
    g.ndata["type_id"] = g.ndata.pop(dgl.NID)

    # Find the mapping from global node IDs to type-specific node IDs.
    inv_target = torch.empty((g.num_nodes(),), dtype=torch.int64).to(device)
    inv_target[target_idx] = torch.arange(
        0, target_idx.shape[0], dtype=inv_target.dtype
    ).to(device)

    # Create RGCN model.
    in_size = g.num_nodes()  # featureless with one-hot encoding
    out_size = data.num_classes
    num_bases = 20
    fanouts = [4, 4]
    model = RGCN(in_size, 16, out_size, num_rels, num_bases, fanouts).to(device)

    train(device, g, target_idx, labels, train_mask, model, fanouts)
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
    # Note: cugraph-ops aggregators are designed for sampled graphs (MFGs) and
    # expect max_in_degree as input for performance considerations. Hence, we
    # have to update max_in_degree with the fanouts of test_sampler.
    test_sampler = MultiLayerNeighborSampler([500, 500])
    model.update_max_in_degree(test_sampler.fanouts)
    test_loader = DataLoader(
        g,
        target_idx[test_idx].type(g.idtype),
        test_sampler,
        device=device,
        batch_size=32,
        shuffle=False,
    )
    acc = evaluate(model, labels, test_loader, inv_target)
    print(f"Test accuracy {acc:.4f}")
