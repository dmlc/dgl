"""
[Modeling Relational Data with Graph Convolutional Networks]
(https://arxiv.org/abs/1703.06103)

Reference Code: https://github.com/tkipf/relational-gcn
"""
import argparse

import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.data.rdf import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset


class RelGraphConvLayer(nn.Module):
    def __init__(self, in_size, out_size, relation_names):
        super(RelGraphConvLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.relation_names = relation_names
        self.weight = {
            rel: nn.Parameter(torch.Tensor(in_size, out_size))
            for rel in self.relation_names
        }
        for w in self.weight.values():
            nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain("relu"))

    def forward(self, adjs, x_dict):
        h_dict = {ntype: 0 for ntype in x_dict.keys()}
        for stype, etype, dtype in adjs.keys():
            h = x_dict[stype] @ self.weight[etype]  # dense mm
            h_dict[dtype] += adjs[(stype, etype, dtype)] @ h
        h_dict = {ntype: torch.relu(h) for ntype, h in h_dict.items()}
        return h_dict


class EntityClassify(nn.Module):
    def __init__(self, adjs, ntype_to_num_nodes, h_dim, out_dim):
        super(EntityClassify, self).__init__()
        self.adjs = adjs
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.relation_names = list(set(etype for _, etype, _ in adjs.keys()))

        self.embeds = nn.ParameterDict()
        for ntype, num_nodes in ntype_to_num_nodes.items():
            embed = nn.Parameter(torch.Tensor(num_nodes, h_dim))
            nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain("relu"))
            self.embeds[ntype] = embed

        self.layers = nn.ModuleList()
        self.layers.append(
            RelGraphConvLayer(self.h_dim, self.h_dim, self.relation_names)
        )
        self.layers.append(
            RelGraphConvLayer(self.h_dim, self.out_dim, self.relation_names)
        )

    def forward(self):
        h = self.embeds
        for layer in self.layers:
            h = layer(self.adjs, h)
        return h


def evaluate(g, logits, category):
    labels = g.nodes[category].data["labels"]
    train_mask = g.nodes[category].data["train_mask"]
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    test_mask = g.nodes[category].data["test_mask"]
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
    val_idx = train_idx

    val_acc = torch.sum(
        logits[val_idx].argmax(dim=1) == labels[val_idx]
    ).item() / len(val_idx)
    test_acc = torch.sum(
        logits[test_idx].argmax(dim=1) == labels[test_idx]
    ).item() / len(test_idx)
    return val_acc, test_acc


def train(model, g, category):

    labels = g.nodes[category].data["labels"]
    train_mask = g.nodes[category].data["train_mask"]
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    model.train()
    for epoch in range(50):
        # Forward
        logits = model()[category]
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute prediction.
        val_acc, test_acc = evaluate(g, logits, category)
        print(
            f"In epoch {epoch}, loss: {loss:.3f}, val acc: {val_acc:.3f}, test"
            f" acc: {test_acc:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGCN")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Dataset name ('aifb', 'mutag', 'bgs', 'am').",
    )

    args = parser.parse_args()

    # Load graph data
    if args.dataset == "aifb":
        dataset = AIFBDataset()
    elif args.dataset == "mutag":
        dataset = MUTAGDataset()
    elif args.dataset == "bgs":
        dataset = BGSDataset()
    elif args.dataset == "am":
        dataset = AMDataset()
    else:
        raise ValueError()

    g = dataset[0]

    # Create a matrix dictionary from a heterogeneous graph
    def create_adjs(g):
        adjs = {}
        for rel in g.canonical_etypes:
            stype, _, dtype = rel
            row, col = g.edges(etype=rel)
            adjs[rel] = dglsp.spmatrix(
                torch.stack([col, row]),
                shape=(g.num_nodes(dtype), g.num_nodes(stype)),
            )
        return adjs

    # Dict: canonical_etype -> SparseMatrix
    adjs = create_adjs(g)
    ntype_to_num_nodes = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
    model = EntityClassify(adjs, ntype_to_num_nodes, 16, dataset.num_classes)
    print(dataset.predict_category)

    # Kick off training
    train(model, g, dataset.predict_category)
