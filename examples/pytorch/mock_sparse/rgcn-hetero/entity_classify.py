"""Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Reference Code: https://github.com/tkipf/relational-gcn
"""
import argparse
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset

from dgl.mock_sparse import create_from_coo


class RelGraphConvLayer(nn.Module):
    def __init__(self, in_feat, out_feat, rel_names):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.weight = {
            rel: nn.Parameter(th.Tensor(in_feat, out_feat))
            for rel in self.rel_names
        }
        for w in self.weight.values():
            nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain("relu"))

    def forward(self, adjs, x_dict):
        h_dict = {ntype: 0 for ntype in x_dict.keys()}
        for stype, etype, dtype in adjs.keys():
            h = x_dict[stype] @ self.weight[etype]  # dense mm
            h_dict[dtype] += adjs[(stype, etype, dtype)] @ h
        h_dict = {ntype: th.relu(h) for ntype, h in h_dict.items()}
        return h_dict


class EntityClassify(nn.Module):
    def __init__(self, adjs, ntype_to_num_nodes, h_dim, out_dim):
        super(EntityClassify, self).__init__()
        self.adjs = adjs
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.rel_names = list(set(etype for _, etype, _ in adjs.keys()))

        self.embeds = nn.ParameterDict()
        for ntype, num_nodes in ntype_to_num_nodes.items():
            embed = nn.Parameter(th.Tensor(num_nodes, h_dim))
            nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain("relu"))
            self.embeds[ntype] = embed

        self.layers = nn.ModuleList()
        self.layers.append(
            RelGraphConvLayer(self.h_dim, self.h_dim, self.rel_names)
        )
        self.layers.append(
            RelGraphConvLayer(self.h_dim, self.out_dim, self.rel_names)
        )

    def forward(self):
        h = self.embeds
        for layer in self.layers:
            h = layer(self.adjs, h)
        return h


def main(args):
    # load graph data
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
    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_mask = g.nodes[category].data.pop("train_mask")
    test_mask = g.nodes[category].data.pop("test_mask")
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    labels = g.nodes[category].data.pop("labels")
    val_idx = train_idx

    def create_adjs(g):
        adjs = {}
        for rel in g.canonical_etypes:
            stype, _, dtype = rel
            row, col = g.edges(etype=rel)
            adjs[rel] = create_from_coo(
                col,
                row,
                shape=(g.number_of_nodes(dtype), g.number_of_nodes(stype)),
            )
        return adjs

    # Dict: canonical_etype -> SparseMatrix
    adjs = create_adjs(g)
    ntype_to_num_nodes = {ntype: g.number_of_nodes(ntype) for ntype in g.ntypes}
    model = EntityClassify(adjs, ntype_to_num_nodes, 16, num_classes)

    optimizer = th.optim.Adam(model.parameters(), lr=1e-2)

    print("start training...")
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        logits = model()[category]
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        train_acc = th.sum(
            logits[train_idx].argmax(dim=1) == labels[train_idx]
        ).item() / len(train_idx)
        val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
        val_acc = th.sum(
            logits[val_idx].argmax(dim=1) == labels[val_idx]
        ).item() / len(val_idx)
        print(
            "Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Valid Acc: {:.4f} | Valid loss: {:.4f}".format(
                epoch,
                train_acc,
                loss.item(),
                val_acc,
                val_loss.item(),
            )
        )
    print()

    model.eval()
    logits = model.forward()[category]
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    test_acc = th.sum(
        logits[test_idx].argmax(dim=1) == labels[test_idx]
    ).item() / len(test_idx)
    print(
        "Test Acc: {:.4f} | Test loss: {:.4f}".format(
            test_acc, test_loss.item()
        )
    )
    print()


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
    print(args)
    main(args)
