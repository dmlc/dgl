"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Reference Code: https://github.com/tkipf/relational-gcn

This script trains and tests a Hetero Relational Graph Convolutional Networks 
(Hetero-RGCN) model based on the information of a full graph.

This flowchart describes the main functional sequence of the provided example.
main
│
├───> Load and preprocess full dataset
│
├───> Instantiate Hetero-RGCN model
│
├───> train
│     │
│     └───> Training loop
│           │
│           └───> Hetero-RGCN.forward
└───> test
      │
      └───> Evaluate the model
"""
import argparse
import time

import dgl
import dgl.sparse as dglsp

import numpy as np

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dgl.data.rdf import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset


class RelGraphEmbed(nn.Module):
    r"""Embedding layer for featureless heterograph."""

    def __init__(
        self,
        ntype_num,
        embed_size,
    ):
        super(RelGraphEmbed, self).__init__()
        self.embed_size = embed_size
        self.dropout = nn.Dropout(0.0)

        # Create weight embeddings for each node for each relation.
        self.embeds = nn.ParameterDict()
        for ntype, num_nodes in ntype_num.items():
            embed = nn.Parameter(th.Tensor(num_nodes, self.embed_size))
            nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain("relu"))
            self.embeds[ntype] = embed

    def forward(self):
        return self.embeds


class HeteroRelationalGraphConv(nn.Module):
    r"""HeteroRelational graph convolution layer.

    Parameters
    ----------
    in_size : int
        Input feature size.
    out_size : int
        Output feature size.
    relation_names : list[str]
        Relation names.
    """

    def __init__(
        self,
        in_size,
        out_size,
        relation_names,
        activation=None,
    ):
        super(HeteroRelationalGraphConv, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.relation_names = relation_names
        self.activation = activation

        ########################################################################
        # (HIGHLIGHT) HeteroGraphConv is a graph convolution operator over
        # heterogeneous graphs. A dictionary is passed where the key is the
        # relation name and the value is the insatnce of conv layer.
        ########################################################################
        self.W = nn.ModuleDict(
            {str(rel): nn.Linear(in_size, out_size) for rel in relation_names}
        )

        self.dropout = nn.Dropout(0.0)

    def forward(self, A, inputs):
        """Forward computation

        Parameters
        ----------
        A : Hetero Sparse Matrix
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.

        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        hs = {}
        for rel in A:
            src_type, edge_type, dst_type = rel
            if dst_type not in hs:
                hs[dst_type] = th.zeros(
                    inputs[dst_type].shape[0], self.out_size
                )
            ####################################################################
            # (HIGHLIGHT) Sparse library use hetero sparse matrix to present
            # heterogeneous graphs. A dictionary is passed where the key is
            # the tuple of (source node type, edge type, destination node type)
            # and the value is the sparse matrix contructed from the key on
            # global graph. The convolution operation is the multiplication of
            # sparse matrix and convolutional layer.
            ####################################################################
            hs[dst_type] = hs[dst_type] + (
                A[rel].T @ self.W[str(edge_type)](inputs[src_type])
            )
            if self.activation:
                hs[dst_type] = self.activation(hs[dst_type])
            hs[dst_type] = self.dropout(hs[dst_type])

        return hs


class EntityClassify(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        relation_names,
        embed_layer,
    ):
        super(EntityClassify, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.relation_names = relation_names
        self.relation_names.sort()
        self.embed_layer = embed_layer

        self.layers = nn.ModuleList()
        # Input to hidden.
        self.layers.append(
            HeteroRelationalGraphConv(
                self.in_size,
                self.in_size,
                self.relation_names,
                activation=F.relu,
            )
        )
        # Hidden to output.
        self.layers.append(
            HeteroRelationalGraphConv(
                self.in_size,
                self.out_size,
                self.relation_names,
            )
        )

    def forward(self, A):
        h = self.embed_layer()
        for layer in self.layers:
            h = layer(A, h)
        return h


def main(args):
    # Load graph data.
    if args.dataset == "aifb":
        dataset = AIFBDataset()
    elif args.dataset == "bgs":
        dataset = BGSDataset()
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

    # Split dataset into train, validate, test.
    val_idx = train_idx[: len(train_idx) // 5]
    train_idx = train_idx[len(train_idx) // 5 :]

    embed_layer = RelGraphEmbed(
        {ntype: g.num_nodes(ntype) for ntype in g.ntypes}, 16
    )

    # Create model.
    model = EntityClassify(
        16,
        num_classes,
        list(set(g.etypes)),
        embed_layer,
    )

    # Optimizer.
    optimizer = th.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0)

    # Construct hetero sparse matrix.
    A = {}
    for stype, etype, dtype in g.canonical_etypes:
        eg = g[stype, etype, dtype]
        indices = th.stack(eg.edges("uv"))
        A[(stype, etype, dtype)] = dglsp.spmatrix(
            indices, shape=(g.num_nodes(stype), g.num_nodes(dtype))
        )
        ###########################################################
        # (HIGHLIGHT) Compute the normalized adjacency matrix with
        # Sparse Matrix API
        ###########################################################
        D1_hat = dglsp.diag(A[(stype, etype, dtype)].sum(1)) ** -0.5
        D2_hat = dglsp.diag(A[(stype, etype, dtype)].sum(0)) ** -0.5
        A[(stype, etype, dtype)] = D1_hat @ A[(stype, etype, dtype)] @ D2_hat

    # Training loop.
    print("start training...")
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        logits = model(A)[category]
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
            f"Epoch {epoch:05d} | Train Acc: {train_acc:.4f} | "
            f"Train Loss: {loss.item():.4f} | Valid Acc: {val_acc:.4f} | "
            f"Valid loss: {val_loss.item():.4f} "
        )
    print()

    model.eval()
    logits = model.forward(A)[category]
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
        "-d", "--dataset", type=str, required=True, help="dataset to use"
    )

    args = parser.parse_args()
    print(args)
    main(args)
