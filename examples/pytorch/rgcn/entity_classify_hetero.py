"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn

Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""

import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.nn.pytorch import RelGraphConv
from dataset import load_entity
from functools import partial

from dgl.data.rdf import AIFB, MUTAG, BGS, AM

from model import BaseRGCN

class RelGraphEmbed(nn.Module):
    r"""Embedding layer for featureless heterograph."""
    def __init__(self,
                 g,
                 node_tids,
                 num_of_ntype,
                 input_size,
                 embed_size,
                 embed_name='embed',
                 activation=None,
                 dropout=0.0):
        super(RelGraphEmbed, self).__init__()
        self.g = g
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        self.num_of_ntype = num_of_ntype

        none_embed = nn.Parameter(torch.Tensor(g.number_of_nodes(), self.embed_size))
        nn.init.xavier_uniform_(none_embed, gain=nn.init.calculate_gain('relu'))
        self.embeds[str(-1)] = none_embed
        for ntype in range(num_of_ntype):
            if input_size[ntype] is None:
                loc = node_tids == ntype
                input_emb_size = node_tids[loc].shape[0]
                embed = nn.Parameter(torch.Tensor(input_emb_size, self.embed_size))
                nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
                self.embeds[str(ntype)] = embed
            # else, it is none_embed          

    def forward(self, node_ids, node_tids, features):
        """Forward computation

        Returns
        -------
        DGLHeteroGraph
            The block graph fed with embeddings.
        """
        embeds = self.embeds[str(-1)]
        for ntype in range(self.num_of_ntype):
            if features[ntype] is not None:
                loc = node_tids == ntype
                embeds[loc] = features[ntype] @ self.embeds[str(ntype)]

        return embeds

class EntityClassify(BaseRGCN):
    def build_input_layer(self):
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout)

    def build_hidden_layer(self, idx):
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout)

    def build_output_layer(self):
        return RelGraphConv(self.h_dim, self.out_dim, self.num_rels, "basis",
                self.num_bases, activation=None,
                self_loop=self.use_self_loop)

def main(args):
    # load graph data
    #data = load_entity(args.dataset, bfs_level=args.bfs_level, relabel=args.relabel)
    # load graph data
    if args.dataset == 'aifb':
        dataset = AIFB()
    elif args.dataset == 'mutag':
        dataset = MUTAG()
    elif args.dataset == 'bgs':
        dataset = BGS()
    elif args.dataset == 'am':
        dataset = AM()
    else:
        raise ValueError()

    g = dataset.graph
    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_idx = dataset.train_idx
    test_idx = dataset.test_idx
    labels = dataset.labels

    # split dataset into train, validate, test
    if args.validation:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx

    category_id = len(g.ntypes)
    for i, ntype in enumerate(g.ntypes):
        print(i)
        print(ntype)
        print(g.number_of_nodes(ntype))
        if ntype == category:
            category_id = i

    num_rels = len(g.canonical_etypes)
    num_classes = len(g.ntypes)
    num_of_ntype = len(g.ntypes)
    g = dgl.to_homo(g)

    # since the nodes are featureless, the input feature is then the node id.
    node_ids = torch.arange(dataset.number_of_nodes)

    # edge type and normalization factor
    edge_type = g.edata[dgl.ETYPE]
    node_tids = g.ndata[dgl.NTYPE]
    print(node_tids[node_tids.long() == (category_id -1)].shape)

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        node_ids = node_ids.cuda()
        edge_type = edge_type.cuda()
        labels = labels.cuda()
        node_tids = node_tids.cuda()

    embed_layer = RelGraphEmbed(g,
                                node_tids,
                                num_of_ntype,
                                [None] * num_of_ntype,
                                args.n_hidden)

    # create model
    model = EntityClassify(dataset.number_of_nodes,
                           args.n_hidden,
                           num_classes,
                           num_rels,
                           num_bases=args.n_bases,
                           num_hidden_layers=args.n_layers - 2,
                           dropout=args.dropout,
                           use_self_loop=args.use_self_loop,
                           use_cuda=use_cuda)

    if use_cuda:
        embed_layer.cuda()
        model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []
    model.train()
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        t0 = time.time()
        feats = embed_layer(node_ids, node_tids, [None] * num_of_ntype)
        logits = model(g, feats, edge_type, None)
        loc = (node_tids == category_id)
        logits = logits[loc]
        print(labels.shape)
        print(logits.shape)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        t1 = time.time()
        loss.backward()
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
              format(epoch, forward_time[-1], backward_time[-1]))
        train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
        val_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        print("Train Accuracy: {:.4f} | Train Loss: {:.4f} | Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
              format(train_acc, loss.item(), val_acc, val_loss.item()))
    print()

    model.eval()
    feats = embed_layer(node_ids, node_tids, [None] * num_of_ntype)
    logits = model.forward(g, feats, edge_type, None)
    loc = (node_tids == category_id)
    logits = logits[loc]
    print(logits.shape)
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    test_acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
    print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
    print()

    print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
    print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=50,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--relabel", default=False, action='store_true',
            help="remove untouched nodes and relabel")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    args.bfs_level = args.n_layers + 1 # pruning used nodes for memory
    main(args)
