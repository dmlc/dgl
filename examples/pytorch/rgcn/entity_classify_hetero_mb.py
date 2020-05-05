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
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
from dgl import DGLGraph
from dataset import load_entity
from functools import partial

from dgl.data.rdf import AIFB, MUTAG, BGS, AM

from model import RelGraphEmbedLayer, RelGraphConvLayer

class EntityClassify(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=False):
        super(EntityClassify, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda

        self.layers = nn.ModuleList()

        # i2h
        self.layers.append(RelGraphConvLayer(
            self.h_dim, self.h_dim, self.num_rels, "basis",
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            dropout=self.dropout))
        # h2h
        for idx in range(self.num_hidden_layers):
            self.layers.append(RelGraphConvLayer(
                self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout))
        # h2o
        self.layers.append(RelGraphConvLayer(
            self.h_dim, self.out_dim, self.num_rels, "basis",
            self.num_bases, activation=None,
            self_loop=self.use_self_loop))

    def forward(self, blocks, feats, norm=None):
        if blocks is None:
            # full graph training
            blocks = [self.g] * len(self.layers)
        h = feats
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        return h

class NeighborSampler:
    """Neighbor sampler

    Parameters
    ----------
    g : DGLGraph
        Full graph
    fanouts : list of int
        Fanout of each hop starting from the seed nodes. If a fanout is None,
        sample full neighbors.
    """
    def __init__(self, g, target_idx, fanouts, device):
        self.g = g
        self.target_idx = target_idx
        self.fanouts = fanouts
        self.device = device

    def sample_blocks(self, seeds):
        blocks = []
        etypes = []
        norms = []
        ntypes = []
        seeds = th.tensor(seeds).long()
        cur = self.target_idx[seeds]
        for fanout in self.fanouts:
            if fanout is None or fanout == -1:
                frontier = dgl.in_subgraph(self.g, cur)
            else:
                frontier = dgl.sampling.sample_neighbors(self.g, cur, fanout)
            etypes = self.g.edata[dgl.ETYPE][frontier.edata[dgl.EID]].to(self.device)
            norm = self.g.edata['norm'][frontier.edata[dgl.EID]].to(self.device)
            block = dgl.to_block(frontier, cur)
            block.srcdata[dgl.NTYPE] = self.g.ndata[dgl.NTYPE][block.srcdata[dgl.NID]].to(self.device)
            block.edata['etype'] = etypes
            block.edata['norm'] = norm
            cur = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return seeds, blocks

def main(args):
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

    # Load from hetero-graph
    hg = dataset.graph
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

    num_rels = len(hg.canonical_etypes)
    num_of_ntype = len(hg.ntypes)

    # calculate norm for each edge type and store in edge
    for canonical_etypes in hg.canonical_etypes:
        u, v, eid = hg.all_edges(form='all', etype=canonical_etypes)
        _, inverse_index, count = th.unique(v, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = th.ones(eid.shape[0]) / degrees
        norm = norm.unsqueeze(1)
        hg.edges[canonical_etypes].data['norm'] = norm
    
    # get target category id
    category_id = len(hg.ntypes)
    for i, ntype in enumerate(hg.ntypes):
        if ntype == category:
            category_id = hg.nodes[ntype].data[dgl.NTYPE][0]

    g = dgl.to_homo(hg)
    node_ids = th.arange(g.number_of_nodes())

    # edge type and normalization factor
    edge_type = g.edata[dgl.ETYPE]
    node_tids = g.ndata[dgl.NTYPE]
    loc = (node_tids == category_id)
    target_idx = node_ids[loc]
    print(th.max(edge_type))

    # check cuda
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)
        node_ids = node_ids.cuda()
        edge_type = edge_type.cuda()
        labels = labels.cuda()
        node_tids = node_tids.cuda()
        g.edata['norm'] = g.edata['norm'].cuda()

    embed_layer = RelGraphEmbedLayer(g,
                                    node_tids,
                                    num_of_ntype,
                                    [None] * num_of_ntype,
                                    args.n_hidden)

    # create model
    model = EntityClassify(g.number_of_nodes(),
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

    sampler = NeighborSampler(g, target_idx, [args.fanout] * args.n_layers, args.gpu)
    loader = DataLoader(dataset=train_idx.numpy(),
                        batch_size=args.batch_size,
                        collate_fn=sampler.sample_blocks,
                        shuffle=True,
                        num_workers=0)

    # validation sampler
    val_sampler = NeighborSampler(g, target_idx, [None] * args.n_layers, args.gpu)
    val_loader = DataLoader(dataset=val_idx.numpy(),
                            batch_size=args.batch_size,
                            collate_fn=val_sampler.sample_blocks,
                            shuffle=False,
                            num_workers=0)

    # validation sampler
    test_sampler = NeighborSampler(g, target_idx, [None] * args.n_layers, args.gpu)
    test_loader = DataLoader(dataset=test_idx.numpy(),
                             batch_size=args.batch_size,
                             collate_fn=test_sampler.sample_blocks,
                             shuffle=False,
                             num_workers=0)

    # optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []

    for epoch in range(args.n_epochs):
        model.train()
        optimizer.zero_grad()

        for i, (seeds, blocks) in enumerate(loader):
            t0 = time.time()
            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                blocks[0].srcdata[dgl.NTYPE],
                                [None] * num_of_ntype)
            logits = model(blocks, feats)

            loss = F.cross_entropy(logits, labels[seeds])
            t1 = time.time()
            loss.backward()
            optimizer.step()
            t2 = time.time()

            forward_time.append(t1 - t0)
            backward_time.append(t2 - t1)
            print("Epoch {:05d}:{:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
                   format(epoch, i, forward_time[-1], backward_time[-1]))
            train_acc = th.sum(logits.argmax(dim=1) == labels[seeds]).item() / len(seeds)
            print("Train Accuracy: {:.4f} | Train Loss: {:.4f}".
                  format(train_acc, loss.item()))

        model.eval()
        eval_logtis = []
        eval_seeds = []
        for i, (seeds, blocks) in enumerate(val_loader):
            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                blocks[0].srcdata[dgl.NTYPE],
                                [None] * num_of_ntype)
            logits = model(blocks, feats)
            eval_logtis.append(logits)
            eval_seeds.append(seeds)
        eval_logtis = th.cat(eval_logtis)
        eval_seeds = th.cat(eval_seeds)
        val_loss = F.cross_entropy(eval_logtis, labels[eval_seeds])
        val_acc = th.sum(eval_logtis.argmax(dim=1) == labels[eval_seeds]).item() / len(eval_seeds)
        print("Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
                  format(val_acc, val_loss.item()))
    print()

    model.eval()
    test_logtis = []
    test_seeds = []
    for i, (seeds, blocks) in enumerate(test_loader):
        feats = embed_layer(blocks[0].srcdata[dgl.NID],
                            blocks[0].srcdata[dgl.NTYPE],
                            [None] * num_of_ntype)
        logits = model(blocks, feats)
        test_logtis.append(logits)
        test_seeds.append(seeds)
    test_logtis = th.cat(test_logtis)
    test_seeds = th.cat(test_seeds)
    test_loss = F.cross_entropy(test_logtis, labels[test_seeds])
    test_acc = th.sum(test_logtis.argmax(dim=1) == labels[test_seeds]).item() / len(test_seeds)
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
    parser.add_argument("--fanout", type=int, default=4,
            help="Fan-out of neighbor sampling.")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.add_argument("--batch-size", type=int, default=100,
            help="Mini-batch size. ")
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    args.bfs_level = args.n_layers + 1 # pruning used nodes for memory
    main(args)
