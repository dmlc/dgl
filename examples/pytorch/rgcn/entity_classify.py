"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn

Difference compared to tkipf/relation-gcn
* edge directions are reversed (kipf did not transpose adj before spmv)
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
from dgl.data import load_data
from functools import partial

from layers import RGCNBasisLayer as RGCNLayer
from model import RGCN

class EntityClassify(RGCN):
    '''
    def __init__():
        super(EntityClassify, self).__init__()
    '''

    def build_input_layer(self):
        return RGCNLayer(len(self.g), self.h_dim, len(self.subgraphs), self.num_rels, self.num_bases, activation=F.relu, featureless=True)

    def build_hidden_layer(self):
        return RGCNLayer(self.h_dim, self.h_dim, len(self.subgraphs), self.num_rels, self.num_bases, activation=F.relu)

    def build_output_layer(self):
        return RGCNLayer(self.h_dim, self.out_dim, len(self.subgraphs), self.num_rels,self.num_bases, activation=partial(F.softmax, dim=1))


def main(args):
    # load graph data
    data = load_data(args)
    num_nodes = data.num_nodes
    edges = data.edges
    relations = data.relations
    labels = data.labels
    train_idx = data.train_idx
    test_idx = data.test_idx

    if args.relation_limit > 0 and args.relation_limit < relations.shape[1]:
        print("using first {} relaitons".format(args.relation_limit))
        relations = relations[:, :args.relation_limit]

    # split dataset into train, validate, test
    if args.validation:
        train_idx = train_idx[len(train_idx) // 5:]
        val_idx = train_idx[:len(train_idx) // 5]
    else:
        val_idx = train_idx

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    # create graph
    g = DGLGraph()
    g.add_nodes_from(np.arange(num_nodes))
    g.add_edges_from(edges)


    # create model
    model = EntityClassify(g,
                           args.n_hidden,
                           labels.shape[1],
                           relations,
                           num_bases=args.n_bases,
                           num_layers=args.n_layers,
                           dropout=args.dropout,
                           use_cuda=use_cuda)

    # convert to pytorch label format
    labels = np.argmax(labels, axis=1)
    labels = torch.from_numpy(labels).view(-1)

    # hack to make subgraph merging work for featureless case
    features = torch.arange(len(g), dtype=torch.float).view(-1, 1)

    if use_cuda:
        model.cuda()
        labels = labels.cuda()
        features = features.cuda()

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
        logits = model.forward(features)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        t1 = time.time()
        loss.backward()
        t2 = time.time()

        optimizer.step()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".format(epoch, forward_time[-1], backward_time[-1]))
        train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
        val_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        print("Train Accuracy: {:.4f} | Train Loss: {:.4f} | Validation Accuracy: {:.4f} | Validation loss: {:.4f}".format(train_acc, loss.item(), val_acc, val_loss.item()))
    print()

    model.eval()
    logits = model.forward(features)
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
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=50,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("-r", "--relation-limit", type=int, default=-1,
            help="max number of relations to use")
    parser.add_argument("--relabel", default=False, action='store_true',
            help="remove untouched nodes and relabel")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    args.bfs_level = args.n_layers + 1 # pruning used nodes for memory
    main(args)

