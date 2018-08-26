"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn

Difference compared to tkipf/relation-gcn
* edge directions are reversed (kipf did not transpose adj before spmv)
* l2norm applied to all weights
* remove both in and out edges for nodes that won't be touched
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

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_gs, num_rels, num_bases=-1, bias=None, activation=None):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_bases = min(num_bases, num_gs)
        self.num_rels = min(num_rels, num_gs)
        self.bias = bias
        self.activation = activation
        if self.num_bases < 0:
            self.num_bases = self.num_rels
        self.weights = nn.ParameterList([nn.Parameter(torch.Tensor(in_feat, out_feat)) for _ in range(self.num_bases)])
        for w in self.weights:
            nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
            nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))
        if self.bias == True:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))

    def forward(self, parent, children):
        if self.num_bases < self.num_rels:
            weights = torch.stack(list(self.weights)).permute(1, 0, 2)
            weights = torch.matmul(self.w_comp, weights).view(-1, self.out_feat)
            weights = torch.split(weights, self.in_feat, dim=0)
        else:
            weights = self.weights

        for idx, g in enumerate(children):
            # update subgraph node repr
            g.copy_from(parent)
            # propagate subgraphs
            g.update_all('src_mul_edge', 'sum',
                         lambda node, accum: torch.mm(accum, weights[idx]),
                         batchable=True)

        # merge node repr
        parent.merge(children, node_reduce_func='sum', edge_reduce_func=None)

        # apply bias and activation
        node_repr = parent.get_n_repr()
        if self.bias:
            node_repr = node_repr + self.bias
        if self.activation:
            node_repr = self.activation(node_repr)

        parent.set_n_repr(node_repr)


class RGCN(nn.Module):
    def __init__(self, g, in_dim, h_dim, out_dim, relations, num_bases=-1, num_layers=1, dropout=0, use_cuda=False):
        super(RGCN, self).__init__()
        self.g = g
        self.dropout = dropout
        num_rels = relations.shape[1]
        assert num_bases != 0 and num_bases <= num_rels

        # generate subgraphs
        self.subgraphs = []
        src, dst = np.transpose(np.array(self.g.edge_list))
        for rel in range(num_rels):
            sub_rel = relations[:, rel]
            if np.count_nonzero(sub_rel) == 0:
                # skip relations with no edges
                continue
            sub_eid = sub_rel > 0
            u = src[sub_eid]
            v = dst[sub_eid]
            sub_rel = sub_rel[sub_eid]
            subgrh = self.g.edge_subgraph(u, v)
            edge_repr = torch.from_numpy(sub_rel).view(-1, 1)
            if use_cuda:
                edge_repr = edge_repr.cuda()
            subgrh.set_e_repr(edge_repr)
            self.subgraphs.append(subgrh)

        # create rgcn layers
        num_subgraphs = len(self.subgraphs)
        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RGCNLayer(in_dim, h_dim, num_subgraphs, num_rels, num_bases, activation=F.relu))
        # h2h
        for _ in range(1, num_layers):
            self.layers.append(RGCNLayer(h_dim, h_dim, num_subgraphs, num_rels, num_bases, activation=F.relu))
        # h2o
        self.layers.append(RGCNLayer(h_dim, out_dim, num_subgraphs, num_rels, num_bases, activation=partial(F.softmax, dim=1)))

        # dropout layer
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, features):
        self.g.set_n_repr(features)
        self.layers[0](self.g, self.subgraphs)
        for i in range(1, len(self.layers)):
            if self.dropout:
                features = self.dropout(self.g.get_n_repr())
                self.g.set_n_repr(features)
            self.layers[i](self.g, self.subgraphs)
        return self.g.pop_n_repr()


def main(args):
    # load graph data
    data = load_data(args)
    num_nodes = data.num_nodes
    edges = data.edges
    relations = data.relations
    labels = data.labels
    train_idx = data.train_idx
    test_idx = data.test_idx

    # default features are identity matrix, switch to other features here
    features = torch.eye(num_nodes)

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
    model = RGCN(g,
                 features.shape[1],
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

    if use_cuda:
        model.cuda()
        features = features.cuda()
        labels = labels.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    forward_time = []
    backward_time = []
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        t0 = time.time()
        logits = model.forward(features)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        t1 = time.time()
        loss.backward()
        t2 = time.time()
        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        optimizer.step()

        print("Epoch {:05d} | Loss {:.4f} | Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".format(epoch, loss.item(), forward_time[-1], backward_time[-1]))

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
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=20,
            help="number of training epochs")
    parser.add_argument("--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    args.bfs_level = args.n_layers + 1 # pruning used nodes for memory
    main(args)

