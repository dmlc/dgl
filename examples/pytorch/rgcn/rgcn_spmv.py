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

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None, activation=None):
        super(RGCNLayer, self).__init__()
        self.num_bases = num_bases
        self.num_rels = num_rels
        self.bias = bias
        self.activation = activation
        if self.num_bases < 0:
            self.num_bases = self.num_rels
        self.weights = nn.ParameterList([nn.Parameter(torch.Tensor(in_feat, out_feat)) for _ in range(self.num_bases)])
        if self.num_bases < self.num_rels:
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
        if self.bias == True:
            self.bias = nn.Parameter(torch.Tensor(out_feat))

    def forward(self, parent, children):
        if self.num_bases < self.num_rels:
            weights = torch.stack(self.weights).permute(1, 0, 2)
            weights = torch.matmul(self.w_comp, weights).permute(1, 0, 2)
            weights = torch.split(weights, 1, dim=0)
        else:
            weights = self.weights

        for idx, g in enumerate(children):
            # update subgraph node repr
            g.copy_from(parent)
            # propagate subgraphs
            g.update_all('src_mul_edge', 'sum',
                         lambda node, accum: torch.mm(accum, weights[idx]))

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
    def __init__(self, g, in_dim, h_dim, out_dim, relations, num_layers=1, num_bases=-1, dropout=0, use_cuda=False):
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
            sub_eid = sub_rel > 0
            u = src[sub_eid]
            v = dst[sub_eid]
            sub_rel = sub_rel[sub_eid]
            subgrh = self.g.edge_subgraph(u, v)
            edge_repr = torch.from_numpy(sub_rel)
            if use_cuda:
                edge_repr = edge_repr.cuda()
            subgrh.set_e_repr(edge_repr)
            self.subgraphs.append(subgrh)

        # create rgcn layers
        # FIXME: more than 2 laeyrs?
        self.i2h = RGCNLayer(in_dim, h_dim, num_rels, num_bases, F.relu)
        self.h2o = RGCNLayer(h_dim, out_dim, num_rels, num_bases, F.softmax)

        # dropout layer
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, features):
        self.g.set_n_repr(features)
        self.i2h(self.g, self.subgraphs)
        if self.dropout:
            features = self.dropout(self.g.get_n_repr())
            self.g.set_n_repr(features)
        self.h2o(self.g, self.subgraphs)
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
                 args.n_bases,
                 dropout=args.dropout,
                 use_cuda=use_cuda)

    # convert to pytorch label format
    labels = np.argmax(labels, axis=0)
    labels = torch.from_numpy(labels).view(-1)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
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
        if epoch >= 3:
            t0 = time.time()
        logits = model.forward(features)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        if epoch >= 3:
            t1 = time.time()
        loss.backward()
        if epoch >= 3:
            t2 = time.time()
            forward_time.append(t1 - t0)
            backward_time.append(t2 - t1)
        optimizer.step()

        print("Epoch {:05d} | Loss {:.4f} | Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".format(epoch, loss.item(), np.mean(forward_time), np.mean(backward_time)))


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
    parser.add_argument("--n-epochs", type=int, default=50,
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

