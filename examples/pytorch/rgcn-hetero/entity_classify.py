"""Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Reference Code: https://github.com/tkipf/relational-gcn
"""
import argparse
import numpy as np
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import dgl.function as fn
from dgl.data.rdf import AIFB, MUTAG, BGS, AM

class RelGraphConvHetero(nn.Module):
    r"""Relational graph convolution layer.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : int
        Relation names.
    regularizer : str
        Which weight regularizer to use "basis" or "bdd"
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 regularizer="basis",
                 num_bases=None,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvHetero, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_rels = len(rel_names)
        self.regularizer = regularizer
        self.num_bases = num_bases
        if self.num_bases is None or self.num_bases > self.num_rels or self.num_bases < 0:
            self.num_bases = self.num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        if regularizer == "basis":
            # add basis weights
            self.weight = nn.Parameter(th.Tensor(self.num_bases, self.in_feat, self.out_feat))
            if self.num_bases < self.num_rels:
                # linear combination coefficients
                self.w_comp = nn.Parameter(th.Tensor(self.num_rels, self.num_bases))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            if self.num_bases < self.num_rels:
                nn.init.xavier_uniform_(self.w_comp,
                                        gain=nn.init.calculate_gain('relu'))
        else:
            raise ValueError("Only basis regularizer is supported.")

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def basis_weight(self):
        """Message function for basis regularizer"""
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.num_bases,
                                      self.in_feat * self.out_feat)
            weight = th.matmul(self.w_comp, weight).view(
                self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight
        return {self.rel_names[i] : w.squeeze(0) for i, w in enumerate(th.split(weight, 1, dim=0))}

    def forward(self, g, xs):
        """ Forward computation

        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        xs : list of torch.Tensor
            Node feature for each node type.

        Returns
        -------
        list of torch.Tensor
            New node features for each node type.
        """
        g = g.local_var()
        for i, ntype in enumerate(g.ntypes):
            g.nodes[ntype].data['x'] = xs[i]
        ws = self.basis_weight()
        funcs = {}
        for i, (srctype, etype, dsttype) in enumerate(g.canonical_etypes):
            g.nodes[srctype].data['h%d' % i] = th.matmul(
                g.nodes[srctype].data['x'], ws[etype])
            funcs[(srctype, etype, dsttype)] = (fn.copy_u('h%d' % i, 'm'), fn.mean('m', 'h'))
        # message passing
        g.multi_update_all(funcs, 'sum')

        hs = [g.nodes[ntype].data['h'] for ntype in g.ntypes]
        for i in range(len(hs)):
            h = hs[i]
            # apply bias and activation
            if self.self_loop:
                h = h + th.matmul(xs[i], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            h = self.dropout(h)
            hs[i] = h
        return hs

class RelGraphConvHeteroEmbed(nn.Module):
    r"""Embedding layer for featureless heterograph."""
    def __init__(self,
                 embed_size,
                 g,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvHeteroEmbed, self).__init__()
        self.embed_size = embed_size
        self.g = g
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterList()
        for srctype, etype, dsttype in g.canonical_etypes:
            embed = nn.Parameter(th.Tensor(g.number_of_nodes(srctype), self.embed_size))
            nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
            self.embeds.append(embed)

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(th.Tensor(embed_size))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.self_embeds = nn.ParameterList()
            for ntype in g.ntypes:
                embed = nn.Parameter(th.Tensor(g.number_of_nodes(ntype), embed_size))
                nn.init.xavier_uniform_(embed,
                                        gain=nn.init.calculate_gain('relu'))
                self.self_embeds.append(embed)

        self.dropout = nn.Dropout(dropout)

    def forward(self):
        """ Forward computation

        Returns
        -------
        torch.Tensor
            New node features.
        """
        g = self.g.local_var()
        funcs = {}
        for i, (srctype, etype, dsttype) in enumerate(g.canonical_etypes):
            g.nodes[srctype].data['embed-%d' % i] = self.embeds[i]
            funcs[(srctype, etype, dsttype)] = (fn.copy_u('embed-%d' % i, 'm'), fn.mean('m', 'h'))
        g.multi_update_all(funcs, 'sum')
        
        hs = [g.nodes[ntype].data['h'] for ntype in g.ntypes]
        for i in range(len(hs)):
            h = hs[i]
            # apply bias and activation
            if self.self_loop:
                h = h + self.self_embeds[i]
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            h = self.dropout(h)
            hs[i] = h
        return hs

class EntityClassify(nn.Module):
    def __init__(self,
                 g,
                 h_dim, out_dim,
                 num_bases,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False):
        super(EntityClassify, self).__init__()
        self.g = g
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.rel_names = list(set(g.etypes))
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop

        self.embed_layer = RelGraphConvHeteroEmbed(
            self.h_dim, g, activation=F.relu, self_loop=self.use_self_loop,
            dropout=self.dropout)
        self.layers = nn.ModuleList()
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(RelGraphConvHetero(
                self.h_dim, self.h_dim, self.rel_names, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout))
        # h2o
        self.layers.append(RelGraphConvHetero(
            self.h_dim, self.out_dim, self.rel_names, "basis",
            self.num_bases, activation=partial(F.softmax, dim=1),
            self_loop=self.use_self_loop))

    def forward(self):
        h = self.embed_layer()
        for layer in self.layers:
            h = layer(self.g, h)
        return h

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

    g = dataset.graph
    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_idx = dataset.train_idx
    test_idx = dataset.test_idx
    labels = dataset.labels
    category_id = len(g.ntypes)
    for i, ntype in enumerate(g.ntypes):
        if ntype == category:
            category_id = i

    # split dataset into train, validate, test
    if args.validation:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx

    # check cuda
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)
        labels = labels.cuda()
        train_idx = train_idx.cuda()
        test_idx = test_idx.cuda()
        labels = labels.cuda()

    # create model
    model = EntityClassify(g,
                           args.n_hidden,
                           num_classes,
                           num_bases=args.n_bases,
                           num_hidden_layers=args.n_layers - 2,
                           dropout=args.dropout,
                           use_self_loop=args.use_self_loop)

    if use_cuda:
        model.cuda()

    # optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    dur = []
    model.train()
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        if epoch > 5:
            t0 = time.time()
        logits = model()[category_id]
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()
        t1 = time.time()

        if epoch > 5:
            dur.append(t1 - t0)
        train_acc = th.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
        val_acc = th.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        print("Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Valid Acc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".
              format(epoch, train_acc, loss.item(), val_acc, val_loss.item(), np.average(dur)))
    print()

    model.eval()
    logits = model.forward()[category_id]
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    test_acc = th.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
    print("Test Acc: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
    print()

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
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    main(args)
