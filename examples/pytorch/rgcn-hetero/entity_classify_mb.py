"""Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Reference Code: https://github.com/tkipf/relational-gcn
"""
import argparse
import itertools
import numpy as np
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial

import dgl
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
    use_weight : bool, optional
        If True, multiply the input node feature with a learnable weight matrix
        before message passing.
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
                 use_weight=True,
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

        self.use_weight = use_weight
        if use_weight:
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
        """Forward computation

        Parameters
        ----------
        g : DGLHeteroGraph
            Input block graph.
        xs : dict[str, torch.Tensor]
            Node feature for each node type.

        Returns
        -------
        list of torch.Tensor
            New node features for each node type.
        """
        g = g.local_var()
        for ntype, x in xs.items():
            g.srcnodes[ntype].data['x'] = x
        if self.use_weight:
            ws = self.basis_weight()
            funcs = {}
            for i, (srctype, etype, dsttype) in enumerate(g.canonical_etypes):
                if srctype not in xs:
                    continue
                g.srcnodes[srctype].data['h%d' % i] = th.matmul(
                    g.srcnodes[srctype].data['x'], ws[etype])
                funcs[(srctype, etype, dsttype)] = (fn.copy_u('h%d' % i, 'm'), fn.mean('m', 'h'))
        else:
            funcs = {}
            for i, (srctype, etype, dsttype) in enumerate(g.canonical_etypes):
                if srctype not in xs:
                    continue
                g.srcnodes[srctype].data['h%d' % i] = g.srcnodes[srctype].data['x']
                funcs[(srctype, etype, dsttype)] = (fn.copy_u('h%d' % i, 'm'), fn.mean('m', 'h'))
        # message passing
        g.multi_update_all(funcs, 'sum')

        hs = {}
        for ntype in g.dsttypes:
            print(ntype, 'h' in g.dstnodes[ntype].data)
            if 'h' in g.dstnodes[ntype].data:
                hs[ntype] = g.dstnodes[ntype].data['h']
        def _apply(h):
            # apply bias and activation
            if self.self_loop:
                h = h + th.matmul(xs[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            h = self.dropout(h)
            return h
        hs = {ntype : _apply(h) for ntype, h in hs.items()}
        return hs

class RelGraphEmbed(nn.Module):
    r"""Embedding layer for featureless heterograph."""
    def __init__(self,
                 g,
                 embed_size,
                 activation=None,
                 dropout=0.0):
        super(RelGraphEmbed, self).__init__()
        self.g = g
        self.embed_size = embed_size
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        for ntype in g.ntypes:
            embed = nn.Parameter(th.Tensor(g.number_of_nodes(ntype), self.embed_size))
            nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
            self.embeds[ntype] = embed


    def forward(self, block=None):
        """Forward computation

        Parameters
        ----------
        block : DGLHeteroGraph, optional
            If not specified, directly return the full graph with embeddings stored in
            :attr:`embed_name`. Otherwise, extract and store the embeddings to the block
            graph and return.

        Returns
        -------
        DGLHeteroGraph
            The block graph fed with embeddings.
        """
        return self.embeds

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
        self.rel_names.sort()
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop

        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConvHetero(
            self.h_dim, self.h_dim, self.rel_names, "basis",
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            dropout=self.dropout, use_weight=False))
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(RelGraphConvHetero(
                self.h_dim, self.h_dim, self.rel_names, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout))
        # h2o
        self.layers.append(RelGraphConvHetero(
            self.h_dim, self.out_dim, self.rel_names, "basis",
            self.num_bases, activation=None,
            self_loop=self.use_self_loop))

    def forward(self, h, blocks):
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        return h

class HeteroNeighborSampler:
    def __init__(self, g, category, fanouts, full_neighbor=False):
        self.g = g
        self.category = category
        self.fanouts = fanouts
        self.full_neighbor = full_neighbor

    def sample_blocks(self, seeds):
        blocks = []
        seeds = {self.category : th.tensor(seeds)}
        cur = seeds
        for fanout in self.fanouts:
            if self.full_neighbor:
                frontier = dgl.in_subgraph(self.g, cur)
            else:
                frontier = dgl.sampling.sample_neighbors(self.g, cur, fanout)
            block = dgl.to_block(frontier, cur)
            cur = {}
            for ntype in block.srctypes:
                cur[ntype] = block.srcnodes[ntype].data[dgl.NID]
            blocks.insert(0, block)
        return seeds, blocks

def extract_embed(node_embed, block):
    emb = {}
    for ntype in block.srctypes:
        nid = block.nodes[ntype].data[dgl.NID]
        emb[ntype] = node_embed[ntype][nid]
    return emb


def evaluate(model, seeds, blocks, node_embed, labels, category, use_cuda):
    model.eval()
    emb = extract_embed(node_embed, blocks[0])
    lbl = labels[seeds]
    if use_cuda:
        emb = {k : e.cuda() for k, e in emb.items()}
        lbl = lbl.cuda()
    logits = model(emb, blocks)[category]
    loss = F.cross_entropy(logits, lbl)
    acc = th.sum(logits.argmax(dim=1) == lbl).item() / len(seeds)
    return loss, acc

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

    train_label = labels[train_idx]
    val_label = labels[val_idx]
    test_label = labels[test_idx]

    # create embeddings
    embed_layer = RelGraphEmbed(g, args.n_hidden)
    node_embed = embed_layer()
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

    # train sampler
    sampler = HeteroNeighborSampler(g, category, [args.fanout] * args.n_layers)
    loader = DataLoader(dataset=list(train_idx.numpy()),
                        batch_size=args.batch_size,
                        collate_fn=sampler.sample_blocks,
                        shuffle=True,
                        num_workers=0)

    # validation sampler
    val_sampler = HeteroNeighborSampler(g, category, [args.fanout] * args.n_layers, True)
    _, val_blocks = val_sampler.sample_blocks(val_idx)

    # test sampler
    test_sampler = HeteroNeighborSampler(g, category, [args.fanout] * args.n_layers, True)
    _, test_blocks = test_sampler.sample_blocks(test_idx)

    # optimizer
    all_params = itertools.chain(model.parameters(), embed_layer.parameters())
    optimizer = th.optim.Adam(all_params, lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        optimizer.zero_grad()
        if epoch > 3:
            t0 = time.time()

        for i, (seeds, blocks) in enumerate(loader):
            batch_tic = time.time()
            emb = extract_embed(node_embed, blocks[0])
            lbl = labels[seeds[category]]
            if use_cuda:
                emb = {k : e.cuda() for k, e in emb.items()}
                lbl = lbl.cuda()
            logits = model(emb, blocks)[category]
            loss = F.cross_entropy(logits, lbl)
            loss.backward()
            optimizer.step()

            train_acc = th.sum(logits.argmax(dim=1) == lbl).item() / len(seeds[category])
            print("Epoch {:05d} | Batch {:03d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Time: {:.4f}".
                  format(epoch, i, train_acc, loss.item(), time.time() - batch_tic))

        if epoch > 3:
            dur.append(time.time() - t0)

        val_loss, val_acc = evaluate(model, val_idx, val_blocks, node_embed, labels, category, use_cuda)
        print("Epoch {:05d} | Valid Acc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".
              format(epoch, val_acc, val_loss.item(), np.average(dur)))
    print()
    if args.model_path is not None:
        th.save(model.state_dict(), args.model_path)

    test_loss, test_acc = evaluate(model, test_idx, test_blocks, node_embed, labels, category, use_cuda)
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
    parser.add_argument("-e", "--n-epochs", type=int, default=20,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--model_path", type=str, default=None,
            help='path for save the model')
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    parser.add_argument("--batch-size", type=int, default=100,
            help="Mini-batch size. If -1, use full graph training.")
    parser.add_argument("--fanout", type=int, default=4,
            help="Fan-out of neighbor sampling.")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    main(args)
