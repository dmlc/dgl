"""
Graph Attention Networks
Paper: https://arxiv.org/abs/1710.10903
Code: https://github.com/PetarV-/GAT

GAT with batch processing
"""

import argparse
import numpy as np
import time
import mxnet as mx
from mxnet import gluon
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

def elu(data):
    return mx.nd.LeakyReLU(data, act_type='elu')

def gat_message(src, edge):
    return {'ft' : src['ft'], 'a2' : src['a2']}

class GATReduce(gluon.Block):
    def __init__(self, attn_drop):
        super(GATReduce, self).__init__()
        self.attn_drop = attn_drop

    def forward(self, node, msgs):
        a1 = mx.nd.expand_dims(node['a1'], 1)  # shape (B, 1, 1)
        a2 = msgs['a2'] # shape (B, deg, 1)
        ft = msgs['ft'] # shape (B, deg, D)
        # attention
        a = a1 + a2  # shape (B, deg, 1)
        e = mx.nd.softmax(mx.nd.LeakyReLU(a))
        if self.attn_drop != 0.0:
            e = mx.nd.Dropout(e, self.attn_drop)
        return {'accum' : mx.nd.sum(e * ft, axis=1)} # shape (B, D)

class GATFinalize(gluon.Block):
    def __init__(self, headid, indim, hiddendim, activation, residual):
        super(GATFinalize, self).__init__()
        self.headid = headid
        self.activation = activation
        self.residual = residual
        self.residual_fc = None
        if residual:
            if indim != hiddendim:
                self.residual_fc = gluon.nn.Dense(hiddendim)

    def forward(self, node):
        ret = node['accum']
        if self.residual:
            if self.residual_fc is not None:
                ret = self.residual_fc(node['h']) + ret
            else:
                ret = node['h'] + ret
        return {'head%d' % self.headid : self.activation(ret)}

class GATPrepare(gluon.Block):
    def __init__(self, indim, hiddendim, drop):
        super(GATPrepare, self).__init__()
        self.fc = gluon.nn.Dense(hiddendim)
        self.drop = drop
        self.attn_l = gluon.nn.Dense(1)
        self.attn_r = gluon.nn.Dense(1)

    def forward(self, feats):
        h = feats
        if self.drop != 0.0:
            h = mx.nd.Dropout(h, self.drop)
        ft = self.fc(h)
        a1 = self.attn_l(ft)
        a2 = self.attn_r(ft)
        return {'h' : h, 'ft' : ft, 'a1' : a1, 'a2' : a2}

class GAT(gluon.Block):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_heads,
                 activation,
                 in_drop,
                 attn_drop,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prp = gluon.nn.Sequential()
        self.red = gluon.nn.Sequential()
        self.fnl = gluon.nn.Sequential()
        # input projection (no residual)
        for hid in range(num_heads):
            self.prp.add(GATPrepare(in_dim, num_hidden, in_drop))
            self.red.add(GATReduce(attn_drop))
            self.fnl.add(GATFinalize(hid, in_dim, num_hidden, activation, False))
        # hidden layers
        for l in range(num_layers - 1):
            for hid in range(num_heads):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.prp.add(GATPrepare(num_hidden * num_heads, num_hidden, in_drop))
                self.red.add(GATReduce(attn_drop))
                self.fnl.add(GATFinalize(hid, num_hidden * num_heads,
                                         num_hidden, activation, residual))
        # output projection
        self.prp.add(GATPrepare(num_hidden * num_heads, num_classes, in_drop))
        self.red.add(GATReduce(attn_drop))
        self.fnl.add(GATFinalize(0, num_hidden * num_heads,
                                 num_classes, activation, residual))
        # sanity check
        assert len(self.prp) == self.num_layers * self.num_heads + 1
        assert len(self.red) == self.num_layers * self.num_heads + 1
        assert len(self.fnl) == self.num_layers * self.num_heads + 1

    def forward(self, features):
        last = features
        for l in range(self.num_layers):
            for hid in range(self.num_heads):
                i = l * self.num_heads + hid
                # prepare
                self.g.set_n_repr(self.prp[i](last))
                # message passing
                self.g.update_all(gat_message, self.red[i], self.fnl[i], batchable=True)
            # merge all the heads
            last = mx.nd.concat(
                    *[self.g.pop_n_repr('head%d' % hid) for hid in range(self.num_heads)],
                    dim=1)
        # output projection
        self.g.set_n_repr(self.prp[-1](last))
        self.g.update_all(gat_message, self.red[-1], self.fnl[-1], batchable=True)
        return self.g.pop_n_repr('head0')

def main(args):
    # load and preprocess dataset
    data = load_data(args)

    features = mx.nd.array(data.features)
    labels = mx.nd.array(data.labels)
    mask = mx.nd.array(data.train_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        mask = mask.cuda()

    # create GCN model
    g = DGLGraph(data.graph)

    # create model
    model = GAT(g,
                args.num_layers,
                in_feats,
                args.num_hidden,
                n_classes,
                args.num_heads,
                elu,
                args.in_drop,
                args.attn_drop,
                args.residual)

    if cuda:
        model.cuda()
    model.initialize()

    # use optimizer
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': args.lr})

    # initialize graph
    dur = []
    for epoch in range(args.epochs):
        if epoch >= 3:
            t0 = time.time()
        # forward
        with mx.autograd.record():
            logits = model(features)
            loss = mx.nd.softmax_cross_entropy(logits, labels)

        #optimizer.zero_grad()
        loss.backward()
        trainer.step(features.shape[0])

        if epoch >= 3:
            dur.append(time.time() - t0)
            print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | ETputs(KTEPS) {:.2f}".format(
                epoch, loss.asnumpy()[0], np.mean(dur), n_edges / np.mean(dur) / 1000))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
            help="Which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=20,
            help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=3,
            help="number of attentional heads to use")
    parser.add_argument("--num-layers", type=int, default=1,
            help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
            help="size of hidden units")
    parser.add_argument("--residual", action="store_false",
            help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
            help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
            help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
            help="learning rate")
    args = parser.parse_args()
    print(args)

    main(args)
