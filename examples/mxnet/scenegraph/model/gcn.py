import dgl
import gluoncv as gcv
import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet.gluon import nn
from dgl.utils import toindex
from dgl.nn.mxnet import GraphConv

__all__ = ['EdgeGCN']

class EdgeLinkMLP(nn.Block):
    def __init__(self, n_hidden, n_classes):
        super(EdgeLinkMLP, self).__init__()
        self.mlp1 = nn.Dense(n_hidden)
        self.relu1 = nn.Activation('relu')
        self.mlp2 = nn.Dense(n_hidden)
        self.relu2 = nn.Activation('relu')
        self.mlp3 = nn.Dense(n_classes)

    def forward(self, edges):
        feat = nd.concat(edges.src['node_class_prob'], edges.src['pred_bbox'],
                         edges.dst['node_class_prob'], edges.dst['pred_bbox'])
        out = self.relu1(self.mlp1(feat))
        out = self.relu2(self.mlp2(out))
        out = self.mlp3(out)
        return {'link_preds': out}

class EdgeMLP(nn.Block):
    def __init__(self, n_hidden, n_classes):
        super(EdgeMLP, self).__init__()
        self.mlp1 = nn.Dense(n_hidden)
        self.relu1 = nn.Activation('relu')
        self.mlp2 = nn.Dense(n_hidden)
        self.relu2 = nn.Activation('relu')
        self.mlp3 = nn.Dense(n_classes)

    def forward(self, edges):
        feat = nd.concat(edges.src['node_class_prob'], edges.src['emb'], edges.src['pred_bbox'],
                         edges.dst['node_class_prob'], edges.dst['emb'], edges.dst['pred_bbox'])
        out = self.relu1(self.mlp1(feat))
        out = self.relu2(self.mlp2(out))
        out = self.mlp3(out)
        return {'preds': out}

class EdgeGCN(nn.Block):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 pretrained_base,
                 ctx):
        super(EdgeGCN, self).__init__()
        self.layers = nn.Sequential()
        # input layer
        self.layers.add(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.add(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.edge_link_mlp = EdgeLinkMLP(50, 2)
        self.edge_mlp = EdgeMLP(100, n_classes)

    def forward(self, g):
        if g is None or g.number_of_nodes() == 0:
            return g
        cls = g.ndata['node_class_pred']
        g.ndata['node_class_prob'] = nd.softmax(cls)
        # link pred
        g.apply_edges(self.edge_link_mlp)
        # subgraph for gconv
        if mx.autograd.is_training():
            eids = np.where(g.edata['link'].asnumpy() > 0)
            sub_g = g.edge_subgraph(toindex(eids[0].tolist()))
            sub_g.copy_from_parent()
            # graph conv
            x = sub_g.ndata['node_feat']
            for i, layer in enumerate(self.layers):
                x = layer(sub_g, x)
            sub_g.ndata['emb'] = x
            sub_g.copy_to_parent()
            # link classification
            g.apply_edges(self.edge_mlp)
        else:
            # graph conv
            x = g.ndata['node_feat']
            for i, layer in enumerate(self.layers):
                x = layer(g, x)
            g.ndata['emb'] = x
            # link classification
            g.apply_edges(self.edge_mlp)
        return g
