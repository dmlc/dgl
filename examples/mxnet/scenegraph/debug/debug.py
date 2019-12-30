import dgl
import gluoncv as gcv
import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet.gluon import nn
from dgl.utils import toindex
from dgl.nn.mxnet import GraphConv

class EdgeGCN(nn.Block):
    def __init__(self, in_feats, n_hidden, activation):
        super(EdgeGCN, self).__init__()
        self.layer = GraphConv(in_feats, n_hidden, activation=activation)

    def forward(self, g):
        # graph conv
        x = g.ndata['node_feat']
        x = self.layer(g, x)
        g.ndata['emb'] = x
        return g

ctx = mx.gpu(1)

net = EdgeGCN(10, 5, nd.relu)
net.initialize(ctx=ctx)
trainer = mx.gluon.Trainer(net.collect_params(), 'adam',
                           {'learning_rate': 0.01})

def build_complete_graph(n_nodes):
    g = dgl.DGLGraph()
    g.add_nodes(n_nodes)
    edge_list = []
    for i in range(n_nodes-1):
        for j in range(i+1, n_nodes):
            edge_list.append((i, j))
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    g.add_edges(dst, src)
    return g

g = build_complete_graph(5)
g.ndata['node_feat'] = nd.ones((5, 10), ctx=ctx)
g.ndata['node_label'] = nd.ones((5, 1), ctx=ctx)

L = mx.gluon.loss.SoftmaxCELoss()

with mx.autograd.record():
    g = net(g)
    loss = L(g.ndata['emb'], g.ndata['node_label'])
nd.waitall()
loss.backward()
trainer.step(1)

nd.waitall()
print(g.ndata['emb'])
