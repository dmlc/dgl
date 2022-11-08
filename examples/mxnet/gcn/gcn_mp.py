"""GCN using basic message passing

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import mxnet as mx
from mxnet import gluon


def gcn_msg(edge):
    msg = edge.src["h"] * edge.src["norm"]
    return {"m": msg}


def gcn_reduce(node):
    accum = mx.nd.sum(node.mailbox["m"], 1) * node.data["norm"]
    return {"h": accum}


class NodeUpdate(gluon.Block):
    def __init__(self, out_feats, activation=None, bias=True):
        super(NodeUpdate, self).__init__()
        with self.name_scope():
            if bias:
                self.bias = self.params.get(
                    "bias", shape=(out_feats,), init=mx.init.Zero()
                )
            else:
                self.bias = None
        self.activation = activation

    def forward(self, node):
        h = node.data["h"]
        if self.bias is not None:
            h = h + self.bias.data(h.context)
        if self.activation:
            h = self.activation(h)
        return {"h": h}


class GCNLayer(gluon.Block):
    def __init__(self, g, in_feats, out_feats, activation, dropout, bias=True):
        super(GCNLayer, self).__init__()
        self.g = g
        self.dropout = dropout
        with self.name_scope():
            self.weight = self.params.get(
                "weight", shape=(in_feats, out_feats), init=mx.init.Xavier()
            )
            self.node_update = NodeUpdate(out_feats, activation, bias)

    def forward(self, h):
        if self.dropout:
            h = mx.nd.Dropout(h, p=self.dropout)
        h = mx.nd.dot(h, self.weight.data(h.context))
        self.g.ndata["h"] = h
        self.g.update_all(gcn_msg, gcn_reduce, self.node_update)
        h = self.g.ndata.pop("h")
        return h


class GCN(gluon.Block):
    def __init__(
        self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super(GCN, self).__init__()
        self.layers = gluon.nn.Sequential()
        # input layer
        self.layers.add(GCNLayer(g, in_feats, n_hidden, activation, 0))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.add(
                GCNLayer(g, n_hidden, n_hidden, activation, dropout)
            )
        # output layer
        self.layers.add(GCNLayer(g, n_hidden, n_classes, None, dropout))

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(h)
        return h
