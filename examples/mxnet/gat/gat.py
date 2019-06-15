"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import mxnet as mx
from mxnet import gluon
import mxnet.ndarray as nd
import mxnet.gluon.nn as nn
import dgl.function as fn
from dgl.nn.mxnet import edge_softmax


class GraphAttention(gluon.Block):
    def __init__(self,
                 g,
                 in_dim,
                 out_dim,
                 num_heads,
                 feat_drop,
                 attn_drop,
                 alpha,
                 residual=False):
        super(GraphAttention, self).__init__()
        self.g = g
        self.num_heads = num_heads
        self.fc = nn.Dense(num_heads * out_dim, use_bias=False,
                           weight_initializer=mx.init.Xavier())
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x : x
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x : x
        self.attn_l = self.params.get("left_att", grad_req="add",
                                      shape=(1, num_heads, out_dim),
                                      init=mx.init.Xavier())
        self.attn_r = self.params.get("right_att", grad_req="add",
                                      shape=(1, num_heads, out_dim),
                                      init=mx.init.Xavier())
        self.alpha = alpha
        self.softmax = edge_softmax
        self.residual = residual
        if residual:
            if in_dim != out_dim:
                self.res_fc = nn.Dense(num_heads * out_dim, use_bias=False,
                                       weight_initializer=mx.init.Xavier())
            else:
                self.res_fc = None

    def forward(self, inputs):
        # prepare
        h = self.feat_drop(inputs)  # NxD
        ft = self.fc(h).reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
        a1 = (ft * self.attn_l.data(ft.context)).sum(axis=-1).expand_dims(-1)  # N x H x 1
        a2 = (ft * self.attn_r.data(ft.context)).sum(axis=-1).expand_dims(-1)  # N x H x 1
        self.g.ndata.update({'ft' : ft, 'a1' : a1, 'a2' : a2})
        # 1. compute edge attention
        self.g.apply_edges(self.edge_attention)
        # 2. compute softmax
        self.edge_softmax()
        # 3. compute the aggregated node features
        self.g.update_all(fn.src_mul_edge('ft', 'a_drop', 'ft'),
                          fn.sum('ft', 'ft'))
        ret = self.g.ndata['ft']
        # 4. residual
        if self.residual:
            if self.res_fc is not None:
                resval = self.res_fc(h).reshape(
                    (h.shape[0], self.num_heads, -1))  # NxHxD'
            else:
                resval = nd.expand_dims(h, axis=1)  # Nx1xD'
            ret = resval + ret
        return ret

    def edge_attention(self, edges):
        # an edge UDF to compute unnormalized attention values from src and dst
        a = nd.LeakyReLU(edges.src['a1'] + edges.dst['a2'], slope=self.alpha)
        return {'a' : a}

    def edge_softmax(self):
        attention = self.softmax(self.g, self.g.edata.pop('a'))
        # Dropout attention scores and save them
        self.g.edata['a_drop'] = self.attn_drop(attention)


class GAT(nn.Block):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 alpha,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = []
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GraphAttention(
            g, in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, alpha, False))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GraphAttention(
                g, num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, alpha, residual))
        # output projection
        self.gat_layers.append(GraphAttention(
            g, num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, alpha, residual))
        for i, layer in enumerate(self.gat_layers):
            self.register_child(layer, "gat_layer_{}".format(i))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](h).flatten()
            h = self.activation(h)
        # output projection
        logits = self.gat_layers[-1](h).mean(1)
        return logits
