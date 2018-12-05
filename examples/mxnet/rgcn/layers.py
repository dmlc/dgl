import math

import mxnet as mx
from mxnet import gluon
import mxnet.ndarray as F
import dgl.function as fn

class RGCNLayer(gluon.Block):
    def __init__(self, in_feat, out_feat, bias=None, activation=None,
                 self_loop=False, dropout=0.0):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        if self.bias == True:
            self.bias = self.params.get('bias', shape=(out_feat,),
                                        init=mx.init.Xavier(magnitude=math.sqrt(2.0)))

        # weight for self loop
        if self.self_loop:
            self.loop_weight = self.params.get('loop_weight', shape=(in_feat, out_feat),
                                               init=mx.init.Xavier(magnitude=math.sqrt(2.0)))
        if dropout:
            self.dropout = gluon.nn.Dropout(dropout)
        else:
            self.dropout = None

    # define how propagation is done in subclass
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g):
        if self.self_loop:
            loop_message = F.dot(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        self.propagate(g)

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)

        g.ndata['h'] = node_repr


class RGCNBasisLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNBasisLayer, self).__init__(in_feat, out_feat, bias, activation)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # add basis weights
        if self.num_bases < self.num_rels:
            # linear combination coefficients
            self.weight = self.params.get('weight', shape=(self.num_bases, self.in_feat * self.out_feat))
            self.w_comp = self.params.get('w_comp', shape=(self.num_rels, self.num_bases),
                                          init=mx.init.Xavier(magnitude=math.sqrt(2.0)))
        else:
            self.weight = self.params.get('weight', shape=(self.num_bases, self.in_feat, self.out_feat),
                                          init=mx.init.Xavier(magnitude=math.sqrt(2.0)))

    def propagate(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = F.dot(self.w_comp.data(), self.weight.data()).reshape((self.num_rels, self.in_feat, self.out_feat))
        else:
            weight = self.weight.data()

        if self.is_input_layer:
            def msg_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = F.reshape(weight, (-1, self.out_feat))
                index = edges.data['type'] * self.in_feat + edges.src['id']
                return {'msg': embed[index] * edges.data['norm']}
        else:
            def msg_func(edges):
                w = weight[edges.data['type']]
                msg = F.batch_dot(edges.src['h'].expand_dims(1), w).reshape(-1, self.out_feat)
                msg = msg * edges.data['norm']
                return {'msg': msg}

        g.update_all(msg_func, fn.sum(msg='msg', out='h'), None)