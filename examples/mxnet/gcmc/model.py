import mxnet.ndarray as F
import numpy as np
import warnings
from mxnet.gluon import nn, HybridBlock, Block
from utils import get_activation
import mxnet as mx
import dgl.function as fn

class GCMCLayer(Block):
    def __init__(self,
                 rating_vals,
                 user_in_units,
                 movie_in_units,
                 msg_units,
                 out_units,
                 dropout_rate=0.0,
                 agg='stack',  # or 'sum'
                 agg_act=None,
                 out_act=None):
        super(GCMCLayer, self).__init__()
        self.rating_vals = rating_vals
        self.agg = agg
        with self.name_scope():
            self.dropout = nn.Dropout(dropout_rate)
            self.W_r = {}
            for rating in rating_vals:
                rating = str(rating)
                self.W_r[rating] = self.params.get(
                    'W_r_%s' % rating, shape=(user_in_units, msg_units),
                    dtype=np.float32, allow_deferred_init=True)
                self.W_r['rev-%s' % rating] = self.params.get(
                    'revW_r_%s' % rating, shape=(movie_in_units, msg_units),
                    dtype=np.float32, allow_deferred_init=True)
            self.ufc = nn.Dense(out_units)
            self.ifc = nn.Dense(out_units)
            self.agg_act = get_activation(agg_act)
            self.out_act = get_activation(out_act)

    def forward(self, graph, ufeat, ifeat):
        funcs = {}
        # left norm
        ufeat = ufeat * graph.nodes['user'].data['cj']
        ifeat = ifeat * graph.nodes['movie'].data['cj']
        for i, rating in enumerate(self.rating_vals):
            rating = str(rating)
            # W_r * x
            graph.nodes['user'].data['h%d' % i] = mx.nd.dot(
                self.dropout(ufeat), self.W_r[rating].data())
            graph.nodes['movie'].data['h%d' % i] = mx.nd.dot(
                self.dropout(ifeat), self.W_r['rev-%s' % rating].data())
            funcs[rating] = (fn.copy_u('h%d' % i, 'm'), fn.sum('m', 'h'))
            funcs['rev-%s' % rating] = (fn.copy_u('h%d' % i, 'm'), fn.sum('m', 'h'))
        # message passing
        graph.multi_update_all(funcs, self.agg)
        ufeat = graph.nodes['user'].data.pop('h')
        ifeat = graph.nodes['movie'].data.pop('h')
        # right norm
        ufeat = ufeat * graph.nodes['user'].data['ci']
        ifeat = ifeat * graph.nodes['movie'].data['ci']
        # fc and non-linear
        ufeat = self.agg_act(ufeat)
        ifeat = self.agg_act(ifeat)
        ufeat = self.ufc(ufeat)
        ifeat = self.ifc(ifeat)
        return self.out_act(ufeat), self.out_act(ifeat)

class BiDecoder(HybridBlock):
    def __init__(self, in_units, out_units, num_basis_functions=2, prefix=None, params=None):
        super(BiDecoder, self).__init__(prefix=prefix, params=params)
        self._num_basis_functions = num_basis_functions
        with self.name_scope():
            for i in range(num_basis_functions):
                self.__setattr__('weight{}'.format(i),
                                 self.params.get('weight{}'.format(i), shape=(in_units, in_units),
                                                 init=mx.initializer.Orthogonal(scale=1.1,
                                                                                rand_type='normal'),
                                                 allow_deferred_init=True))
            self.rate_out = nn.Dense(units=out_units, flatten=False, use_bias=False, prefix="rate_")

    def hybrid_forward(self, F, data1, data2, **kwargs):
        basis_outputs_l = []
        for i in range(self._num_basis_functions):
            basis_out = F.sum(F.dot(data1, kwargs["weight{}".format(i)]) * data2,
                              axis=1, keepdims=True)
            basis_outputs_l.append(basis_out)
        basis_outputs = F.concat(*basis_outputs_l, dim=1)
        out = self.rate_out(basis_outputs)
        return out


class InnerProductLayer(HybridBlock):
    def __init__(self, mid_units=None, **kwargs):
        super(InnerProductLayer, self).__init__(**kwargs)
        self._mid_units = mid_units
        if self._mid_units is not None:
            self._mid_map = nn.Dense(mid_units, flatten=False)

    def hybrid_forward(self, F, data1, data2):
        if self._mid_units is not None:
            data1 = self._mid_map(data1)
            data2 = self._mid_map(data2)
        score = F.sum(data1 * data2, axis=1, keepdims=True)
        return score
