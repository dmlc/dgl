import math
import warnings

import numpy as np
import mxnet as mx
import mxnet.ndarray as F
from mxnet.gluon import nn, HybridBlock, Block
import dgl.function as fn

from utils import get_activation

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
                 out_act=None,
                 share_user_item_param=False):
        super(GCMCLayer, self).__init__()
        self.rating_vals = rating_vals
        self.agg = agg
        self.share_user_item_param = share_user_item_param
        if agg == 'stack':
            # divide the original msg unit size by number of ratings to keep
            # the dimensionality
            assert msg_units % len(rating_vals) == 0
            msg_units = msg_units // len(rating_vals)
        with self.name_scope():
            if share_user_item_param and user_in_units != movie_in_units:
                raise ValueError('Sharing user and movie parameters requires the feature '
                                 'dimensions of users and movies to be the same. Got %d and %d.'
                                 % (user_in_units, movie_in_units))
            self.dropout = nn.Dropout(dropout_rate)
            self.W_r = {}
            for rating in rating_vals:
                rating = str(rating)
                if share_user_item_param:
                    self.W_r[rating] = self.params.get(
                        'W_r_%s' % rating, shape=(user_in_units, msg_units),
                        dtype=np.float32, allow_deferred_init=True)
                    self.W_r['rev-%s' % rating] = self.W_r[rating]
                else:
                    self.W_r[rating] = self.params.get(
                        'W_r_%s' % rating, shape=(user_in_units, msg_units),
                        dtype=np.float32, allow_deferred_init=True)
                    self.W_r['rev-%s' % rating] = self.params.get(
                        'revW_r_%s' % rating, shape=(movie_in_units, msg_units),
                        dtype=np.float32, allow_deferred_init=True)
            self.ufc = nn.Dense(out_units)
            if share_user_item_param:
                self.ifc = self.ufc
            else:
                self.ifc = nn.Dense(out_units)
            self.agg_act = get_activation(agg_act)
            self.out_act = get_activation(out_act)

    def forward(self, graph, ufeat, ifeat):
        num_u = graph.number_of_nodes('user')
        num_i = graph.number_of_nodes('movie')
        funcs = {}
        for i, rating in enumerate(self.rating_vals):
            rating = str(rating)
            # W_r * x
            x_u = dot_or_identity(ufeat, self.W_r[rating].data())
            x_i = dot_or_identity(ifeat, self.W_r['rev-%s' % rating].data())
            # left norm
            x_u = x_u * graph.nodes['user'].data['cj']
            x_i = x_i * graph.nodes['movie'].data['cj']
            # dropout (TODO: row dropout)
            x_u = self.dropout(x_u)
            x_i = self.dropout(x_i)
            graph.nodes['user'].data['h%d' % i] = x_u
            graph.nodes['movie'].data['h%d' % i] = x_i
            funcs[rating] = (fn.copy_u('h%d' % i, 'm'), fn.sum('m', 'h'))
            funcs['rev-%s' % rating] = (fn.copy_u('h%d' % i, 'm'), fn.sum('m', 'h'))
        # message passing
        graph.multi_update_all(funcs, self.agg)
        ufeat = graph.nodes['user'].data.pop('h').reshape((num_u, -1))
        ifeat = graph.nodes['movie'].data.pop('h').reshape((num_i, -1))
        # right norm
        ufeat = ufeat * graph.nodes['user'].data['ci']
        ifeat = ifeat * graph.nodes['movie'].data['ci']
        # fc and non-linear
        ufeat = self.agg_act(ufeat)
        ifeat = self.agg_act(ifeat)
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        ufeat = self.ufc(ufeat)
        ifeat = self.ifc(ifeat)
        return self.out_act(ufeat), self.out_act(ifeat)

class BiDecoder(HybridBlock):
    def __init__(self,
                 rating_vals,
                 in_units,
                 num_basis_functions=2,
                 dropout_rate=0.0,
                 prefix=None, params=None):
        super(BiDecoder, self).__init__(prefix=prefix, params=params)
        self.rating_vals = rating_vals
        self._num_basis_functions = num_basis_functions
        self.dropout = nn.Dropout(dropout_rate)
        self.Ps = []
        with self.name_scope():
            for i in range(num_basis_functions):
                self.Ps.append(self.params.get(
                    'Ps_%d' % i, shape=(in_units, in_units),
                    #init=mx.initializer.Orthogonal(scale=1.1, rand_type='normal'),
                    init=mx.initializer.Xavier(magnitude=math.sqrt(2.0)),
                    allow_deferred_init=True))
            self.rate_out = nn.Dense(units=len(rating_vals), flatten=False, use_bias=False)

    def forward(self, graph, ufeat, ifeat):
        """Forward function.

        Parameters
        ----------
        graph : DGLHeteroGraph
            The user-movie rating graph.
        ufeat : mx.nd.NDArray
            User embeddings. Shape: (|V_u|, D)
        ifeat : mx.nd.NDArray
            Movie embeddings. Shape: (|V_m|, D)

        Returns
        -------
        mx.nd.NDArray
            Predicting scores for each user-movie edge.
        """
        graph = graph.local_var()
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        graph.nodes['movie'].data['h'] = ifeat
        basis_out = []
        for i in range(self._num_basis_functions):
            graph.nodes['user'].data['h'] = F.dot(ufeat, self.Ps[i].data())
            graph.apply_edges(fn.u_dot_v('h', 'h', 'sr'))
            basis_out.append(graph.edata['sr'].expand_dims(1))
        out = F.concat(*basis_out, dim=1)
        out = self.rate_out(out)
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

def dot_or_identity(A, B):
    # if A is None, treat as identity matrix
    if A is None:
        return B
    else:
        return mx.nd.dot(A, B)
