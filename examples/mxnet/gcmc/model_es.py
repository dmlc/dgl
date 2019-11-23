"""NN modules"""
import math

import numpy as np
import mxnet as mx
import mxnet.ndarray as F
from mxnet.gluon import nn, Block
import dgl.function as fn

from utils import get_activation

import dgl

class GCMCLayer(Block):
    r"""GCMC layer

    .. math::
        z_j^{(l+1)} = \sigma_{agg}\left[\mathrm{agg}\left(
        \sum_{j\in\mathcal{N}_1}\frac{1}{c_{ij}}W_1h_j, \ldots,
        \sum_{j\in\mathcal{N}_R}\frac{1}{c_{ij}}W_Rh_j
        \right)\right]

    After that, apply an extra output projection:

    .. math::
        h_j^{(l+1)} = \sigma_{out}W_oz_j^{(l+1)}

    The equation is applied to both user nodes and movie nodes and the parameters
    are not shared unless ``share_user_item_param`` is true.

    Parameters
    ----------
    rating_vals : list of int or float
        Possible rating values.
    user_in_units : int
        Size of user input feature
    movie_in_units : int
        Size of movie input feature
    msg_units : int
        Size of message :math:`W_rh_j`
    out_units : int
        Size of of final output user and movie features
    dropout_rate : float, optional
        Dropout rate (Default: 0.0)
    agg : str, optional
        Function to aggregate messages of different ratings.
        Could be any of the supported cross type reducers:
        "sum", "max", "min", "mean", "stack".
        (Default: "stack")
    agg_act : callable, str, optional
        Activation function :math:`sigma_{agg}`. (Default: None)
    out_act : callable, str, optional
        Activation function :math:`sigma_{agg}`. (Default: None)
    share_user_item_param : bool, optional
        If true, user node and movie node share the same set of parameters.
        Require ``user_in_units`` and ``move_in_units`` to be the same.
        (Default: False)
    """
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
            self.dropout = nn.Dropout(dropout_rate)
            self.W_r = {}
            for rating in rating_vals:
                rating = str(rating)
                if share_user_item_param and user_in_units == movie_in_units:
                    self.W_r[rating] = self.params.get(
                        'W_r_%s' % rating,
                        shape=(user_in_units, msg_units),
                        dtype=np.float32, allow_deferred_init=True)
                    self.W_r['rev-%s' % rating] = self.W_r[rating]
                else:
                    self.W_r[rating] = self.params.get(
                        'W_r_%s' % rating,
                        shape=(user_in_units, msg_units),
                        dtype=np.float32, allow_deferred_init=True)
                    self.W_r['rev-%s' % rating] = self.params.get(
                        'revW_r_%s' % rating,
                        shape=(movie_in_units, msg_units),
                        dtype=np.float32, allow_deferred_init=True)
            self.ufc = nn.Dense(out_units)
            if share_user_item_param:
                self.ifc = self.ufc
            else:
                self.ifc = nn.Dense(out_units)
            self.agg_act = get_activation(agg_act)
            self.out_act = get_activation(out_act)

    def forward(self, head_graph, tail_graph, ctx):
        """Forward function

        Normalizer constant :math:`c_{ij}` is stored as two node data "ci"
        and "cj".

        Parameters
        ----------
        graph : DGLHeteroGraph
            User-movie rating graph. It should contain two node types: "user"
            and "movie" and many edge types each for one rating value.
        ufeat : mx.nd.NDArray, optional
            User features. If None, using an identity matrix.
        ifeat : mx.nd.NDArray, optional
            Movie features. If None, using an identity matrix.

        Returns
        -------
        new_ufeat : mx.nd.NDArray
            New user features
        new_ifeat : mx.nd.NDArray
            New movie features
        """
        ifeat = head_graph.nodes['movie'].data[dgl.NID].as_in_context(ctx)
        num_u = head_graph.number_of_nodes('user')
        funcs_head = {}
        for i, rating in enumerate(self.rating_vals):
            rating = str(rating)
            x_i = dot_or_identity(ifeat, self.W_r['rev-%s' % rating].data())
            x_i = x_i * self.dropout(head_graph.nodes['movie'].data['cj'])
            head_graph.nodes['movie'].data['h%d' % i] = x_i
            funcs_head['rev-%s' % rating] = (fn.copy_u('h%d' % i, 'm'), fn.sum('m', 'h'))

        head_graph.multi_update_all(funcs_head, self.agg)
        ufeat = head_graph.nodes['user'].data.pop('h')
        ufeat = ufeat.reshape((num_u, -1))
        ufeat = ufeat * head_graph.nodes['user'].data['ci']
        ufeat = self.agg_act(ufeat)
        ufeat = self.dropout(ufeat)
        ufeat = self.ufc(ufeat)
        head_feat = self.out_act(ufeat)

        ufeat = tail_graph.nodes['user'].data[dgl.NID].as_in_context(ctx)
        num_i = tail_graph.number_of_nodes('movie')
        funcs_tail = {}
        for i, rating in enumerate(self.rating_vals):
            rating = str(rating)
            x_u = dot_or_identity(ufeat, self.W_r[rating].data())
            x_u = x_u * self.dropout(tail_graph.nodes['user'].data['cj'])
            tail_graph.nodes['user'].data['h%d' % i] = x_u
            funcs_tail[rating] = (fn.copy_u('h%d' % i, 'm'), fn.sum('m', 'h'))

        tail_graph.multi_update_all(funcs_tail, self.agg)
        ifeat = tail_graph.nodes['movie'].data.pop('h')
        ifeat = ifeat.reshape((num_i, -1))
        ifeat = ifeat * tail_graph.nodes['movie'].data['ci']
        ifeat = self.agg_act(ifeat)
        ifeat = self.dropout(ifeat)
        ifeat = self.ifc(ifeat)
        tail_feat = self.out_act(ifeat)

        return head_feat, tail_feat

class BiDecoder(Block):
    r"""Bilinear decoder.

    .. math::
        p(M_{ij}=r) = \text{softmax}(u_i^TQ_rv_j)

    The trainable parameter :math:`Q_r` is further decomposed to a linear
    combination of basis weight matrices :math:`P_s`:

    .. math::
        Q_r = \sum_{s=1}^{b} a_{rs}P_s

    Parameters
    ----------
    rating_vals : list of int or float
        Possible rating values.
    in_units : int
        Size of input user and movie features
    num_basis_functions : int, optional
        Number of basis. (Default: 2)
    dropout_rate : float, optional
        Dropout raite (Default: 0.0)
    """
    def __init__(self,
                 rating_vals,
                 in_units,
                 num_basis_functions=2,
                 dropout_rate=0.0):
        super(BiDecoder, self).__init__()
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

    def forward(self, ufeat, ifeat):
        """Forward function.

        Parameters
        ----------
        ufeat : mx.nd.NDArray
            User embeddings. Shape: (|V_u|, D)
        ifeat : mx.nd.NDArray
            Movie embeddings. Shape: (|V_m|, D)

        Returns
        -------
        mx.nd.NDArray
            Predicting scores for each user-movie edge.
        """
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        basis_out = []
        for i in range(self._num_basis_functions):
            ufeat = mx.nd.dot(ufeat, self.Ps[i].data())
            basis_out.append(mx.nd.batch_dot(mx.nd.expand_dims(ufeat, axis=1), mx.nd.expand_dims(ifeat, axis=2)).squeeze().expand_dims(1))
        out = F.concat(*basis_out, dim=1)
        out = self.rate_out(out)
        return out

def dot_or_identity(A, B):
    # if A is None, treat as identity matrix
    if len(A.shape) == 1:
        return B[A]
    else:
        return mx.nd.dot(A, B)
