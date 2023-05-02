"""Utilities for pytorch NN package"""
# pylint: disable=no-member, invalid-name

import numpy as np
from mxnet import gluon, nd

from ... import DGLGraph


def matmul_maybe_select(A, B):
    """Perform Matrix multiplication C = A * B but A could be an integer id vector.

    If A is an integer vector, we treat it as multiplying a one-hot encoded tensor.
    In this case, the expensive dense matrix multiply can be replaced by a much
    cheaper index lookup.

    For example,
    ::

        A = [2, 0, 1],
        B = [[0.1, 0.2],
             [0.3, 0.4],
             [0.5, 0.6]]

    then matmul_maybe_select(A, B) is equivalent to
    ::

        [[0, 0, 1],     [[0.1, 0.2],
         [1, 0, 0],  *   [0.3, 0.4],
         [0, 1, 0]]      [0.5, 0.6]]

    In all other cases, perform a normal matmul.

    Parameters
    ----------
    A : mxnet.NDArray
        lhs tensor
    B : mxnet.NDArray
        rhs tensor

    Returns
    -------
    C : mxnet.NDArray
        result tensor
    """
    if A.dtype in (np.int32, np.int64) and len(A.shape) == 1:
        return nd.take(B, A, axis=0)
    else:
        return nd.dot(A, B)


def bmm_maybe_select(A, B, index):
    """Slice submatrices of A by the given index and perform bmm.

    B is a 3D tensor of shape (N, D1, D2), which can be viewed as a stack of
    N matrices of shape (D1, D2). The input index is an integer vector of length M.
    A could be either:
    (1) a dense tensor of shape (M, D1),
    (2) an integer vector of length M.
    The result C is a 2D matrix of shape (M, D2)

    For case (1), C is computed by bmm:
    ::

        C[i, :] = matmul(A[i, :], B[index[i], :, :])

    For case (2), C is computed by index select:
    ::

        C[i, :] = B[index[i], A[i], :]

    Parameters
    ----------
    A : mxnet.NDArray
        lhs tensor
    B : mxnet.NDArray
        rhs tensor
    index : mxnet.NDArray
        index tensor

    Returns
    -------
    C : mxnet.NDArray
        return tensor
    """
    if A.dtype in (np.int32, np.int64) and len(A.shape) == 1:
        return B[index, A, :]
    else:
        BB = nd.take(B, index, axis=0)
        return nd.batch_dot(A.expand_dims(1), BB).squeeze(1)


def normalize(x, p=2, axis=1, eps=1e-12):
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    For a tensor :attr:`input` of sizes :math:`(n_0, ..., n_{dim}, ..., n_k)`, each
    :math:`n_{dim}` -element vector :math:`v` along dimension :attr:`dim` is transformed as

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}.

    With the default arguments it uses the Euclidean norm over vectors along dimension
     :math:`1` for normalization.

    Args:
        x: input ndarray of any shape
        ord (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
        eps (float): small value to avoid division by zero. Default: 1e-12
    """
    denom = nd.clip(
        nd.norm(x, ord=p, axis=axis, keepdims=True), eps, float("inf")
    )
    return x / denom


class Sequential(gluon.nn.Sequential):
    r"""A squential container for stacking graph neural network blocks

    We support two modes: sequentially apply GNN blocks on the same graph or
    a list of given graphs. In the second case, the number of graphs equals the
    number of blocks inside this container.

    Examples
    --------

    Mode 1: sequentially apply GNN modules on the same graph

    >>> import dgl
    >>> from mxnet import nd
    >>> from mxnet.gluon import nn
    >>> import dgl.function as fn
    >>> from dgl.nn.mxnet import Sequential
    >>> class ExampleLayer(nn.Block):
    >>>     def __init__(self, **kwargs):
    >>>         super().__init__(**kwargs)
    >>>     def forward(self, graph, n_feat, e_feat):
    >>>         with graph.local_scope():
    >>>             graph.ndata['h'] = n_feat
    >>>             graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
    >>>             n_feat += graph.ndata['h']
    >>>             graph.apply_edges(fn.u_add_v('h', 'h', 'e'))
    >>>             e_feat += graph.edata['e']
    >>>             return n_feat, e_feat
    >>>
    >>> g = dgl.DGLGraph()
    >>> g.add_nodes(3)
    >>> g.add_edges([0, 1, 2, 0, 1, 2, 0, 1, 2], [0, 0, 0, 1, 1, 1, 2, 2, 2])
    >>> net = Sequential()
    >>> net.add(ExampleLayer())
    >>> net.add(ExampleLayer())
    >>> net.add(ExampleLayer())
    >>> net.initialize()
    >>> n_feat = nd.random.randn(3, 4)
    >>> e_feat = nd.random.randn(9, 4)
    >>> net(g, n_feat, e_feat)
    (
    [[ 12.412863   99.61184    21.472883  -57.625923 ]
     [ 10.08097   100.68611    20.627377  -60.13458  ]
     [ 11.7912245 101.80654    22.427956  -58.32772  ]]
    <NDArray 3x4 @cpu(0)>,
    [[  21.818504  198.12076    42.72387  -115.147736]
     [  23.070837  195.49811    43.42292  -116.17203 ]
     [  24.330334  197.10927    42.40048  -118.06538 ]
     [  21.907919  199.11469    42.1187   -115.35658 ]
     [  22.849625  198.79213    43.866085 -113.65381 ]
     [  20.926125  198.116      42.64334  -114.246704]
     [  23.003159  197.06662    41.796425 -117.14977 ]
     [  21.391375  198.3348     41.428078 -116.30361 ]
     [  21.291483  200.0701     40.8239   -118.07314 ]]
    <NDArray 9x4 @cpu(0)>)

    Mode 2: sequentially apply GNN modules on different graphs

    >>> import dgl
    >>> from mxnet import nd
    >>> from mxnet.gluon import nn
    >>> import dgl.function as fn
    >>> import networkx as nx
    >>> from dgl.nn.mxnet import Sequential
    >>> class ExampleLayer(nn.Block):
    >>>     def __init__(self, **kwargs):
    >>>         super().__init__(**kwargs)
    >>>     def forward(self, graph, n_feat):
    >>>         with graph.local_scope():
    >>>             graph.ndata['h'] = n_feat
    >>>             graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
    >>>             n_feat += graph.ndata['h']
    >>>             return n_feat.reshape(graph.num_nodes() // 2, 2, -1).sum(1)
    >>>
    >>> g1 = dgl.DGLGraph(nx.erdos_renyi_graph(32, 0.05))
    >>> g2 = dgl.DGLGraph(nx.erdos_renyi_graph(16, 0.2))
    >>> g3 = dgl.DGLGraph(nx.erdos_renyi_graph(8, 0.8))
    >>> net = Sequential()
    >>> net.add(ExampleLayer())
    >>> net.add(ExampleLayer())
    >>> net.add(ExampleLayer())
    >>> net.initialize()
    >>> n_feat = nd.random.randn(32, 4)
    >>> net([g1, g2, g3], n_feat)
    [[-101.289566  -22.584694  -89.25348  -151.6447  ]
     [-130.74239   -49.494812 -120.250854 -199.81546 ]
     [-112.32089   -50.036713 -116.13266  -190.38638 ]
     [-119.23065   -26.78553  -111.11185  -166.08322 ]]
    <NDArray 4x4 @cpu(0)>
    """

    def __init__(self, prefix=None, params=None):
        super(Sequential, self).__init__(prefix=prefix, params=params)

    def forward(self, graph, *feats):
        r"""Sequentially apply modules to the input.

        Parameters
        ----------
        graph : DGLGraph or list of DGLGraphs
            The graph(s) to apply modules on.

        *feats :
            Input features.
            The output of :math:`i`-th block should match that of the input
            of :math:`(i+1)`-th block.
        """
        if isinstance(graph, list):
            for graph_i, module in zip(graph, self):
                if not isinstance(feats, tuple):
                    feats = (feats,)
                feats = module(graph_i, *feats)
        elif isinstance(graph, DGLGraph):
            for module in self:
                if not isinstance(feats, tuple):
                    feats = (feats,)
                feats = module(graph, *feats)
        else:
            raise TypeError(
                "The first argument of forward must be a DGLGraph"
                " or a list of DGLGraph s"
            )
        return feats
