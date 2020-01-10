"""Gluon layer for graph related softmax."""
# pylint: disable= no-member, arguments-differ
import mxnet as mx

from ... import function as fn
from ...base import ALL, is_all

__all__ = ['edge_softmax']


class EdgeSoftmax(mx.autograd.Function):
    r"""Apply softmax over signals of incoming edges.

    For a node :math:`i`, edgesoftmax is an operation of computing

    .. math::
      a_{ij} = \frac{\exp(z_{ij})}{\sum_{j\in\mathcal{N}(i)}\exp(z_{ij})}

    where :math:`z_{ij}` is a signal of edge :math:`j\rightarrow i`, also
    called logits in the context of softmax. :math:`\mathcal{N}(i)` is
    the set of nodes that have an edge to :math:`i`.

    An example of using edgesoftmax is in
    `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__ where
    the attention weights are computed with such an edgesoftmax operation.
    """

    def __init__(self, g, eids):
        super(EdgeSoftmax, self).__init__()
        if not is_all(eids):
            g = g.edge_subgraph(eids.astype('int64'))
        self.g = g

    def forward(self, score):
        """Forward function.

        Pseudo-code:

        .. code:: python

            score = dgl.EData(g, score)
            score_max = score.dst_max()  # of type dgl.NData
            score = score - score_max  # edge_sub_dst, ret dgl.EData
            score_sum = score.dst_sum()  # of type dgl.NData
            out = score / score_sum    # edge_div_dst, ret dgl.EData
            return out.data
        """
        g = self.g.local_var()
        g.edata['s'] = score
        g.update_all(fn.copy_e('s', 'm'), fn.max('m', 'smax'))
        g.apply_edges(fn.e_sub_v('s', 'smax', 'out'))
        g.edata['out'] = g.edata['out'].exp()
        g.update_all(fn.copy_e('out', 'm'), fn.sum('m', 'out_sum'))
        g.apply_edges(fn.e_div_v('out', 'out_sum', 'out'))
        out = g.edata['out']
        self.save_for_backward(out)
        return out

    def backward(self, grad_out):
        """Backward function.

        Pseudo-code:

        .. code:: python

            g, out = ctx.backward_cache
            grad_out = dgl.EData(g, grad_out)
            out = dgl.EData(g, out)
            sds = out * grad_out  # type dgl.EData
            sds_sum = sds.dst_sum()  # type dgl.NData
            grad_score = sds - sds * sds_sum  # multiple expressions
        """
        g = self.g.local_var()
        out, = self.saved_tensors  # pylint: disable=access-member-before-definition, unpacking-non-sequence
        # clear saved tensors explicitly
        self.saved_tensors = None
        g.edata['out'] = out
        g.edata['grad_score'] = out * grad_out
        g.update_all(fn.copy_e('grad_score', 'm'), fn.sum('m', 'accum'))
        g.apply_edges(fn.e_mul_v('out', 'accum', 'out'))
        grad_score = g.edata['grad_score'] - g.edata['out']
        return grad_score

def edge_softmax(graph, logits, eids=ALL):
    r"""Compute edge softmax.

    For a node :math:`i`, edge softmax is an operation of computing

    .. math::
      a_{ij} = \frac{\exp(z_{ij})}{\sum_{j\in\mathcal{N}(i)}\exp(z_{ij})}

    where :math:`z_{ij}` is a signal of edge :math:`j\rightarrow i`, also
    called logits in the context of softmax. :math:`\mathcal{N}(i)` is
    the set of nodes that have an edge to :math:`i`.

    An example of using edge softmax is in
    `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__ where
    the attention weights are computed with such an edge softmax operation.

    Parameters
    ----------
    graph : DGLGraph
        The graph to perform edge softmax
    logits : mxnet.NDArray
        The input edge feature
    eids : mxnet.NDArray or ALL, optional
        Edges on which to apply edge softmax. If ALL, apply edge softmax
        on all edges in the graph. Default: ALL.

    Returns
    -------
    Tensor
        Softmax value

    Notes
    -----
        * Input shape: :math:`(E, *, 1)` where * means any number of
          additional dimensions, :math:`E` equals the length of eids.
          If eids is ALL, :math:`E` equals number of edges in the graph.
        * Return shape: :math:`(E, *, 1)`

    Examples
    --------
    >>> from dgl.nn.mxnet.softmax import edge_softmax
    >>> import dgl
    >>> from mxnet import nd

    Create a :code:`DGLGraph` object and initialize its edge features.

    >>> g = dgl.DGLGraph()
    >>> g.add_nodes(3)
    >>> g.add_edges([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
    >>> edata = nd.ones((6, 1))
    >>> edata
    [[1.]
     [1.]
     [1.]
     [1.]
     [1.]
     [1.]]
    <NDArray 6x1 @cpu(0)>

    Apply edge softmax on g:

    >>> edge_softmax(g, edata)
    [[1.        ]
     [0.5       ]
     [0.33333334]
     [0.5       ]
     [0.33333334]
     [0.33333334]]
    <NDArray 6x1 @cpu(0)>

    Apply edge softmax on first 4 edges of g:

    >>> edge_softmax(g, edata, nd.array([0,1,2,3], dtype='int64'))
    [[1. ]
     [0.5]
     [1. ]
     [0.5]]
    <NDArray 4x1 @cpu(0)>
    """
    softmax_op = EdgeSoftmax(graph, eids)
    return softmax_op(logits)
