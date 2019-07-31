"""Gluon layer for graph related softmax."""
# pylint: disable= no-member, arguments-differ
import mxnet as mx

from ... import function as fn

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

    def __init__(self, g):
        super(EdgeSoftmax, self).__init__()
        self.g = g

    def forward(self, score):
        g = self.g.local_scope()
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
        g = self.g.local_scope()
        out, = self.saved_tensors  # pylint: disable=access-member-before-definition, unpacking-non-sequence
        # clear saved tensors explicitly
        self.saved_tensors = None
        g.edata['out'] = out
        g.edata['grad_score'] = out * grad_out
        g.update_all(fn.copy_e('grad_score', 'm'), fn.sum('m', 'accum'))
        g.apply_edges(fn.e_mul_v('out', 'accum', 'out'))
        grad_score = g.edata['grad_score'] - g.edata['out']
        return grad_score

def edge_softmax(graph, logits):
    r"""Compute edge softmax.

    Parameters
    ----------
    graph : DGLGraph
        The graph to perform edge softmax
    logits : torch.Tensor
        The input edge feature

    Returns
    -------
    Tensor
        Softmax value

    Notes
    -----
        * Input shape: :math:`(N, *, 1)` where * means any number of
            additional dimensions, :math:`N` is the number of edges.
        * Return shape: :math:`(N, *, 1)`

    Examples
    --------
    >>> import dgl.function as fn
    >>> attention = EdgeSoftmax(logits, graph)
    """
    softmax_op = EdgeSoftmax(graph)
    return softmax_op(logits)
