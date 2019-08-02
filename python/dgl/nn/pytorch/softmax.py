"""Torch modules for graph related softmax."""
# pylint: disable= no-member, arguments-differ
import torch as th

from ... import function as fn

__all__ = ['edge_softmax']


class EdgeSoftmax(th.autograd.Function):
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

    @staticmethod
    def forward(ctx, g, score):
        """Forward function.

        Pseudo-code:
        score = dgl.EData(g, score)
        score_max = score.dst_max()  # of type dgl.NData
        score = score - score_max  # edge_sub_dst, ret dgl.EData
        score_sum = score.dst_sum()  # of type dgl.NData
        out = score / score_sum    # edge_div_dst, ret dgl.EData
        return out.data
        """
        # remember to save the graph to backward cache before making it
        # a local variable
        ctx.backward_cache = g
        g = g.local_var()
        g.edata['s'] = score
        g.update_all(fn.copy_e('s', 'm'), fn.max('m', 'smax'))
        g.apply_edges(fn.e_sub_v('s', 'smax', 'out'))
        g.edata['out'] = th.exp(g.edata['out'])
        g.update_all(fn.copy_e('out', 'm'), fn.sum('m', 'out_sum'))
        g.apply_edges(fn.e_div_v('out', 'out_sum', 'out'))
        out = g.edata['out']
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """Backward function.

        Pseudo-code:
        g, out = ctx.backward_cache
        grad_out = dgl.EData(g, grad_out)
        out = dgl.EData(g, out)
        sds = out * grad_out  # type dgl.EData
        sds_sum = sds.dst_sum()  # type dgl.NData
        grad_score = sds - sds * sds_sum  # multiple expressions
        return grad_score.data
        """
        g = ctx.backward_cache
        g = g.local_var()
        out, = ctx.saved_tensors
        # clear backward cache explicitly
        ctx.backward_cache = None
        g.edata['out'] = out
        g.edata['grad_s'] = out * grad_out
        g.update_all(fn.copy_e('grad_s', 'm'), fn.sum('m', 'accum'))
        g.apply_edges(fn.e_mul_v('out', 'accum', 'out'))
        grad_score = g.edata['grad_s'] - g.edata['out']
        return None, grad_score


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
    return EdgeSoftmax.apply(graph, logits)
