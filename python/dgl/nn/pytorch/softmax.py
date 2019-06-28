"""Torch modules for graph related softmax."""
# pylint: disable= no-member, arguments-differ
import torch as th

from ... import utils
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
        """
        score = dgl.EData(g, score)
        score_max = score.dst_max()  # of type dgl.NData
        score = score - score_max  # edge_sub_dst, ret dgl.EData
        score_sum = score.dst_sum()  # of type dgl.NData
        out = score / score_sum    # edge_div_dst, ret dgl.EData
        return out.data
        """
        score_name = utils.get_edata_name(g, 'score')
        tmp_name = utils.get_ndata_name(g, 'tmp')
        out_name = utils.get_edata_name(g, 'out')
        g.edata[score_name] = score
        g.update_all(fn.copy_e(score_name, 'm'), fn.max('m', tmp_name))
        g.apply_edges(fn.e_sub_v(score_name, tmp_name, out_name))
        g.edata[out_name] = th.exp(g.edata[out_name])
        g.update_all(fn.copy_e(out_name, 'm'), fn.sum('m', tmp_name))
        g.apply_edges(fn.e_div_v(out_name, tmp_name, out_name))
        g.edata.pop(score_name)
        g.ndata.pop(tmp_name)
        out = g.edata.pop(out_name)
        ctx.save_for_backward(out)
        ctx.backward_cache = g
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        g, out = ctx.backward_cache
        grad_out = dgl.EData(g, grad_out)
        out = dgl.EData(g, out)
        sds = out * grad_out  # type dgl.EData
        sds_sum = sds.dst_sum()  # type dgl.NData
        grad_score = sds - out * sds_sum  # multiple expressions
        return grad_score.data
        """
        g = ctx.backward_cache
        out, = ctx.saved_tensors
        # clear backward cache explicitly
        ctx.backward_cache = None
        out_name = utils.get_edata_name(g, 'out')
        accum_name = utils.get_ndata_name(g, 'accum')
        grad_score_name = utils.get_edata_name(g, 'grad_score')
        g.edata[out_name] = out
        g.edata[grad_score_name] = out * grad_out
        g.update_all(fn.copy_e(grad_score_name, 'm'), fn.sum('m', accum_name))
        g.apply_edges(fn.e_mul_v(out_name, accum_name, out_name))
        g.ndata.pop(accum_name)
        grad_score = g.edata.pop(grad_score_name) - g.edata.pop(out_name)
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
