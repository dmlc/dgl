"""Torch modules for graph related softmax."""
# pylint: disable= no-member, arguments-differ
import torch as th

from ... import ndarray as nd
from ... import backend as F
from ... import utils
from ... import function as fn
from ...runtime import spmv

__all__ = ['EdgeSoftmax']


class EdgeSoftmax(object):
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

    def __call__(self, logits, graph):
        r"""Compute edge softmax.

        Parameters
        ----------
        logits : torch.Tensor
            The input edge feature
        graph : DGLGraph
            The graph to perform edge softmax

        Returns
        -------
        Unnormalized scores : torch.Tensor
            This part gives :math:`\exp(z_{ij})`'s
        Normalizer : torch.Tensor
            This part gives :math:`\sum_{j\in\mathcal{N}(i)}\exp(z_{ij})`

        Notes
        -----
            * Input shape: :math:`(N, *, 1)` where * means any number of
              additional dimensions, :math:`N` is the number of edges.
            * Unnormalized scores shape: :math:`(N, *, 1)` where all but the
              last dimension are the same shape as the input.
            * Normalizer shape: :math:`(M, *, 1)` where :math:`M` is the number
              of nodes and all but the first and the last dimensions are the
              same as the input.

        Note that this computation is still one step away from getting real
        softmax results. The last step can be proceeded as follows:

        >>> import dgl.function as fn
        >>> scores, normalizer = EdgeSoftmax(logits, graph)
        >>> graph.edata['a'] = scores
        >>> graph.ndata['normalizer'] = normalizer
        >>> graph.apply_edges(
                lambda edges: {'a': edges.data['a'] / edges.dst['normalizer']})

        We left this last step to users as depending on the particular use
        case, this step can be combined with other computation at once.
        """
        num_nodes = graph.number_of_nodes()
        ctx = utils.to_dgl_context(F.context(logits))
        gidx, _, nbits = spmv.build_adj_matrix_graph(graph)
        gidx = gidx(ctx)
        _, dst, _ = graph._graph.edges()
        dst = dst.tousertensor(F.context(logits))
        empty_map = (None, None)
        max_logits_ = F.copy_reduce("max", gidx, fn.TargetCode.EDGE, logits,
                                    num_nodes, empty_map, empty_map)
        logits = (logits - max_logits_.index_select(0, dst)).exp()
        norm = F.copy_reduce("sum", gidx, fn.TargetCode.EDGE, logits,
                             num_nodes, empty_map, empty_map)
        return logits, norm

class EdgeSoftmax1(th.autograd.Function):
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
        g.update_all(fn.copy_edge(score_name, 'm'), fn.max('m', tmp_name))
        g.apply_edges(fn.edge_sub_dst(score_name, tmp_name, out_name))
        g.edata[out_name] = th.exp(g.edata[out_name])
        g.update_all(fn.copy_edge(out_name, 'm'), fn.sum('m', tmp_name))
        g.apply_edges(fn.edge_div_dst(out_name, tmp_name, out_name))
        g.edata.pop(score_name)
        g.ndata.pop(tmp_name)
        out = g.edata.pop(out_name)
        ctx.backward_cache = (g, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        g, out = ctx.backward_cache
        grad_out = dgl.EData(g, grad_out)
        out = dgl.EData(g, out)
        sds = out * grad_out  # type dgl.EData
        sds_sum = sds.dst_sum()  # type dgl.NData
        grad_score = sds - sds * sds_sum  # multiple expressions
        return grad_score.data
        """
        g, out = ctx.backward_cache
        out_name = utils.get_edata_name(g, 'out')
        accum_name = utils.get_ndata_name(g, 'accum')
        tmp_score_name = utils.get_edata_name(g, 'tmp_score')
        grad_score_name = utils.get_edata_name(g, 'grad_score')
        g.edata[out_name] = out
        g.edata[grad_score_name] = out * grad_out
        g.update_all(fn.copy_edge(grad_score_name, 'm'), fn.sum('m', accum_name))
        g.apply_edges(fn.edge_mul_dst(out_name, accum_name, out_name))
        grad_score = g.edata[grad_score_name] - g.edata[out_name]
        return grad_score

edge_softmax = EdgeSoftmax1.apply
