"""Torch modules for graph related softmax."""
# pylint: disable= no-member, arguments-differ
from ... import ndarray as nd
from ... import backend as F
from ... import utils

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
        csr = graph._graph.csr_adjacency_matrix(True, ctx)
        inv_csr = graph._graph.csr_adjacency_matrix(False, ctx)
        _, dst, _ = graph._graph.edges(None, ctx)
        dst = dst.tousertensor()
        indptr, indices, edge_map = csr
        inv_indptr, inv_indices, inv_edge_map = inv_csr
        spmat = (indptr, indices, inv_indptr, inv_indices)
        out_map = nd.empty([])
        max_logits_ = F.copy_edge_reduce("max", spmat, logits, num_nodes,
                                         (edge_map, inv_edge_map), out_map)
        logits = (logits - max_logits_.index_select(0, dst)).exp()
        norm = F.copy_edge_reduce("sum", spmat, logits, num_nodes,
                                  (edge_map, inv_edge_map), out_map)
        return logits, norm
