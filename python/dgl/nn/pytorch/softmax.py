"""Torch modules for graph related softmax."""
# pylint: disable= no-member, arguments-differ
from ... import ndarray as nd
from ... import backend as F

import scipy.sparse as sp
import numpy as np

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

    def __call__(logits, edge_tuple, num_nodes):
        r"""Compute edge softmax.

        Parameters
        ----------
        logits : torch.Tensor
            The input edge feature
        edge_tuple : (torch.Tensor, torch.Tensor, torch.Tensor)
            A tuple of representing the edges to perform softmax. The three
            elements in the tuple are source node ids, destination node ids,
            and edge ids
        num_nodes : int
            Number of nodes in the graph

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
        >>> edge_tuple = graph.all_edges('all')
        >>> num_nodes = graph.number_of_nodes()
        >>> scores, normalizer = EdgeSoftmax(logits, edge_tuple, num_nodes)
        >>> graph.edata['a'] = scores
        >>> graph.ndata['normalizer'] = normalizer
        >>> graph.apply_edges(
                lambda edges: {'a': edges.data['a'] / edges.dst['normalizer']})

        We left this last step to users as depending on the particular use
        case, this step can be combined with other computation at once.
        """
        _, v, _ = edge_tuple
        indptr, indices, inv_indptr, inv_indices, edge_map, inv_edge_map \
            = _build_adj_and_edge_map(edge_tuple, F.context(logits))
        spmat = (indptr, indices, inv_indptr, inv_indices)
        out_map = nd.empty([])
        max_logits_ = F.copy_edge_reduce("max", spmat, logits, num_nodes,
                                         (edge_map, inv_edge_map), out_map)
        logits = (logits - max_logits_[v]).exp()
        norm = F.copy_edge_reduce("sum", spmat, logits, num_nodes,
                                  (edge_map, inv_edge_map), out_map)
        return logits, norm


def _build_adj_and_edge_map(edge_tuple, num_nodes, ctx):
    u, v, eid = edge_tuple
    u = u.numpy()
    v = v.numpy()
    dat = np.arange(len(v), dtype=np.int64)
    csr = sp.csr_matrix((dat, (u, v)), shape=(num_nodes, num_nodes))
    inv_csr = sp.csr_matrix((dat, (v, u)), shape=(num_nodes, num_nodes))
    res = [
        F.copy_to(F.zerocopy_from_numpy(csr.indptr.astype(np.int64))),
        F.copy_to(F.zerocopy_from_numpy(csr.indices.astype(np.int64))),
        F.copy_to(F.zerocopy_from_numpy(inv_csr.indptr.astype(np.int64))),
        F.copy_to(F.zerocopy_from_numpy(inv_csr.indices.astype(np.int64))),
        F.copy_to(eid[csr.data]),
        F.copy_to(eid[inv_csr.data]),
    ]
    return res
