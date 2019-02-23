"""Torch modules for graph related softmax."""
# pylint: disable= no-member, arguments-differ
import torch as th
from torch import nn

from ... import function as fn

__all__ = ['EdgeSoftmax']

class EdgeSoftmax(nn.Module):
    r"""Apply softmax over signals of incoming edges.

    For a node :math:`i`, edgesoftmax is an operation of computing

    .. math::
      a_ij = \frac{\exp^{z_ij}}{\sum_{j\in\mathcal{N}(i)}\exp^{z_ij}}

    where :math:`z_ij` is a signal of edge :math:`j\rightarrow i`, also
    called logits in the context of softmax. :math:`\mathcal{N}(i)` is
    the set of nodes that have an edge to :math:`i`.

    An example of using edgesoftmax is in
    `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__ where
    the attention weights are computed with such an edgesoftmax operation.

    Parameters
    ----------
    logits_name : str
        A field name such that g.edata[logits_name] gives the logits :math:`z_ij`'s,
        based on which edgesoftmax will be computed.
    max_logits_name : str
        A field name such that g.ndata[max_logits_name] will store
        :math:`\max_{j\in\mathcal{N}(i)}z_{ij}`.
    normalizer_name : str
        A field name such that g.ndata[normalizer_name] will store
        :math:`\sum_{j\in\mathcal{N}(i)}\exp^{z_{ij}}`
    """
    def __init__(self,
                 logits_name,
                 max_logits_name,
                 normalizer_name):
        super(EdgeSoftmax, self).__init__()
        # compute the softmax
        self._logits_name = logits_name
        self._max_logits_name = max_logits_name
        self._normalizer_name = normalizer_name

    def forward(self, logits, graph):
        r"""Compute edge softmax.

        Parameters
        ----------
        logits : torch.Tensor
            The input edge feature
        graph : DGLGraph
            The graph.

        Returns
        -------
        Unnormalized scores : torch.Tensor
            This part gives :math:`\exp^{z_{ij}'s`
        Normalizer : torch.Tensor
            This part gives :math:`\sum_{j\in\mathcal{N}(i)}\exp^{z_{ij}}`

        Notes
        -----
            * Input shape: :math:`(N, *, 1)` where * means any number of additional
              dimensions, :math:`N` is the number of edges.
            * Unnormalized scores shape: :math:`(N, *, 1)` where all but the last
              dimension are the same shape as the input.
            * Normalizer shape: :math:`(M, *, 1)` where :math:`M` is the number of
              nodes and all but the first and the last dimensions are the same as
              the input.

        Note that this computation is still one step away from getting real softmax
        results. The last step can be proceeded as follows:

        >>> import dgl.function as fn
        >>>
        >>> scores, normalizer = EdgeSoftmax(...).forward(logits, graph)
        >>> graph.edata['a'] = scores
        >>> graph.ndata['normalizer'] = normalizer
        >>> graph.apply_edges(lambda edges : {'a' : edges.data['a'] / edges.dst['normalizer']})

        We left this last step to users as depending on the particular use case,
        this step can be combined with other computation at once.
        """
        graph.edata[self._logits_name] = logits

        # compute the softmax
        graph.update_all(fn.copy_edge(self._logits_name, self._logits_name),
                         fn.max(self._logits_name, self._max_logits_name))
        # minus the max and exp
        graph.apply_edges(
            lambda edges: {self._logits_name : th.exp(edges.data[self._logits_name] -
                                                      edges.dst[self._max_logits_name])})
        # compute normalizer
        graph.update_all(fn.copy_edge(self._logits_name, self._logits_name),
                         fn.sum(self._logits_name, self._normalizer_name))
        return graph.edata.pop(self._logits_name), graph.ndata.pop(self._normalizer_name)

    def __repr__(self):
        return 'EdgeSoftmax(logits_name={}, max_logits_name={}, normalizer_name={})'.format(
            self._logits_name, self._max_logits_name, self._normalizer_name)
