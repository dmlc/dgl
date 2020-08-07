"""dgl edge_softmax operator module."""
from ..backend import edge_softmax as edge_softmax_internal
from ..base import ALL

__all__ = ['edge_softmax']


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
    logits : torch.Tensor
        The input edge feature
    eids : torch.Tensor or ALL, optional
        Edges on which to apply edge softmax. If ALL, apply edge
        softmax on all edges in the graph. Default: ALL.

    Returns
    -------
    Tensor
        Softmax value

    Notes
    -----
        * Input shape: :math:`(E, *, 1)` where * means any number of
          additional dimensions, :math:`E` equals the length of eids.
        * Return shape: :math:`(E, *, 1)`

    Examples
    --------
    The following example is written in PyTorch, for other backend frameworks
    the usage is similar.

    >>> from dgl.nn.pytorch.softmax import edge_softmax
    >>> import dgl
    >>> import torch as th

    Create a :code:`DGLGraph` object and initialize its edge features.

    >>> g = dgl.DGLGraph()
    >>> g.add_nodes(3)
    >>> g.add_edges([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
    >>> edata = th.ones(6, 1).float()
    >>> edata
    tensor([[1.],
            [1.],
            [1.],
            [1.],
            [1.],
            [1.]])

    Apply edge softmax on g:

    >>> edge_softmax(g, edata)
    tensor([[1.0000],
        [0.5000],
        [0.3333],
        [0.5000],
        [0.3333],
        [0.3333]])

    Apply edge softmax on first 4 edges of g:

    >>> edge_softmax(g, edata[:4], th.Tensor([0,1,2,3]))
    tensor([[1.0000],
        [0.5000],
        [1.0000],
        [0.5000]])
    """
    return edge_softmax_internal(graph._graph, logits, eids=eids)
