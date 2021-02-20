"""dgl edge_softmax operator module."""
from ..backend import edge_softmax as edge_softmax_internal
from ..backend import astype
from ..base import ALL, is_all

__all__ = ['edge_softmax']


def edge_softmax(graph, logits, eids=ALL, norm_by='dst'):
    r"""Compute softmax over weights of incoming edges for every node.

    For a node :math:`i`, edge softmax is an operation that computes

    .. math::
      a_{ij} = \frac{\exp(z_{ij})}{\sum_{j\in\mathcal{N}(i)}\exp(z_{ij})}

    where :math:`z_{ij}` is a signal of edge :math:`j\rightarrow i`, also
    called logits in the context of softmax. :math:`\mathcal{N}(i)` is
    the set of nodes that have an edge to :math:`i`.

    By default edge softmax is normalized by destination nodes(i.e. :math:`ij`
    are incoming edges of `i` in the formula above). We also support edge
    softmax normalized by source nodes(i.e. :math:`ij` are outgoing edges of
    `i` in the formula). The former case corresponds to softmax in GAT and
    Transformer, and the latter case corresponds to softmax in Capsule network.
    An example of using edge softmax is in
    `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__ where
    the attention weights are computed with this operation.
    Other non-GNN examples using this are
    `Transformer <https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf>`__,
    `Capsule <https://arxiv.org/pdf/1710.09829.pdf>`__, etc.

    Parameters
    ----------
    graph : DGLGraph
        The graph over which edge softmax will be performed.
    logits : torch.Tensor
        The input edge feature.
    eids : torch.Tensor or ALL, optional
        The IDs of the edges to apply edge softmax. If ALL, it will apply edge
        softmax to all edges in the graph. Default: ALL.
    norm_by : str, could be `src` or `dst`
        Normalized by source nodes or destination nodes. Default: `dst`.

    Returns
    -------
    Tensor
        Softmax value.

    Notes
    -----
        * Input shape: :math:`(E, *, 1)` where * means any number of
          additional dimensions, :math:`E` equals the length of eids.
          If the `eids` is ALL, :math:`E` equals the number of edges in
          the graph.
        * Return shape: :math:`(E, *, 1)`

    Examples
    --------
    The following example uses PyTorch backend.

    >>> from dgl.nn.functional import edge_softmax
    >>> import dgl
    >>> import torch as th

    Create a :code:`DGLGraph` object and initialize its edge features.

    >>> g = dgl.graph((th.tensor([0, 0, 0, 1, 1, 2]), th.tensor([0, 1, 2, 1, 2, 2])))
    >>> edata = th.ones(6, 1).float()
    >>> edata
        tensor([[1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.]])

    Apply edge softmax over g:

    >>> edge_softmax(g, edata)
        tensor([[1.0000],
                [0.5000],
                [0.3333],
                [0.5000],
                [0.3333],
                [0.3333]])

    Apply edge softmax over g normalized by source nodes:

    >>> edge_softmax(g, edata, norm_by='src')
        tensor([[0.3333],
                [0.3333],
                [0.3333],
                [0.5000],
                [0.5000],
                [1.0000]])

    Apply edge softmax to first 4 edges of g:

    >>> edge_softmax(g, edata[:4], th.Tensor([0,1,2,3]))
        tensor([[1.0000],
                [0.5000],
                [1.0000],
                [0.5000]])
    """
    if not is_all(eids):
        eids = astype(eids, graph.idtype)
    return edge_softmax_internal(graph._graph, logits,
                                 eids=eids, norm_by=norm_by)
