"""Utils for tacking graph homophily and heterophily"""
from . import backend as F, function as fn

__all__ = ['node_homophily']


def node_homophily(graph, y):
    """Homophily measure from `Geom-GCN: Geometric Graph Convolutional Networks
    <https://arxiv.org/abs/2002.05287>`__

    We follow the practice of a later paper `Large Scale Learning on
    Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods
    <https://arxiv.org/abs/2110.14446>`__ to call it node homophily.

    Mathematically it is defined as follows:

    .. math::
      \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \frac{ | \{ (u,v) : u
      \in \mathcal{N}(v) \wedge y_v = y_u \} |  } { |\mathcal{N}(v)| }

    where :math:`\mathcal{V}` is the set of nodes, :math:`\mathcal{N}(v)` is
    the predecessors of node :math:`v`, and :math:`y_v` is the class of node
    :math:`v`.

    Parameters
    ----------
    graph : DGLGraph
        The graph
    y : Tensor
        The node labels, which is a tensor of shape (|V|)

    Returns
    -------
    float
        The node homophily value

    Examples
    --------
    >>> import dgl
    >>> import torch

    >>> graph = dgl.graph(([1, 2, 0, 4], [0, 1, 2, 3]))
    >>> y = torch.tensor([0, 0, 0, 0, 1])
    >>> dgl.node_homophily(graph, y)
    0.6000000238418579
    """
    with graph.local_scope():
        src, dst = graph.edges()
        # Handle the case where graph is of dtype int32.
        src = F.astype(src, F.int64)
        dst = F.astype(dst, F.int64)
        # Compute y_v = y_u for all edges.
        graph.edata['same_class'] = F.astype(y[src] == y[dst], F.float32)
        graph.update_all(
            fn.copy_e('same_class', 'm'), fn.mean('m', 'node_value')
        )
        return graph.ndata['node_value'].mean().item()
