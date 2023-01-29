"""Edge coarsening procedure used in Metis and Graclus, for pytorch"""
# pylint: disable=no-member, invalid-name, W0613
from .. import remove_self_loop
from .capi import _neighbor_matching

__all__ = ["neighbor_matching"]


def neighbor_matching(graph, e_weights=None, relabel_idx=True):
    r"""
    Description
    -----------
    The neighbor matching procedure of edge coarsening in
    `Metis <http://cacs.usc.edu/education/cs653/Karypis-METIS-SIAMJSC98.pdf>`__
    and
    `Graclus <https://www.cs.utexas.edu/users/inderjit/public_papers/multilevel_pami.pdf>`__
    for homogeneous graph coarsening. This procedure keeps picking an unmarked
    vertex and matching it with one its unmarked neighbors (that maximizes its
    edge weight) until no match can be done.

    If no edge weight is given, this procedure will randomly pick neighbor for each
    vertex.

    The GPU implementation is based on `A GPU Algorithm for Greedy Graph Matching
    <http://www.staff.science.uu.nl/~bisse101/Articles/match12.pdf>`__

    NOTE: The input graph must be bi-directed (undirected) graph. Call :obj:`dgl.to_bidirected`
          if you are not sure your graph is bi-directed.

    Parameters
    ----------
    graph : DGLGraph
        The input homogeneous graph.
    edge_weight : torch.Tensor, optional
        The edge weight tensor holding non-negative scalar weight for each edge.
        default: :obj:`None`
    relabel_idx : bool, optional
        If true, relabel resulting node labels to have consecutive node ids.
        default: :obj:`True`

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import torch, dgl
    >>> from dgl.geometry import neighbor_matching
    >>>
    >>> g = dgl.graph(([0, 1, 1, 2], [1, 0, 2, 1]))
    >>> res = neighbor_matching(g)
        tensor([0, 1, 1])
    """
    assert (
        graph.is_homogeneous
    ), "The graph used in graph node matching must be homogeneous"
    if e_weights is not None:
        graph.edata["e_weights"] = e_weights
        graph = remove_self_loop(graph)
        e_weights = graph.edata["e_weights"]
        graph.edata.pop("e_weights")
    else:
        graph = remove_self_loop(graph)
    return _neighbor_matching(
        graph._graph, graph.num_nodes(), e_weights, relabel_idx
    )
