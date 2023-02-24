"""Utils for tacking graph homophily and heterophily"""
from . import backend as F, function as fn
from .convert import graph as create_graph

try:
    import torch
except ImportError:
    pass

__all__ = ["node_homophily", "edge_homophily", "linkx_homophily"]


def node_homophily(graph, y):
    r"""Homophily measure from `Geom-GCN: Geometric Graph Convolutional Networks
    <https://arxiv.org/abs/2002.05287>`__

    We follow the practice of a later paper `Large Scale Learning on
    Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods
    <https://arxiv.org/abs/2110.14446>`__ to call it node homophily.

    Mathematically it is defined as follows:

    .. math::
      \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \frac{ | \{u
      \in \mathcal{N}(v): y_v = y_u \} |  } { |\mathcal{N}(v)| }

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
        graph.edata["same_class"] = F.astype(y[src] == y[dst], F.float32)
        graph.update_all(
            fn.copy_e("same_class", "m"), fn.mean("m", "same_class_deg")
        )
        return F.as_scalar(F.mean(graph.ndata["same_class_deg"], dim=0))


def edge_homophily(graph, y):
    r"""Homophily measure from `Beyond Homophily in Graph Neural Networks:
    Current Limitations and Effective Designs
    <https://arxiv.org/abs/2006.11468>`__

    Mathematically it is defined as follows:

    .. math::
      \frac{| \{ (u,v) : (u,v) \in \mathcal{E} \wedge y_u = y_v \} | }
      {|\mathcal{E}|}

    where :math:`\mathcal{E}` is the set of edges, and :math:`y_u` is the class
    of node :math:`u`.

    Parameters
    ----------
    graph : DGLGraph
        The graph
    y : Tensor
        The node labels, which is a tensor of shape (|V|)

    Returns
    -------
    float
        The edge homophily ratio value

    Examples
    --------
    >>> import dgl
    >>> import torch

    >>> graph = dgl.graph(([1, 2, 0, 4], [0, 1, 2, 3]))
    >>> y = torch.tensor([0, 0, 0, 0, 1])
    >>> dgl.edge_homophily(graph, y)
    0.75
    """
    with graph.local_scope():
        src, dst = graph.edges()
        # Handle the case where graph is of dtype int32.
        src = F.astype(src, F.int64)
        dst = F.astype(dst, F.int64)
        # Compute y_v = y_u for all edges.
        edge_indicator = F.astype(y[src] == y[dst], F.float32)
        return F.as_scalar(F.mean(edge_indicator, dim=0))


def linkx_homophily(graph, y):
    r"""Homophily measure from `Large Scale Learning on Non-Homophilous Graphs:
    New Benchmarks and Strong Simple Methods
    <https://arxiv.org/abs/2110.14446>`__

    Mathematically it is defined as follows:

    .. math::
      \frac{1}{C-1} \sum_{k=1}^{C} \max \left(0, \frac{\sum_{v\in C_k}|\{u\in
      \mathcal{N}(v): y_v = y_u \}|}{\sum_{v\in C_k}|\mathcal{N}(v)|} -
      \frac{|\mathcal{C}_k|}{|\mathcal{V}|} \right)

    where :math:`C` is the number of node classes, :math:`C_k` is the set of
    nodes that belong to class k, :math:`\mathcal{N}(v)` is the predecessors of
    node :math:`v`, :math:`y_v` is the class of node :math:`v`, and
    :math:`\mathcal{V}` is the set of nodes.

    Parameters
    ----------
    graph : DGLGraph
        The graph
    y : Tensor
        The node labels, which is a tensor of shape (|V|)

    Returns
    -------
    float
        The homophily value

    Examples
    --------
    >>> import dgl
    >>> import torch

    >>> graph = dgl.graph(([0, 1, 2, 3], [1, 2, 0, 4]))
    >>> y = torch.tensor([0, 0, 0, 0, 1])
    >>> dgl.linkx_homophily(graph, y)
    0.19999998807907104
    """
    with graph.local_scope():
        # Compute |{u\in N(v): y_v = y_u}| for each node v.
        src, dst = graph.edges()
        # Handle the case where graph is of dtype int32.
        src = src.long()
        dst = dst.long()
        # Compute y_v = y_u for all edges.
        graph.edata["same_class"] = (y[src] == y[dst]).float()
        graph.update_all(
            fn.copy_e("same_class", "m"), fn.mean("m", "same_class_deg")
        )

        # Compute |N(v)| for each node v.
        deg = graph.in_degrees().float()

        # To compute \sum_{v\in C_k}|{u\in N(v): y_v = y_u}| for all k
        # efficiently, construct a directed graph from nodes to their class.
        num_classes = F.max(y, dim=0).item() + 1
        src = graph.nodes().to(dtype=y.dtype)
        dst = y + graph.num_nodes()
        class_graph = create_graph((src, dst))
        # Add placeholder values for the class nodes.
        class_placeholder = torch.zeros(
            (num_classes), dtype=deg.dtype, device=class_graph.device
        )
        class_graph.ndata["same_class_deg"] = torch.cat(
            [graph.ndata["same_class_deg"], class_placeholder], dim=0
        )
        class_graph.update_all(
            fn.copy_u("same_class_deg", "m"), fn.sum("m", "class_deg_aggr")
        )

        # Similarly, compute \sum_{v\in C_k}|N(v)| for all k in parallel.
        class_graph.ndata["deg"] = torch.cat([deg, class_placeholder], dim=0)
        class_graph.update_all(fn.copy_u("deg", "m"), fn.sum("m", "deg_aggr"))

        # Compute class_deg_aggr / deg_aggr for all classes.
        num_nodes = graph.num_nodes()
        class_deg_aggr = class_graph.ndata["class_deg_aggr"][num_nodes:]
        deg_aggr = torch.clamp(class_graph.ndata["deg_aggr"][num_nodes:], min=1)
        fraction = (
            class_deg_aggr / deg_aggr - torch.bincount(y).float() / num_nodes
        )
        fraction = torch.clamp(fraction, min=0)

        return fraction.sum().item() / (num_classes - 1)
