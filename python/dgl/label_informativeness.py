"""Utils for computing graph label informativeness"""
from . import to_bidirected

try:
    import torch
except ImportError:
    HAS_TORCH = False
else:
    HAS_TORCH = True

__all__ = ["edge_label_informativeness", "node_label_informativeness"]


def check_pytorch():
    """Check if PyTorch is the backend."""
    if HAS_TORCH is False:
        raise ModuleNotFoundError(
            "This function requires PyTorch to be the backend."
        )


def edge_label_informativeness(graph, y, eps=1e-8):
    r"""Label informativeness (:math:`\mathrm{LI}`) is a characteristic of
    labeled graphs proposed in the `Characterizing Graph Datasets for Node
    Classification: Homophily-Heterophily Dichotomy and Beyond
    <https://arxiv.org/abs/2209.06177>`__

    Label informativeness shows how much information about a node's label we
    get from knowing its neighbor's label. Formally, assume that we sample an
    edge :math:`(\xi,\eta) \in E`. The class labels of nodes :math:`\xi` and
    :math:`\eta` are then random variables :math:`y_\xi` and :math:`y_\eta`.
    We want to measure the amount of knowledge the label :math:`y_\eta` gives
    for predicting :math:`y_\xi`. The entropy :math:`H(y_\xi)` measures the
    `hardness' of predicting the label of :math:`\xi` without knowing
    :math:`y_\eta`. Given :math:`y_\eta`, this value is reduced to the
    conditional entropy :math:`H(y_\xi|y_\eta)`. In other words, :math:`y_\eta`
    reveals :math:`I(y_\xi,y_\eta) = H(y_\xi) - H(y_\xi|y_\eta)` information
    about the label. To make the obtained quantity comparable across different
    datasets, label informativeness is defined as the normalized mutual
    information of :math:`y_{\xi}` and :math:`y_{\eta}`:

    .. math::
      \mathrm{LI} = \frac{I(y_\xi,y_\eta)}{H(y_\xi)}

    Depending on the distribution used for sampling an edge
    :math:`(\xi, \eta)`, several variants of label informativeness can be
    obtained. Two of them are particularly intuitive: in edge label
    informativeness (:math:`\mathrm{LI}_{edge}`), edges are sampled uniformly
    at random, and in node label informativeness (:math:`\mathrm{LI}_{node}`),
    first a node is sampled uniformly at random and then an edge incident to it
    is sampled uniformly at random. These two versions of label informativeness
    differ in how they weight high/low-degree nodes. In edge label
    informativeness, averaging is over the edges, thus high-degree nodes are
    given more weight. In node label informativeness, averaging is over the
    nodes, so all nodes are weighted equally.

    This function computes edge label informativeness.

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    y : torch.Tensor
        The node labels, which is a tensor of shape (|V|).
    eps : float, optional
        A small constant for numerical stability. (default: 1e-8)

    Returns
    -------
    float
        The edge label informativeness value.

    Examples
    --------
    >>> import dgl
    >>> import torch

    >>> graph = dgl.graph(([0, 1, 2, 2, 3, 4], [1, 2, 0, 3, 4, 5]))
    >>> y = torch.tensor([0, 0, 0, 0, 1, 1])
    >>> dgl.edge_label_informativeness(graph, y)
    0.25177597999572754
    """
    check_pytorch()

    graph = to_bidirected(graph.cpu()).to(y.device)

    degrees = graph.in_degrees().float()
    num_classes = y.max() + 1
    class_degree_weighted_probs = torch.zeros(num_classes).to(y.device)
    class_degree_weighted_probs.index_add_(dim=0, index=y, source=degrees)
    class_degree_weighted_probs /= class_degree_weighted_probs.sum()

    edge_probs = torch.zeros(num_classes, num_classes).to(y.device)
    labels_u = y[graph.edges()[0].long()]
    labels_v = y[graph.edges()[1].long()]
    edge_probs.index_put_(
        indices=(labels_u, labels_v),
        values=torch.ones(graph.num_edges()).to(y.device),
        accumulate=True,
    )
    edge_probs /= edge_probs.sum()
    edge_probs += eps

    numerator = (edge_probs * torch.log(edge_probs)).sum()
    denominator = (
        class_degree_weighted_probs * torch.log(class_degree_weighted_probs)
    ).sum()
    li_edge = 2 - numerator / denominator

    return li_edge.item()


def node_label_informativeness(graph, y, eps=1e-8):
    r"""Label informativeness (:math:`\mathrm{LI}`) is a characteristic of
    labeled graphs proposed in the `Characterizing Graph Datasets for Node
    Classification: Homophily-Heterophily Dichotomy and Beyond
    <https://arxiv.org/abs/2209.06177>`__

    Label informativeness shows how much information about a node's label we
    get from knowing its neighbor's label. Formally, assume that we sample an
    edge :math:`(\xi,\eta) \in E`. The class labels of nodes :math:`\xi` and
    :math:`\eta` are then random variables :math:`y_\xi` and :math:`y_\eta`.
    We want to measure the amount of knowledge the label :math:`y_\eta` gives
    for predicting :math:`y_\xi`. The entropy :math:`H(y_\xi)` measures the
    `hardness' of predicting the label of :math:`\xi` without knowing
    :math:`y_\eta`. Given :math:`y_\eta`, this value is reduced to the
    conditional entropy :math:`H(y_\xi|y_\eta)`. In other words, :math:`y_\eta`
    reveals :math:`I(y_\xi,y_\eta) = H(y_\xi) - H(y_\xi|y_\eta)` information
    about the label. To make the obtained quantity comparable across different
    datasets, label informativeness is defined as the normalized mutual
    information of :math:`y_{\xi}` and :math:`y_{\eta}`:

    .. math::
      \mathrm{LI} = \frac{I(y_\xi,y_\eta)}{H(y_\xi)}

    Depending on the distribution used for sampling an edge
    :math:`(\xi, \eta)`, several variants of label informativeness can be
    obtained. Two of them are particularly intuitive: in edge label
    informativeness (:math:`\mathrm{LI}_{edge}`), edges are sampled uniformly
    at random, and in node label informativeness (:math:`\mathrm{LI}_{node}`),
    first a node is sampled uniformly at random and then an edge incident to it
    is sampled uniformly at random. These two versions of label informativeness
    differ in how they weight high/low-degree nodes. In edge label
    informativeness, averaging is over the edges, thus high-degree nodes are
    given more weight. In node label informativeness, averaging is over the
    nodes, so all nodes are weighted equally.

    This function computes node label informativeness.

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    y : torch.Tensor
        The node labels, which is a tensor of shape (|V|).
    eps : float, optional
        A small constant for numerical stability. (default: 1e-8)

    Returns
    -------
    float
        The node label informativeness value.

    Examples
    --------
    >>> import dgl
    >>> import torch

    >>> graph = dgl.graph(([0, 1, 2, 2, 3, 4], [1, 2, 0, 3, 4, 5]))
    >>> y = torch.tensor([0, 0, 0, 0, 1, 1])
    >>> dgl.node_label_informativeness(graph, y)
    0.3381872773170471
    """
    check_pytorch()

    graph = to_bidirected(graph.cpu()).to(y.device)

    degrees = graph.in_degrees().float()
    num_classes = y.max() + 1

    class_probs = torch.zeros(num_classes).to(y.device)
    class_probs.index_add_(
        dim=0, index=y, source=torch.ones(graph.num_nodes()).to(y.device)
    )
    class_probs /= class_probs.sum()

    class_degree_weighted_probs = torch.zeros(num_classes).to(y.device)
    class_degree_weighted_probs.index_add_(dim=0, index=y, source=degrees)
    class_degree_weighted_probs /= class_degree_weighted_probs.sum()

    num_nonzero_degree_nodes = (degrees > 0).sum()

    edge_probs = torch.zeros(num_classes, num_classes).to(y.device)
    labels_u = y[graph.edges()[0].long()]
    labels_v = y[graph.edges()[1].long()]
    degrees_u = degrees[graph.edges()[0].long()]
    edge_probs.index_put_(
        indices=(labels_u, labels_v),
        values=1 / (num_nonzero_degree_nodes * degrees_u),
        accumulate=True,
    )
    edge_probs += eps

    log = torch.log(
        edge_probs
        / (class_probs[:, None] * class_degree_weighted_probs[None, :])
    )
    numerator = (edge_probs * log).sum()
    denominator = (class_probs * torch.log(class_probs)).sum()
    li_node = -numerator / denominator

    return li_node.item()
