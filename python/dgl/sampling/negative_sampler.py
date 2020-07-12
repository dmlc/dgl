"""Negative samplers"""

class Uniform(object):
    """Uniform negative sampler.

    For each edge with type `(utype, etype, vtype)`, ``k`` pairs of random nodes
    with node type ``utype`` and ``vtype`` will be returned.  The nodes will be
    chosen uniformly.

    Parameters
    ----------
    k : int
        The number of negative examples.

    Examples
    --------
    >>> g = dgl.graph(([0, 1, 2], [1, 2, 3]))
    >>> neg_sampler = dgl.sampling.negative_sampler.Uniform(2)
    >>> neg_sampler(g, [0, 1])
    (tensor([2, 3, 2, 1]), tensor([1, 0, 2, 3]))
    """
    def __init__(self, k):
        pass

    def __call__(self, g, eid):
        """Returns negative examples.

        Parameters
        ----------
        g : DGLHeteroGraph
            The graph.
        eid : Tensor or dict[etype, Tensor]
            The sampled edges in the minibatch.

        Returns
        -------
        tuple[Tensor, Tensor] or dict[etype, tuple[Tensor, Tensor]]
            The returned source-destination pairs as negative examples.
        """
        pass

class UniformBySource(object):
    """Uniform negative sampler that randomly chooses negative destination nodes
    for each source node.

    For each edge with type `(utype, etype, vtype)`, ``k`` pairs of nodes
    with node type ``utype`` and ``vtype`` will be returned.  The source nodes
    will always be the source node of the edge, while the destination nodes
    are chosen uniformly.

    Parameters
    ----------
    k : int
        The number of negative examples.

    Examples
    --------
    >>> g = dgl.graph(([0, 1, 2], [1, 2, 3]))
    >>> neg_sampler = dgl.sampling.negative_sampler.Uniform(2)
    >>> neg_sampler(g, [0, 1])
    (tensor([0, 0, 1, 1]), tensor([1, 0, 2, 3]))
    """
    def __init__(self, k):
        pass

    def __call__(self, g, eid):
        """Returns negative examples.

        Parameters
        ----------
        g : DGLHeteroGraph
            The graph.
        eid : Tensor or dict[etype, Tensor]
            The sampled edges in the minibatch.

        Returns
        -------
        tuple[Tensor, Tensor] or dict[etype, tuple[Tensor, Tensor]]
            The returned source-destination pairs as negative examples.
        """
        pass
