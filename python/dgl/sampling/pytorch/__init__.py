"""DGL PyTorch DataLoaders"""
from torch.utils.data import DataLoader
from ..dataloader import NodeCollator, EdgeCollator

class NodeDataLoader(DataLoader):
    """PyTorch dataloader for batch-iterating over a set of nodes, generating the list
    of blocks as computation dependency of the said minibatch.

    Parameters
    ----------
    g : DGLHeteroGraph
        The graph.
    nids : Tensor or dict[ntype, Tensor]
        The node set to compute outputs.
    block_sampler : :py:class:`~dgl.sampling.BlockSampler`
        The neighborhood sampler.
    kwargs : dict
        Arguments being passed to `torch.utils.data.DataLoader`.

    Examples
    --------
    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from all neighbors (assume
    the backend is PyTorch):
    >>> sampler = dgl.sampling.NeighborSampler([None, None, None])
    >>> dataloader = dgl.sampling.NodeDataLoader(
    ...     g, train_nid, sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, output_nodes, blocks in dataloader:
    ...     train_on(input_nodes, output_nodes, blocks)
    """
    def __init__(self, g, nids, block_sampler, **kwargs):
        self.collator = NodeCollator(g, nids, block_sampler)
        super().__init__(self.collator.dataset, collate_fn=self.collator.collate, **kwargs)

class EdgeDataLoader(DataLoader):
    """PyTorch dataloader for batch-iterating over a set of edges, generating the list
    of blocks as computation dependency of the said minibatch for edge classification,
    edge regression, and link prediction.

    Parameters
    ----------
    g : DGLHeteroGraph
        The graph.
    nids : Tensor or dict[ntype, Tensor]
        The node set to compute outputs.
    block_sampler : :py:class:`~dgl.sampling.BlockSampler`
        The neighborhood sampler.
    exclude : str, optional
        Whether and how to exclude dependencies related to the sampled edges in the
        minibatch.  Possible values are

        * None,
        * ``reverse``,
        * ``reverse_types``

        See the docstring in :py:class:`~dgl.sampling.EdgeCollator`.
    reverse_edge_ids : Tensor or dict[etype, Tensor], optional
        See the docstring in :py:class:`~dgl.sampling.EdgeCollator`.
    reverse_etypes : dict[etype, etype], optional
        See the docstring in :py:class:`~dgl.sampling.EdgeCollator`.
    negative_sampler : callable, optional
        The negative sampler.

        See the docstring in :py:class:`~dgl.sampling.EdgeCollator`.
    kwargs : dict
        Arguments being passed to `torch.utils.data.DataLoader`.

    Examples
    --------
    The following example shows how to train a 3-layer GNN for edge classification on a
    set of edges ``train_eid`` on a homogeneous undirected graph.  Each node takes
    messages from all neighbors.  We first make ``g`` bidirectional by adding reverse
    edges:
    >>> g = dgl.add_reverse(g)

    Then we create a ``DataLoader`` for iterating.  Note that the sampled edges as well
    as their reverse edges are removed from computation dependencies of the incident nodes.
    This is a common trick to avoid information leakage.
    >>> sampler = dgl.sampling.NeighborSampler([None, None, None])
    >>> dataloader = dgl.sampling.EdgeDataLoader(
    ...     g, train_eid, sampler, exclude='reverse',
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, pair_graph, blocks in dataloader:
    ...     train_on(input_nodes, pair_graph, blocks)

    To train a 3-layer GNN for link prediction on a set of edges ``train_eid`` on a
    homogeneous graph where each node takes messages from all neighbors (assume the
    backend is PyTorch), with 5 uniformly chosen negative samples per edge:
    >>> sampler = dgl.sampling.NeighborSampler([None, None, None])
    >>> neg_sampler = dgl.sampling.negative_sampler.Uniform(5)
    >>> dataloader = dgl.sampling.EdgeDataLoader(
    ...     g, train_eid, sampler, exclude='reverse', negative_sampler=neg_sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, pos_pair_graph, neg_pair_graph, blocks in dataloader:
    ...     train_on(input_nodse, pair_graph, neg_pair_graph, blocks)

    See also
    --------
    :py:class:`~dgl.sampling.EdgeCollator`
    """
    pass
