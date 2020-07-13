"""Data loaders"""

from collections.abc import Mapping
from abc import ABC, abstractproperty, abstractmethod
from .. import transform
from ..base import NID, EID
from .. import utils

# pylint: disable=unused-argument
def assign_block_eids(block, frontier, block_id, g, seed_nodes, *args, **kwargs):
    """Assigns edge IDs from the original graph to the block.

    This is the default block postprocessor for samplers created with
    ``return_eids`` as True.

    See also
    --------
    BlockSampler
    MultiLayerNeighborSampler
    """
    for etype in block.canonical_etypes:
        block.edges[etype].data[EID] = frontier.edges[etype].data[EID][
            block.edges[etype].data[EID]]
    return block

def _default_frontier_postprocessor(frontier, block_id, g, seed_nodes, *args, **kwargs):
    return frontier

def _default_block_postprocessor(block, frontier, block_id, g, seed_nodes, *args, **kwargs):
    return block

class BlockSampler(object):
    """Abstract class specifying the neighborhood sampling strategy for DGL data loaders.

    The main method for BlockSampler is :func:`~dgl.sampling.BlockSampler.sample_blocks`,
    which generates a list of blocks for a multi-layer GNN given a set of seed nodes to
    have their outputs computed.

    The default implementation of :py:meth:`~dgl.sampling.BlockSampler.sample_blocks` is
    to repeat ``num_hops`` times the following:

    * Obtain a frontier with the same nodes as the original graph but only the edges
      involved in message passing on the last layer.
      Customizable via :py:meth:`~dgl.sampling.BlockSampler.sample_frontier`.

    * Optionally, post-process the obtained frontier (e.g. by removing edges connecting training
      node pairs).  One can add such postprocessors via
      :py:meth:`~dgl.sampling.BlockSampler.add_frontier_postprocessor`.

    * Convert the frontier into a block.

    * Optionally, post-process the block (e.g. by assigning edge IDs).  One can add such
      postprocessors via
      :py:meth:`~dgl.sampling.BlockSampler.add_block_postprocessor`.

    * Prepend the block to the block list to be returned.

    All subclasses should either

    * Override :py:meth:`~dgl.sampling.BlockSampler.sample_blocks` method, or

    * Override
      :py:meth:`~dgl.sampling.BlockSampler.sample_frontier` method while specifying
      the number of layers to sample in ``num_hops`` argument.

    See also
    --------
    For the concept of frontiers and blocks, please refer to User Guide Section 6.
    """
    def __init__(self, num_hops):
        self.num_hops = num_hops
        self._frontier_postprocessor = _default_frontier_postprocessor
        self._block_postprocessor = _default_block_postprocessor

    @property
    def frontier_postprocessor(self):
        """Frontier postprocessor."""
        return self._frontier_postprocessor

    @property
    def block_postprocessor(self):
        """B;pcl postprocessor."""
        return self._block_postprocessor

    def set_frontier_postprocessor(self, postprocessor):
        """Set a frontier postprocessor.

        The postprocessor must have the following signature:

        .. code::

           postprocessor(frontier, block_id, g, seed_nodes, *args, **kwargs)

        where

        * ``frontier`` represents the frontier obtained by
          :py:meth:`~dgl.sampling.BlockSampler.sample_frontier` method.

        * ``block_id`` represents which GNN layer the block is currently generated for.

        * ``g`` represents the original graph.

        * ``seed_nodes`` represents the output nodes on the current layer.

        * Other arguments are the same ones passed into
          :py:meth:`~dgl.sampling.BlockSampler.sample_blocks` method.

        Parameters
        ----------
        postprocessor : callable
            The postprocessor.
        """
        self._frontier_postprocessor = postprocessor

    def set_block_postprocessor(self, postprocessor):
        """Set a block postprocessor.

        The postprocessor must have the following signature:

        .. code::

           postprocessor(block, frontier, block_id, g, seed_nodes, *args, **kwargs)

        where

        * ``block`` represents the block converted from the frontier.

        * ``frontier`` represents the frontier the block is generated from.

        * ``block_id`` represents which GNN layer the block is currently generated for.

        * ``g`` represents the original graph.

        * ``seed_nodes`` represents the output nodes on the current layer.

        * Other arguments are the same ones passed into
          :py:meth:`~dgl.sampling.BlockSampler.sample_blocks` method.

        Parameters
        ----------
        postprocessor : callable
            The postprocessor.
        """
        self._block_postprocessor = postprocessor

    def _postprocess_frontier(self, frontier, block_id, g, seed_nodes, *args, **kwargs):
        """Post-processes the generated frontier."""
        return self._frontier_postprocessor(
            frontier, block_id, g, seed_nodes, *args, **kwargs)

    def _postprocess_block(self, block, frontier, block_id, g, seed_nodes, *args, **kwargs):
        """Post-processes the generated block."""
        return self._block_postprocessor(
            block, frontier, block_id, g, seed_nodes, *args, **kwargs)

    def sample_frontier(self, block_id, g, seed_nodes, *args, **kwargs):
        """
        Generate the frontier given the output nodes.

        Parameters
        ----------
        block_id : int
            Represents which GNN layer the frontier is generated for.
        g : DGLHeteroGraph
            The original graph.
        seed_nodes : Tensor or dict[ntype, Tensor]
            The output nodes by node type.

            If the graph only has one node type, one can just specify a single tensor
            of node IDs.
        args, kwargs :
            Other arguments being passed by
            :py:meth:`~dgl.sampling.BlockSampler.sample_blocks`.

        Returns
        -------
        DGLHeteroGraph
            The frontier generated for the current layer.

        See also
        --------
        For the concept of frontiers and blocks, please refer to User Guide Section 6.
        """
        raise NotImplementedError

    def sample_blocks(self, g, seed_nodes, *args, **kwargs):
        """
        Generate the a list of blocks given the output nodes.

        Parameters
        ----------
        g : DGLHeteroGraph
            The original graph.
        seed_nodes : Tensor or dict[ntype, Tensor]
            The output nodes by node type.

            If the graph only has one node type, one can just specify a single tensor
            of node IDs.
        args, kwargs :
            Other arguments being passed by
            :py:meth:`~dgl.sampling.BlockSampler.sample_blocks`.

        Returns
        -------
        list[DGLHeteroGraph]
            The blocks generated for computing the multi-layer GNN output.

        See also
        --------
        For the concept of frontiers and blocks, please refer to User Guide Section 6.
        """
        blocks = []
        for block_id in reversed(range(self.num_hops)):
            frontier = self.sample_frontier(block_id, g, seed_nodes, *args, **kwargs)
            # Removing edges from the frontier for link prediction training falls
            # into the category of frontier postprocessing
            frontier = self._postprocess_frontier(
                frontier, block_id, g, seed_nodes, *args, **kwargs)

            block = transform.to_block(frontier, seed_nodes)
            # Assigning edge IDs and/or node/edge features falls into the category of block
            # postprocessing
            block = self._postprocess_block(
                block, frontier, block_id, g, seed_nodes, *args, **kwargs)

            seed_nodes = {ntype: block.srcnodes[ntype].data[NID] for ntype in block.srctypes}
            blocks.insert(0, block)
        return blocks

class Collator(ABC):
    """
    Abstract DGL collator for training GNNs on downstream tasks stochastically.

    Provides a ``dataset`` object containing the collection of all nodes or edges,
    as well as a ``collate`` method that combines a set of items from ``dataset`` and
    obtains the blocks.

    See also
    --------
    For the concept of blocks, please refer to User Guide Section 6.
    """
    @abstractproperty
    def dataset(self):
        """Returns the dataset object of the collator."""
        raise NotImplementedError

    @abstractmethod
    def collate(self, items):
        """Combines the items from the dataset object and obtains the list of blocks.

        Parameters
        ----------
        items : list[str, int]
            The list of node or edge type-ID pairs.

        See also
        --------
        For the concept of blocks, please refer to User Guide Section 6.
        """
        raise NotImplementedError

class NodeCollator(Collator):
    """
    DGL collator to combine nodes and their computation dependencies within a minibatch for
    training node classification or regression on a single graph with neighborhood sampling.

    Parameters
    ----------
    g : DGLHeteroGraph
        The graph.
    nids : Tensor or dict[ntype, Tensor]
        The node set to compute outputs.
    block_sampler : :py:class:`~dgl.sampling.BlockSampler`
        The neighborhood sampler.

    Examples
    --------
    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from all neighbors (assume
    the backend is PyTorch):
    >>> sampler = dgl.sampling.NeighborSampler([None, None, None])
    >>> collator = dgl.sampling.NodeCollator(g, train_nid, sampler)
    >>> dataloader = torch.utils.data.DataLoader(
    ...     collator.dataset, collate_fn=collator.collate,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, output_nodes, blocks in dataloader:
    ...     train_on(input_nodes, output_nodes, blocks)
    """
    def __init__(self, g, nids, block_sampler):
        self.g = g
        if not isinstance(nids, Mapping):
            assert len(g.ntypes) == 1, \
                "nids should be a dict of node type and ids for graph with multiple node types"
        self.nids = nids
        self.block_sampler = block_sampler

        if isinstance(nids, Mapping):
            self._dataset = utils.FlattenedDict(nids)
        else:
            self._dataset = nids

    @property
    def dataset(self):
        return self._dataset

    def collate(self, items):
        """Find the list of blocks necessary for computing the representation of given
        nodes for a node classification/regression task.

        Returns
        -------
        input_nodes : Tensor or dict[ntype, Tensor]
            The input nodes necessary for computation in this minibatch.

            If the original graph has multiple node types, return a dictionary of
            node type names and node ID tensors.  Otherwise, return a single tensor.
        output_nodes : Tensor or dict[ntype, Tensor]
            The nodes whose representations are to be computed in this minibatch.

            If the original graph has multiple node types, return a dictionary of
            node type names and node ID tensors.  Otherwise, return a single tensor.
        blocks : list[DGLHeteroGraph]
            The list of blocks necessary for computing the representation.
        """
        if isinstance(items[0], tuple):
            # returns a list of pairs: group them by node types into a dict
            items = utils.group_as_dict(items)
        blocks = self.block_sampler.sample_blocks(self.g, items)

        if len(self.g.ntypes) == 1:
            output_nodes = blocks[-1].dstdata[NID]
            input_nodes = blocks[0].srcdata[NID]
        else:
            output_nodes = {
                ntype: blocks[-1].dstnodes[ntype].data[NID]
                for ntype in blocks[-1].dsttypes}
            input_nodes = {
                ntype: blocks[0].srcnodes[ntype].data[NID]
                for ntype in blocks[0].srctypes}

        return input_nodes, output_nodes, blocks

class EdgeCollator(Collator):
    """
    DGL collator to combine edges and their computation dependencies within a minibatch for
    training edge classification, edge regression, or link prediction on a single graph
    with neighborhood sampling.

    Given a set of edges, the collate function will yield

    * A tensor of input nodes necessary for computing the representation on edges, or
      a dictionary of node type names and such tensors.

    * A graph that contains only the edges in the minibatch and their incident nodes.
      Note that the graph has an identical metagraph with the original graph.

    * If a negative sampler is given, another graph that contains the "negative edges",
      connecting the source and destination nodes yielded from the given negative sampler.

    * A list of blocks necessary for computing the representation of the incident nodes
      of the edges in the minibatch.

    Parameters
    ----------
    g : DGLHeteroGraph
        The graph.
    eids : Tensor or dict[etype, Tensor]
        The edge set to compute outputs.
    block_sampler : :py:class:`~dgl.sampling.BlockSampler`
        The neighborhood sampler.
    exclude : str, optional
        Whether and how to exclude dependencies related to the sampled edges in the
        minibatch.  Possible values are

        * None, which excludes nothing.

        * ``'reverse'``, which excludes the reverse edges of the sampled edges.  The said
          reverse edges have the same edge type as the sampled edges.  Only works
          on edge types whose source node type is the same as its destination node type
          (see also :py:func:`dgl.transform.add_reverse`).

        * ``'reverse_types'``, which excludes the reverse edges of the sampled edges.  The
          said reverse edges have different edge types from the sampled edges (see also
          :py:func:`dgl.transform.add_reverse_types`).
    reverse_edge_ids : Tensor or dict[etype, Tensor], optional
        The mapping from original edge ID to its reverse edge ID.

        Only used when ``exclude`` is set to ``reverse``.

        For heterogeneous graph this will be a dict of edge type and edge IDs.  Note that
        only the edge types whose source node type is the same as destination node type
        are needed.

        If not given, assumes that the reverse edges are added by the function
        :py:func:`dgl.transform.add_reverse`.
    reverse_etypes : dict[etype, etype], optional
        The mapping from the edge type to its reverse edge type.

        Only used when ``exclude`` is set to ``reverse_types``.

        If not given, assumes that the reverse edge type name is the original edge
        type name suffixed by ``'_inv'``, same as the default behavior of
        :py:func:`dgl.transform.add_reverse_types`.
    negative_sampler : callable, optional
        The negative sampler.  Can be omitted if no negative sampling is needed.

        The negative sampler must be a callable that takes in the following arguments:

        * The original (heterogeneous) graph.

        * The ID array of sampled edges in the minibatch, or the dictionary of edge
          types and ID array of sampled edges in the minibatch if the graph is
          heterogeneous.

        It should return

        * A pair of source and destination node ID arrays as negative samples,
          or a dictionary of edge types and such pairs if the graph is heterogenenous.

        A set of builtin negative samplers are provided in
        :py:mod:`dgl.sampling.negative_sampler`.

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
    >>> collator = dgl.sampling.EdgeCollator(g, train_eid, sampler, exclude='reverse')
    >>> dataloader = torch.utils.data.DataLoader(
    ...     collator.dataset, collate_fn=collator.collate,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, pair_graph, blocks in dataloader:
    ...     train_on(input_nodes, pair_graph, blocks)

    To train a 3-layer GNN for link prediction on a set of edges ``train_eid`` on a
    homogeneous graph where each node takes messages from all neighbors (assume the
    backend is PyTorch), with 5 uniformly chosen negative samples per edge:
    >>> sampler = dgl.sampling.NeighborSampler([None, None, None])
    >>> neg_sampler = dgl.sampling.negative_sampler.Uniform(5)
    >>> collator = dgl.sampling.EdgeCollator(
    ...     g, train_eid, sampler, exclude='reverse', negative_sampler=neg_sampler)
    >>> dataloader = torch.utils.data.DataLoader(
    ...     collator.dataset, collate_fn=collator.collate,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, pos_pair_graph, neg_pair_graph, blocks in dataloader:
    ...     train_on(input_nodse, pair_graph, neg_pair_graph, blocks)

    See also
    --------
    Please refer to the following tutorial/examples:

    * Edge classification on heterogeneous graph: GCMC

    * Link prediction on homogeneous graph: GraphSAGE for unsupervised learning

    * Link prediction on heterogeneous graph: RGCN for link prediction.
    """
    def __init__(self, g, eids, block_sampler, exclude=None, reverse_edge_ids=None,
                 reverse_etypes=None, negative_sampler=None):
        self.g = g
        if not isinstance(eids, Mapping):
            assert len(g.etypes) == 1, \
                "eids should be a dict of edge type and ids for graph with multiple edge types"
        self.eids = eids
        self.block_sampler = block_sampler
        self.exclude = exclude
        self.reverse_edge_ids = reverse_edge_ids
        self.reverse_etypes = reverse_etypes
        self.negative_sampler = negative_sampler

        if isinstance(eids, Mapping):
            self._dataset = utils.FlattnedDict(nids)
        else:
            self._dataset = nids

    @property
    def dataset(self):
        return self._dataset

    def collate(self, items):
        """Combines the sampled edges into a minibatch for edge classification, edge
        regression, and link prediction tasks.

        Returns
        -------
        Either ``(input_nodes, pair_graph, blocks)``, or
        ``(input_nodes, pair_graph, negative_pair_graph, blocks)`` if negative sampling is
        enabled.

        input_nodes : Tensor or dict[ntype, Tensor]
            The input nodes necessary for computation in this minibatch.

            If the original graph has multiple node types, return a dictionary of
            node type names and node ID tensors.  Otherwise, return a single tensor.
        pair_graph : DGLHeteroGraph
            The graph that contains only the edges in the minibatch as well as their incident
            nodes.

            Note that the metagraph of this graph will be identical to that of the original
            graph.
        negative_pair_graph : DGLHeteroGraph
            The graph that contains only the edges connecting the source and destination nodes
            yielded from the given negative sampler, if negative sampling is enabled.

            Note that the metagraph of this graph will be identical to that of the original
            graph.
        blocks : list[DGLHeteroGraph]
            The list of blocks necessary for computing the representation of the edges.
        """
        if isinstance(items[0], tuple):
            items = utils.group_as_dict(items)

        pair_graph = self.g.edge_subgraph(items)

        if self.negative_sampler is not None:
            neg_edges = self.negative_sampler(items)
            if isinstance(neg_edges, Mapping):
                neg_edges = {self.g.to_canonical_etype(k): v for k, v in neg_edges.items()}
                neg_graph = heterograph(
                    neg_edges, {ntype: self.g.number_of_ndoes(ntype) for ntype in self.g.ntypes})
