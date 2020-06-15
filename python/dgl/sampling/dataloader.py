"""Data loaders"""

from functools import partial
from itertools import chain
from collections.abc import Mapping
from .. import transform
from .. import backend as F
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

class BlockSampler(object):
    """Abstract class specifying the neighborhood sampling strategy for DGL data loaders.

    The main method for BlockSampler is :func:`~dgl.sampling.BlockSampler.sample_blocks`,
    which generates a list of blocks for a multi-layer GNN given a set of seed nodes to
    have their outputs computed.

    The default implementation of :py:meth:`~dgl.sampling.BlockSampler.sample_blocks` is
    to repeat ``__len__`` times the following:

    * Obtain a message passing dependency graph (frontier) with the same nodes as the original
      graph but only the edges involved in message passing on the last layer.
      This is done with :py:meth:`~dgl.sampling.BlockSampler.sample_frontier`.

    * Postprocess the obtained frontier (e.g. by removing edges connecting training node pairs).
      One can add such postprocessors via
      :py:meth:`~dgl.sampling.BlockSampler.add_frontier_postprocessor`.

    * Convert the dependency graph into a block.

    * Postprocess the block (e.g. by assigning edge IDs).  One can add such postprocessors via
      :py:meth:`~dgl.sampling.BlockSampler.add_block_postprocessor`.

    * Prepend the block to the block list to be returned.

    All subclasses should either

    * Override :py:meth:`~dgl.sampling.BlockSampler.sample_blocks` method, or

    * Override both :py:meth:`~dgl.sampling.BlockSampler.__len__` method and
      :py:meth:`~dgl.sampling.BlockSampler.sample_frontier` method.
    """
    def __init__(self):
        self._frontier_postprocessors = []
        self._block_postprocessors = []

    @property
    def frontier_postprocessors(self):
        """List of postprocessors to be executed sequentially on the obtained frontiers.

        See also
        --------
        add_frontier_postprocessor
        """
        return self._frontier_postprocessors

    @property
    def block_postprocessors(self):
        """List of postprocessors to be executed sequentially on the obtained blocks.

        See also
        --------
        add_block_postprocessor
        """
        return self._block_postprocessors

    def add_frontier_postprocessor(self, postprocessor):
        """Add a frontier postprocessor.

        All postprocessors must have the following argument list:

        .. code::

           postprocessor(frontier, block_id, g, seed_nodes, *args, **kwargs)

        where

        * ``frontier`` represents the frontier obtained by
          :py:meth:`~dgl.sampling.BlockSampler.sample_frontier` method or returned
          by the previous postprocessor.

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
        self._frontier_postprocessors.append(postprocessor)

    def add_block_postprocessor(self, postprocessor):
        """Add a block postprocessor.

        All postprocessors must have the following argument list:

        .. code::

           postprocessor(block, frontier, block_id, g, seed_nodes, *args, **kwargs)

        where

        * ``block`` represents the block converted from the frontier or returned
          by the previous postprocessor.

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
        self._block_postprocessors.append(postprocessor)

    def _postprocess_frontier(self, frontier, block_id, g, seed_nodes, *args, **kwargs):
        """Postprocesses the generated frontier."""
        for proc in self.frontier_postprocessors:
            frontier = proc(frontier, block_id, g, seed_nodes, *args, **kwargs)
        return frontier

    def _postprocess_block(self, block, frontier, block_id, g, seed_nodes, *args, **kwargs):
        """Postprocesses the generated bloick."""
        for proc in self.block_postprocessors:
            block = proc(block, frontier, block_id, g, seed_nodes, *args, **kwargs)
        return block

    def sample_frontier(self, block_id, g, seed_nodes, *args, **kwargs):
        """
        Generate the message passing dependency graph given the output nodes.

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
        """
        raise NotImplementedError

    def sample_blocks(self, g, seed_nodes, *args, **kwargs):
        """
        Generate the computation dependency graphs as a list of blocks given the
        output nodes.

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
        """
        blocks = []
        for block_id in reversed(range(len(self))):
            frontier = self.sample_frontier(block_id, g, seed_nodes, *args, **kwargs)
            # Removing edges from message passing dependency for link prediction training falls
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

    def __len__(self):
        """Returns the number of blocks to generate (i.e. number of layers of the GNN)."""
        raise NotImplementedError

class NodeDataLoader(object):
    """
    DGL data loader for training node classification or regression on a single graph.

    This works similarly as the data loader class for each backend (e.g.
    ``torch.utils.data.DataLoader`` for PyTorch).

    Parameters
    ----------
    g : DGLHeteroGraph
        The graph.
    nids : Tensor or dict[ntype, Tensor]
        The node set to compute outputs.
    block_sampler : :ref:`BlockSampler`
        The neighborhood sampler.

    Examples
    --------
    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from all neighbors (assume
    the backend is PyTorch):
    >>> sampler = dgl.sampling.NeighborSampler([None, None, None])
    >>> dataloader = dgl.sampling.NodeDataLoader(
    ...     g, train_nid, sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for blocks in dataloader:
    ...     train_on(blocks)
    """
    def __init__(
            self,
            g,
            nids,
            block_sampler,
            **kwargs):
        self.g = g
        if not isinstance(nids, Mapping):
            assert len(g.ntypes) == 1, \
                "nids should be a dict of node type and ids for graph with multiple node types"
            self.nids = {g.ntypes[0]: nids}
        else:
            self.nids = nids
        self.nids = {
            k: utils.toindex(v, g._idtype_str).tousertensor()
            for k, v in self.nids.items()}
        self.block_sampler = block_sampler

        self.dataloaders = {
            ntype: self._get_dataloader_class(ntype)(self.nids[ntype], **kwargs)
            for ntype in self.nids.keys()}

    def _sample(self, ntype, nids):
        seed_nodes = {ntype: nids}
        return self.block_sampler.sample_blocks(self.g, seed_nodes)

    def _get_dataloader_class(self, ntype):
        backend_name = F.get_preferred_backend()
        collator = partial(self._sample, ntype)

        if backend_name == 'pytorch':
            from torch.utils.data import DataLoader
            return partial(DataLoader, collate_fn=collator)
        elif backend_name == 'mxnet':
            raise NotImplementedError(
                'NeighborSamplerDataLoader for MXNet not implemented yet')
        elif backend_name == 'tensorflow':
            raise NotImplementedError(
                'NeighborSamplerDataLoader for Tensorflow not implemented yet')
        else:
            raise NotImplementedError('Unknown backend {}'.format(backend_name))

    def __iter__(self):
        """Returns the iterator that iterates over all the nodes in batches and
        yields the list of blocks.
        """
        return chain(*self.dataloaders.values())

    def __len__(self):
        """Return the number of batches to be generated."""
        return sum(len(dl) for dl in self.dataloaders.values())
