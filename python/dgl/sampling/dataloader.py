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

class BlockSampler(object):
    """Abstract class specifying the neighborhood sampling strategy for DGL data loaders.

    The main method for BlockSampler is :func:`~dgl.sampling.BlockSampler.sample_blocks`,
    which generates a list of blocks for a multi-layer GNN given a set of seed nodes to
    have their outputs computed.

    The default implementation of :py:meth:`~dgl.sampling.BlockSampler.sample_blocks` is
    to repeat ``__len__`` times the following:

    * Obtain a frontier with the same nodes as the original graph but only the edges
      involved in message passing on the last layer.
      This is done with :py:meth:`~dgl.sampling.BlockSampler.sample_frontier`.

    * Postprocess the obtained frontier (e.g. by removing edges connecting training node pairs).
      One can add such postprocessors via
      :py:meth:`~dgl.sampling.BlockSampler.add_frontier_postprocessor`.

    * Convert the frontier into a block.

    * Postprocess the block (e.g. by assigning edge IDs).  One can add such postprocessors via
      :py:meth:`~dgl.sampling.BlockSampler.add_block_postprocessor`.

    * Prepend the block to the block list to be returned.

    All subclasses should either

    * Override :py:meth:`~dgl.sampling.BlockSampler.sample_blocks` method, or

    * Override both :py:meth:`~dgl.sampling.BlockSampler.__len__` method and
      :py:meth:`~dgl.sampling.BlockSampler.sample_frontier` method.

    See also
    --------
    For the concept of frontiers and blocks, please refer to User Guide Section 6.
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
        for block_id in reversed(range(len(self))):
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

    def __len__(self):
        """Returns the number of blocks to generate (i.e. number of layers of the GNN)."""
        raise NotImplementedError

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

        See also
        --------
        For the concept of blocks, please refer to User Guide Section 6.
        """
        raise NotImplementedError

class NodeCollator(Collator):
    """
    DGL collator to combine training node classification or regression on a single graph.

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
    >>> collator = dgl.sampling.NodeCollator(g, train_nid, sampler)
    >>> dataloader = torch.utils.data.DataLoader(
    ...     collator.dataset, collate_fn=collator.collate,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for blocks in dataloader:
    ...     train_on(blocks)
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
        if isinstance(items[0], tuple):
            # returns a list of pairs: group them by node types into a dict
            items = utils.group_as_dict(items)
        return self.block_sampler.sample_blocks(self.g, items)
