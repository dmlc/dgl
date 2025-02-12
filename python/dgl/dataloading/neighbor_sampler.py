"""Data loading components for neighbor sampling"""
from functools import cache
from .. import backend as F
from ..base import EID, NID
from ..heterograph import DGLGraph
from ..transforms import to_block
from ..utils import get_num_threads
from .base import BlockSampler

import torch
import dgl


class NeighborSampler(BlockSampler):
    """Sampler that builds computational dependency of node representations via
    neighbor sampling for multilayer GNN.

    This sampler will make every node gather messages from a fixed number of neighbors
    per edge type.  The neighbors are picked uniformly.

    Parameters
    ----------
    fanouts : list[int] or list[dict[etype, int]]
        List of neighbors to sample per edge type for each GNN layer, with the i-th
        element being the fanout for the i-th GNN layer.

        If only a single integer is provided, DGL assumes that every edge type
        will have the same fanout.

        If -1 is provided for one edge type on one layer, then all inbound edges
        of that edge type will be included.
    edge_dir : str, default ``'in'``
        Can be either ``'in' `` where the neighbors will be sampled according to
        incoming edges, or ``'out'`` otherwise, same as :func:`dgl.sampling.sample_neighbors`.
    prob : str, optional
        If given, the probability of each neighbor being sampled is proportional
        to the edge feature value with the given name in ``g.edata``.  The feature must be
        a scalar on each edge.

        This argument is mutually exclusive with :attr:`mask`.  If you want to
        specify both a mask and a probability, consider multiplying the probability
        with the mask instead.
    mask : str, optional
        If given, a neighbor could be picked only if the edge mask with the given
        name in ``g.edata`` is True.  The data must be boolean on each edge.

        This argument is mutually exclusive with :attr:`prob`.  If you want to
        specify both a mask and a probability, consider multiplying the probability
        with the mask instead.
    replace : bool, default False
        Whether to sample with replacement
    prefetch_node_feats : list[str] or dict[ntype, list[str]], optional
        The source node data to prefetch for the first MFG, corresponding to the
        input node features necessary for the first GNN layer.
    prefetch_labels : list[str] or dict[ntype, list[str]], optional
        The destination node data to prefetch for the last MFG, corresponding to
        the node labels of the minibatch.
    prefetch_edge_feats : list[str] or dict[etype, list[str]], optional
        The edge data names to prefetch for all the MFGs, corresponding to the
        edge features necessary for all GNN layers.
    output_device : device, optional
        The device of the output subgraphs or MFGs.  Default is the same as the
        minibatch of seed nodes.
    fused : bool, default True
        If True and device is CPU fused sample neighbors is invoked. This version
        requires seed_nodes to be unique

    Examples
    --------
    **Node classification**

    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from 5, 10, 15 neighbors for
    the first, second, and third layer respectively (assuming the backend is PyTorch):

    >>> sampler = dgl.dataloading.NeighborSampler([5, 10, 15])
    >>> dataloader = dgl.dataloading.DataLoader(
    ...     g, train_nid, sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, output_nodes, blocks in dataloader:
    ...     train_on(blocks)

    If training on a heterogeneous graph and you want different number of neighbors for each
    edge type, one should instead provide a list of dicts.  Each dict would specify the
    number of neighbors to pick per edge type.

    >>> sampler = dgl.dataloading.NeighborSampler([
    ...     {('user', 'follows', 'user'): 5,
    ...      ('user', 'plays', 'game'): 4,
    ...      ('game', 'played-by', 'user'): 3}] * 3)

    If you would like non-uniform neighbor sampling:

    >>> g.edata['p'] = torch.rand(g.num_edges())   # any non-negative 1D vector works
    >>> sampler = dgl.dataloading.NeighborSampler([5, 10, 15], prob='p')

    Or sampling on edge masks:

    >>> g.edata['mask'] = torch.rand(g.num_edges()) < 0.2   # any 1D boolean mask works
    >>> sampler = dgl.dataloading.NeighborSampler([5, 10, 15], prob='mask')

    **Edge classification and link prediction**

    This class can also work for edge classification and link prediction together
    with :func:`as_edge_prediction_sampler`.

    >>> sampler = dgl.dataloading.NeighborSampler([5, 10, 15])
    >>> sampler = dgl.dataloading.as_edge_prediction_sampler(sampler)
    >>> dataloader = dgl.dataloading.DataLoader(
    ...     g, train_eid, sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)

    See the documentation :func:`as_edge_prediction_sampler` for more details.

    Notes
    -----
    For the concept of MFGs, please refer to
    :ref:`User Guide Section 6 <guide-minibatch>` and
    :doc:`Minibatch Training Tutorials <tutorials/large/L0_neighbor_sampling_overview>`.
    """

    def __init__(
        self,
        fanouts,
        edge_dir="in",
        prob=None,
        mask=None,
        replace=False,
        prefetch_node_feats=None,
        prefetch_labels=None,
        prefetch_edge_feats=None,
        output_device=None,
        fused=True,
    ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}
        self.g = None

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        # sample_neighbors_fused function requires multithreading to be more efficient
        # than sample_neighbors
        if self.fused and get_num_threads() > 1:
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                for ntype in list(seed_nodes.keys()):
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks

        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )
            block = to_block(frontier, seed_nodes)
            # If sampled from graphbolt-backed DistGraph, `EID` may not be in
            # the block.
            if EID in frontier.edata.keys():
                block.edata[EID] = frontier.edata[EID]
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks


MultiLayerNeighborSampler = NeighborSampler


class MultiLayerFullNeighborSampler(NeighborSampler):
    """Sampler that builds computational dependency of node representations by taking messages
    from all neighbors for multilayer GNN.

    This sampler will make every node gather messages from every single neighbor per edge type.

    Parameters
    ----------
    num_layers : int
        The number of GNN layers to sample.
    kwargs :
        Passed to :class:`dgl.dataloading.NeighborSampler`.

    Examples
    --------
    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from all neighbors for the first,
    second, and third layer respectively (assuming the backend is PyTorch):

    >>> sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
    >>> dataloader = dgl.dataloading.DataLoader(
    ...     g, train_nid, sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, output_nodes, blocks in dataloader:
    ...     train_on(blocks)

    Notes
    -----
    For the concept of MFGs, please refer to
    :ref:`User Guide Section 6 <guide-minibatch>` and
    :doc:`Minibatch Training Tutorials <tutorials/large/L0_neighbor_sampling_overview>`.
    """

    def __init__(self, num_layers, **kwargs):
        super().__init__([-1] * num_layers, **kwargs)

class NeighborSampler_FCR_struct(BlockSampler):
    """
    A neighbor sampler that supports cache-refreshing (FCR) for efficient sampling, 
    tailored for multi-layer GNNs. This sampler augments the sampling process by 
    maintaining a cache of pre-sampled neighborhoods that can be reused across 
    multiple sampling iterations. It introduces cache amplification (via the alpha 
    parameter) and cache refresh cycles (via the T parameter) to manage the balance 
    between sampling efficiency and freshness of the sampled neighborhoods.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    fanouts : list[int] or list[dict[etype, int]]
        List of neighbors to sample per edge type for each GNN layer, with the i-th
        element being the fanout for the i-th GNN layer.
    edge_dir : str, default "in"
        Direction of sampling. Can be either "in" for incoming edges or "out" for outgoing edges.
    prob : str, optional
        Name of the edge feature in g.edata used as the probability for edge sampling.
    alpha : int, default 2
        Cache amplification ratio. Determines the size of the pre-sampled cache relative
        to the actual sampling needs. A larger alpha means more neighbors are pre-sampled.
    T : int, default 1
        Cache refresh cycle. Specifies how often (in terms of sampling iterations) the
        cache should be refreshed.

    Examples
    --------
    Initialize a graph and a NeighborSampler_FCR_struct for a 2-layer GNN with fanouts
    [5, 10]. Assume alpha=2 for double the size of pre-sampling and T=3 for refreshing
    the cache every 3 iterations.

    >>> import dgl
    >>> import torch
    >>> g = dgl.rand_graph(100, 200)  # Random graph with 100 nodes and 200 edges
    >>> g.ndata['feat'] = torch.randn(100, 10)  # Random node features
    >>> sampler = NeighborSampler_FCR_struct(g, [5, 10], alpha=2, T=3)
    
    To perform sampling:

    >>> seed_nodes = torch.tensor([1, 2, 3])  # Nodes for which neighbors are sampled
    >>> for i in range(5):  # Simulate 5 sampling iterations
    ...     seed_nodes, output_nodes, blocks = sampler.sample_blocks(seed_nodes)
    ...     # Process the sampled blocks
    """
    
    def __init__(
            self, 
            g, 
            fanouts, 
            edge_dir='in', 
            alpha=2, 
            T=20,
            prob=None,
            mask=None,
            replace=False,
            prefetch_node_feats=None,
            prefetch_labels=None,
            prefetch_edge_feats=None,
            output_device=None,
            fused=True,
        ):
        self.g = g

        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}

        self.alpha = alpha
        self.cycle = 0  # Initialize sampling cycle counter
        self.amplified_fanouts = [f * alpha for f in fanouts]  # Amplified fanouts for pre-sampling
        self.T = T
        self.Toptim = int(self.g.number_of_nodes() / max(self.amplified_fanouts))
        self.cache_struct = []  # Initialize cache structure
        self.cache_refresh()  # Pre-sample and populate the cache

    def cache_refresh(self,exclude_eids=None):
        """
        Pre-samples neighborhoods with amplified fanouts and refreshes the cache. This method
        is automatically called upon initialization and after every T sampling iterations to
        ensure that the cache is periodically updated with fresh samples.
        """
        self.cache_struct.clear()  # Clear existing cache
        for fanout in self.amplified_fanouts:
            # Sample neighbors for each layer with amplified fanout
            # print("large")
            # print(fanout)
            # print("---")
            frontier = self.g.sample_neighbors(
                torch.arange(0, self.g.number_of_nodes()),  # Consider all nodes as seeds for pre-sampling
                # self.g.number_of_nodes(),
                # 10,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids
            )
            frontier = dgl.add_self_loop(frontier)
            # print(frontier)
            # print(self.cache_struct)
            # print("then append")
            self.cache_struct.append(frontier)  # Update cache with new samples

    def sample_blocks(self, g,seed_nodes, exclude_eids=None):
        """
        Samples blocks from the graph for the specified seed nodes using the cache.

        Parameters
        ----------
        seed_nodes : Tensor
            The nodes for which the neighborhoods are to be sampled.

        Returns
        -------
        tuple
            A tuple containing the seed nodes for the next layer, the output nodes, and
            the list of blocks sampled from the graph.
        """
        output_nodes = seed_nodes

        # refresh cache after a period of time for generalization
        self.cycle += 1
        if self.cycle % self.Toptim == 0:
            self.cache_refresh()  # Refresh cache every T cycles

        blocks = []

        if self.fused and get_num_threads() > 1:
            # print("fused")
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                for ntype in list(seed_nodes.keys()):
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks

        for k in range(len(self.cache_struct)-1,-1,-1):
            cached_structure = self.cache_struct[k]
            fanout = self.fanouts[k]
            frontier = cached_structure.sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids
            )

            # Sample frontier from the cache for acceleration
            block = to_block(frontier, seed_nodes)
            if EID in frontier.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier.edata[EID]
            blocks.insert(0, block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        # output_nodes = seed_nodes
        return seed_nodes, output_nodes, blocks

class NeighborSampler_FCR_struct_shared_cache(BlockSampler):
    def __init__(
            self, 
            g, 
            fanouts, 
            edge_dir='in', 
            alpha=2, 
            T=20,
            prob=None,
            mask=None,
            replace=False,
            prefetch_node_feats=None,
            prefetch_labels=None,
            prefetch_edge_feats=None,
            output_device=None,
            fused=True,
        ):
        self.g = g

        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}

        self.alpha = alpha
        self.cycle = 0  # Initialize sampling cycle counter
        self.amplified_fanouts = [f * alpha for f in fanouts]  # Amplified fanouts for pre-sampling
        self.T = T
        self.Toptim = int(self.g.number_of_nodes() / max(self.amplified_fanouts))
        # self.cache_struct = []  # Initialize cache structure
        self.shared_cache_size = max(self.amplified_fanouts)
        self.shared_cache = None
        self.cache_refresh()  # Pre-sample and populate the cache

    def cache_refresh(self,exclude_eids=None):
        """
        Pre-samples neighborhoods to refresh the shared cache. This method
        is automatically called upon initialization and after every T sampling iterations to
        ensure that the cache is periodically updated with fresh samples.
        """
        del self.shared_cache
        self.shared_cache=self.g.sample_neighbors(
            torch.arange(0, self.g.number_of_nodes()),  # Consider all nodes as seeds for pre-sampling
            self.shared_cache_size,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=exclude_eids
        )

    def sample_blocks(self, g,seed_nodes, exclude_eids=None):
        """
        Samples blocks from the graph for the specified seed nodes using the cache.

        Parameters
        ----------
        seed_nodes : Tensor
            The nodes for which the neighborhoods are to be sampled.

        Returns
        -------
        tuple
            A tuple containing the seed nodes for the next layer, the output nodes, and
            the list of blocks sampled from the graph.
        """
        output_nodes = seed_nodes

        # refresh full cache after every T cycles to learn graph structure
        self.cycle += 1
        if self.cycle % self.Toptim == 0:
            self.cache_refresh()

        blocks = []

        if self.fused and get_num_threads() > 1:
            # print("fused")
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                for ntype in list(seed_nodes.keys()):
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks

        for k in range(len(self.fanouts)-1,-1,-1):
            fanout = self.fanouts[k]
            frontier = self.shared_cache.sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids
            )

            # Sample frontier from the cache for acceleration
            block = to_block(frontier, seed_nodes)
            if EID in frontier.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier.edata[EID]
            blocks.insert(0, block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        # output_nodes = seed_nodes
        return seed_nodes, output_nodes, blocks

class NeighborSampler_OTF_struct_FSCRFCF_shared_cache(BlockSampler):
    """
    Implements an on-the-fly (OTF) neighbor sampling strategy for Deep Graph Library (DGL) graphs. 
    This sampler dynamically samples neighbors while balancing efficiency through caching and 
    freshness of samples by periodically refreshing parts of the cache. It supports specifying 
    fanouts, sampling direction, and probabilities, along with cache management parameters to 
    control the trade-offs between sampling efficiency and cache freshness.

    As for the parameters explanations,
    1. amp_rate: sample a larger cache than the original cache to store the local structure
    2. refresh_rate: decide how many portion should be sampled from disk, and the remaining comes out from cache, then combine them as new disk
    3. T: decide how long time will the cache to refresh and store the new structure (refresh mode in OTF is partially refresh)
    """
    
    def __init__(self, g, 
                fanouts, 
                edge_dir='in', 
                amp_rate=1.5, # cache amplification rate (should be bigger than 1 --> to sample for multiple time)
                refresh_rate=0.4, #propotion of cache to be refresh, should be a positive float smaller than 0.5
                T=100, # refresh time
                prob=None, 
                replace=False, 
                output_device=None, 
                exclude_eids=None,
                mask=None,
                prefetch_node_feats=None,
                prefetch_labels=None,
                prefetch_edge_feats=None,
                fused=True,
                 ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.g = g
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.amp_rate = amp_rate
        self.refresh_rate = refresh_rate
        self.replace = replace
        self.output_device = output_device
        self.exclude_eids = exclude_eids
        self.cycle = 0

        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.fused = fused
        self.mapping = {}
        self.amp_cache_size = [fanout * amp_rate for fanout in fanouts]
        self.Toptim = int(self.g.number_of_nodes() / (max(self.amp_cache_size))*self.amp_rate)
        self.T = T
        # self.cached_graph_structures = [self.initialize_cache(cache_size) for cache_size in self.cache_size]

        self.shared_cache_size = max(self.amp_cache_size)
        self.shared_cache = self.initialize_cache(self.shared_cache_size)

    def initialize_cache(self, fanout_cache_storage):
        """
        Initializes the cache for each layer with an amplified fanout to pre-sample a larger
        set of neighbors. This pre-sampling helps in reducing the need for dynamic sampling 
        at every iteration, thereby improving efficiency.
        """
        cached_graph = self.g.sample_neighbors(
            torch.arange(0, self.g.number_of_nodes()),
            fanout_cache_storage,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
            # mappings=self.mapping
        )
        print("end init cache")
        return cached_graph

    def refresh_cache(self, fanout_cache_refresh):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        fanout_cache_sample = self.shared_cache_size-fanout_cache_refresh
        cache_remain = self.shared_cache.sample_neighbors(
            torch.arange(0, self.g.number_of_nodes()),
            fanout_cache_sample,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )

        disk_to_add = self.g.sample_neighbors(
            torch.arange(0, self.g.number_of_nodes()),
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )

        self.shared_cache = dgl.merge([cache_remain, disk_to_add])
        del cache_remain
        del disk_to_add
        print("end refresh cache")

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        """
        Samples blocks for GNN layers by combining cached samples with dynamically sampled 
        neighbors. This method also partially refreshes the cache based on specified parameters 
        to balance between sampling efficiency and the freshness of the samples.
        """
        self.cycle += 1
        blocks = []
        output_nodes = seed_nodes
        if((self.cycle % self.Toptim)==0):
            # Refresh cache partially
            fanout_cache_refresh = int(self.shared_cache_size * self.refresh_rate)
            self.refresh_cache(fanout_cache_refresh)
            
        for i, (fanout) in enumerate(reversed(self.fanouts)):            
            # Sample from cache
            frontier_from_cache = self.shared_cache.sample_neighbors(
                seed_nodes,
                #fanout_cache_retrieval,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=self.exclude_eids,
            )

            # Convert the merged frontier to a block
            block = to_block(frontier_from_cache, seed_nodes)
            if EID in frontier_from_cache.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier_from_cache.edata[EID]
            blocks.append(block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer
        
        return seed_nodes,output_nodes, blocks

class NeighborSampler_OTF_struct_FSCRFCF(BlockSampler):
    """
    Implements an on-the-fly (OTF) neighbor sampling strategy for Deep Graph Library (DGL) graphs. 
    This sampler dynamically samples neighbors while balancing efficiency through caching and 
    freshness of samples by periodically refreshing parts of the cache. It supports specifying 
    fanouts, sampling direction, and probabilities, along with cache management parameters to 
    control the trade-offs between sampling efficiency and cache freshness.

    As for the parameters explanations,
    1. amp_rate: sample a larger cache than the original cache to store the local structure
    2. refresh_rate: decide how many portion should be sampled from disk, and the remaining comes out from cache, then combine them as new disk
    3. T: decide how long time will the cache to refresh and store the new structure (refresh mode in OTF is partially refresh)
    """
    
    def __init__(self, g, 
                fanouts, 
                edge_dir='in', 
                amp_rate=1.5, # cache amplification rate (should be bigger than 1 --> to sample for multiple time)
                refresh_rate=0.4, #propotion of cache to be refresh, should be a positive float smaller than 0.5
                T=100, # refresh time
                prob=None, 
                replace=False, 
                output_device=None, 
                exclude_eids=None,
                mask=None,
                prefetch_node_feats=None,
                prefetch_labels=None,
                prefetch_edge_feats=None,
                fused=True,
                 ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.g = g
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.amp_rate = amp_rate
        self.refresh_rate = refresh_rate
        self.replace = replace
        self.output_device = output_device
        self.exclude_eids = exclude_eids
        self.cycle = 0

        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.fused = fused
        self.mapping = {}
        self.cache_size = [fanout * amp_rate for fanout in fanouts]
        self.T = T
        self.Toptim = int(self.g.number_of_nodes() / (max(self.cache_size))*self.amp_rate)
        self.cached_graph_structures = [self.initialize_cache(cache_size) for cache_size in self.cache_size]

    def initialize_cache(self, fanout_cache_storage):
        """
        Initializes the cache for each layer with an amplified fanout to pre-sample a larger
        set of neighbors. This pre-sampling helps in reducing the need for dynamic sampling 
        at every iteration, thereby improving efficiency.
        """
        cached_graph = self.g.sample_neighbors(
            torch.arange(0, self.g.number_of_nodes()),
            fanout_cache_storage,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        print("end init cache")
        return cached_graph

    def refresh_cache(self,layer_id, cached_graph_structure, fanout_cache_refresh):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        fanout_cache_sample = self.cache_size[layer_id]-fanout_cache_refresh
        cache_remain = cached_graph_structure.sample_neighbors(
            torch.arange(0, self.g.number_of_nodes()),
            fanout_cache_sample,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )

        disk_to_add = self.g.sample_neighbors(
            torch.arange(0, self.g.number_of_nodes()),
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )

        refreshed_cache = dgl.merge([cache_remain, disk_to_add])
        print("end refresh cache")
        return refreshed_cache

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        """
        Samples blocks for GNN layers by combining cached samples with dynamically sampled 
        neighbors. This method also partially refreshes the cache based on specified parameters 
        to balance between sampling efficiency and the freshness of the samples.
        """
        blocks = []
        output_nodes = seed_nodes
        self.cycle += 1
        if((self.cycle % self.Toptim)==0):
            for i in range(0,len(self.cached_graph_structures)):
                # Refresh cache partially
                fanout_cache_refresh = int(self.cache_size[i] * self.refresh_rate)
                self.cached_graph_structures[i]=self.refresh_cache(i, self.cached_graph_structures[i], fanout_cache_refresh)
            
        for i, (fanout, cached_graph_structure) in enumerate(zip(reversed(self.fanouts), reversed(self.cached_graph_structures))):            
            # Sample from cache
            frontier_from_cache = self.cached_graph_structures[i].sample_neighbors(
                seed_nodes,
                #fanout_cache_retrieval,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=self.exclude_eids,
            )

            # Convert the merged frontier to a block
            block = to_block(frontier_from_cache, seed_nodes)
            if EID in frontier_from_cache.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier_from_cache.edata[EID]
            blocks.append(block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer
        
        return seed_nodes,output_nodes, blocks

class NeighborSampler_OTF_struct_PCFFSCR_shared_cache(BlockSampler):
    """
    Implements an on-the-fly (OTF) neighbor sampling strategy for Deep Graph Library (DGL) graphs. 
    This sampler dynamically samples neighbors while balancing efficiency through caching and 
    freshness of samples by periodically refreshing parts of the cache. It supports specifying 
    fanouts, sampling direction, and probabilities, along with cache management parameters to 
    control the trade-offs between sampling efficiency and cache freshness.

    As for the parameters explanations,
    1. amp_rate: sample a larger cache than the original cache to store the local structure
    2. refresh_rate: decide how many portion should be sampled from disk, and the remaining comes out from cache, then combine them as new disk
    3. T: decide how long time will the cache to refresh and store the new structure (refresh mode in OTF is partially refresh)
    """
    
    def __init__(self, g, 
                fanouts, 
                edge_dir='in', 
                amp_rate=1.5, # cache amplification rate (should be bigger than 1 --> to sample for multiple time)
                fetch_rate=0.4, #propotion of cache to be fetch from cache, should be a positive float smaller than 0.5
                T_fetch=3, # fetch period of time
                T_refresh=None, # refresh time
                prob=None, 
                replace=False, 
                output_device=None, 
                exclude_eids=None,
                mask=None,
                prefetch_node_feats=None,
                prefetch_labels=None,
                prefetch_edge_feats=None,
                fused=True,
                 ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.g = g
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.amp_rate = amp_rate
        self.fetch_rate = fetch_rate
        self.replace = replace
        self.output_device = output_device
        self.exclude_eids = exclude_eids

        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.fused = fused
        self.mapping = {}
        self.amp_cache_size = [fanout * amp_rate for fanout in fanouts]
        if T_refresh!=None:
            self.T_refresh = T_refresh
        else:
            self.T_refresh = int(self.g.number_of_nodes()/max(self.fanouts) *self.amp_rate)
        self.T_fetch = T_fetch
        # self.cached_graph_structures = None
        self.cycle = 0

        self.shared_cache_size = max(self.amp_cache_size)
        self.shared_cache = self.full_cache_refresh(self.shared_cache_size)

    def full_cache_refresh(self, fanout_cache_storage):
        """
        Initializes the cache for each layer with an amplified fanout to pre-sample a larger
        set of neighbors. This pre-sampling helps in reducing the need for dynamic sampling 
        at every iteration, thereby improving efficiency.
        """
        cached_graph = self.g.sample_neighbors(
            torch.arange(0, self.g.number_of_nodes()),
            fanout_cache_storage,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        print("cache refresh")
        return cached_graph

    def OTF_fetch(self,layer_id,  seed_nodes, fanout_cache_fetch):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        print("OTF fetch cache")
        if(fanout_cache_fetch==self.fanouts[layer_id]):
            cache_fetch = self.shared_cache.sample_neighbors(
            seed_nodes,
            fanout_cache_fetch,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
            )
            return cache_fetch
        else:
            fanout_disk_fetch = self.fanouts[layer_id]-fanout_cache_fetch
            cache_fetch = self.shared_cache.sample_neighbors(
                seed_nodes,
                fanout_cache_fetch,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=self.exclude_eids,
            )

            disk_fetch = self.g.sample_neighbors(
                seed_nodes,
                fanout_disk_fetch,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=self.exclude_eids,
            )

            OTF_fetch_res = dgl.merge([cache_fetch, disk_fetch])
            return OTF_fetch_res

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        """
        Samples blocks for GNN layers by combining cached samples with dynamically sampled 
        neighbors. This method also partially refreshes the cache based on specified parameters 
        to balance between sampling efficiency and the freshness of the samples.
        """
        blocks = []
        output_nodes = seed_nodes

        self.cycle += 1
        print("self.T_refresh=",self.T_refresh)
        # refresh full cache after a period of time
        if((self.cycle%self.T_refresh)==0):
            self.shared_cache = self.full_cache_refresh(self.shared_cache_size)
            # self.cached_graph_structures = [self.full_cache_refresh(cache_size) for cache_size in self.cache_size]
        
        for i, (fanout) in enumerate(reversed(self.fanouts)):
            fanout_cache_fetch = int(fanout * self.fetch_rate)

            # fetch cache partially
            if((self.cycle%self.T_fetch)==0):
                frontier_OTF = self.OTF_fetch(i, seed_nodes, fanout_cache_fetch)
            else:
                #frontier_OTF = self.OTF_fetch(i, seed_nodes, self.fanouts[i])
                frontier_OTF = self.shared_cache.sample_neighbors(
                    seed_nodes,
                    fanout,
                    edge_dir=self.edge_dir,
                    prob=self.prob,
                    replace=self.replace,
                    output_device=self.output_device,
                    exclude_edges=self.exclude_eids,
                )
            
            # Convert the merged frontier to a block
            block = to_block(frontier_OTF, seed_nodes)
            if EID in frontier_OTF.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier_OTF.edata[EID]
            blocks.append(block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        return seed_nodes,output_nodes, blocks

class NeighborSampler_OTF_struct_PCFFSCR(BlockSampler):
    """
    Implements an on-the-fly (OTF) neighbor sampling strategy for Deep Graph Library (DGL) graphs. 
    This sampler dynamically samples neighbors while balancing efficiency through caching and 
    freshness of samples by periodically refreshing parts of the cache. It supports specifying 
    fanouts, sampling direction, and probabilities, along with cache management parameters to 
    control the trade-offs between sampling efficiency and cache freshness.

    As for the parameters explanations,
    1. amp_rate: sample a larger cache than the original cache to store the local structure
    2. refresh_rate: decide how many portion should be sampled from disk, and the remaining comes out from cache, then combine them as new disk
    3. T: decide how long time will the cache to refresh and store the new structure (refresh mode in OTF is partially refresh)
    """
    
    def __init__(self, g, 
                fanouts, 
                edge_dir='in', 
                amp_rate=1.5, # cache amplification rate (should be bigger than 1 --> to sample for multiple time)
                fetch_rate=0.4, #propotion of cache to be refresh, should be a positive float smaller than 0.5
                T_fetch=3, # fetch period of time
                T_refresh=None, # refresh time
                prob=None, 
                replace=False, 
                output_device=None, 
                exclude_eids=None,
                mask=None,
                prefetch_node_feats=None,
                prefetch_labels=None,
                prefetch_edge_feats=None,
                fused=True,
                 ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.g = g
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.amp_rate = amp_rate
        self.fetch_rate = fetch_rate
        self.replace = replace
        self.output_device = output_device
        self.exclude_eids = exclude_eids

        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.fused = fused
        self.mapping = {}
        self.cache_size = [fanout * amp_rate for fanout in fanouts]
        if T_refresh!=None:
            self.T_refresh = T_refresh
        else:
            self.T_refresh = int(self.g.number_of_nodes()/max(self.fanouts) *self.amp_rate)
        self.T_fetch = T_fetch
        self.cached_graph_structures = [self.full_cache_refresh(cache_size) for cache_size in self.cache_size]
        self.cycle = 0

    def full_cache_refresh(self, fanout_cache_storage):
        """
        Initializes the cache for each layer with an amplified fanout to pre-sample a larger
        set of neighbors. This pre-sampling helps in reducing the need for dynamic sampling 
        at every iteration, thereby improving efficiency.
        """
        cached_graph = self.g.sample_neighbors(
            torch.arange(0, self.g.number_of_nodes()),
            fanout_cache_storage,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        print("cache refresh")
        return cached_graph

    def OTF_fetch(self,layer_id, cached_graph_structure, seed_nodes, fanout_cache_fetch):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        fanout_disk_fetch = self.fanouts[layer_id]-fanout_cache_fetch
        cache_fetch = cached_graph_structure.sample_neighbors(
            seed_nodes,
            fanout_cache_fetch,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )

        disk_fetch = self.g.sample_neighbors(
            seed_nodes,
            fanout_disk_fetch,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )

        OTF_fetch_res = dgl.merge([cache_fetch, disk_fetch])
        print("OTF fetch cache")
        return OTF_fetch_res

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        """
        Samples blocks for GNN layers by combining cached samples with dynamically sampled 
        neighbors. This method also partially refreshes the cache based on specified parameters 
        to balance between sampling efficiency and the freshness of the samples.
        """
        blocks = []
        output_nodes = seed_nodes
        self.cycle += 1

        # refresh full cache after a period of time
        if((self.cycle%self.T_refresh)==0):
            self.cached_graph_structures = [self.full_cache_refresh(cache_size) for cache_size in self.cache_size]
        
        for i, (fanout, cached_graph_structure) in enumerate(zip(reversed(self.fanouts), reversed(self.cached_graph_structures))):
            fanout_cache_refresh = int(fanout * self.fetch_rate)

            # fetch cache partially
            if((self.cycle%self.T_fetch)==0):
                frontier_OTF = self.OTF_fetch(i, cached_graph_structure, seed_nodes, fanout_cache_refresh)
            else:
                # frontier_OTF = self.OTF_fetch(i, cached_graph_structure, seed_nodes, self.fanouts[i])
                frontier_OTF = cached_graph_structure.sample_neighbors(
                    seed_nodes,
                    fanout,
                    edge_dir=self.edge_dir,
                    prob=self.prob,
                    replace=self.replace,
                    output_device=self.output_device,
                    exclude_edges=self.exclude_eids,
                )
            
            # Convert the merged frontier to a block
            block = to_block(frontier_OTF, seed_nodes)
            if EID in frontier_OTF.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier_OTF.edata[EID]
            blocks.append(block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        return seed_nodes,output_nodes, blocks

class NeighborSampler_OTF_struct_PCFPSCR_SC(BlockSampler):
    """
    Implements an on-the-fly (OTF) neighbor sampling strategy for Deep Graph Library (DGL) graphs. 
    This sampler dynamically samples neighbors while balancing efficiency through caching and 
    freshness of samples by periodically refreshing parts of the cache. It supports specifying 
    fanouts, sampling direction, and probabilities, along with cache management parameters to 
    control the trade-offs between sampling efficiency and cache freshness.

    As for the parameters explanations,
    1. amp_rate: sample a larger cache than the original cache to store the local structure
    2. refresh_rate: decide how many portion should be sampled from disk, and the remaining comes out from cache, then combine them as new disk
    3. T: decide how long time will the cache to refresh and store the new structure (refresh mode in OTF is partially refresh)
    """
    
    def __init__(self, g, 
                fanouts, 
                edge_dir='in', 
                amp_rate=1.5, # cache amplification rate (should be bigger than 1 --> to sample for multiple time)
                refresh_rate=0.4, #propotion of cache to be refresh, should be a positive float smaller than 0.5
                T=50, # refresh time, for example
                prob=None, 
                replace=False, 
                output_device=None, 
                exclude_eids=None,
                mask=None,
                prefetch_node_feats=None,
                prefetch_labels=None,
                prefetch_edge_feats=None,
                fused=True,
                 ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.g = g
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.amp_rate = amp_rate
        self.refresh_rate = refresh_rate
        self.replace = replace
        self.output_device = output_device
        self.exclude_eids = exclude_eids

        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.fused = fused
        self.mapping = {}
        self.amp_cache_size = [fanout * amp_rate for fanout in fanouts]
        self.T = T
        self.cycle = 0
        # self.cached_graph_structures = [self.initialize_cache(cache_size) for cache_size in self.cache_size]

        self.shared_cache_size = max(self.amp_cache_size)
        self.shared_cache = self.initialize_cache(self.shared_cache_size)

    def initialize_cache(self, fanout_cache_storage):
        """
        Initializes the cache for each layer with an amplified fanout to pre-sample a larger
        set of neighbors. This pre-sampling helps in reducing the need for dynamic sampling 
        at every iteration, thereby improving efficiency.
        """
        cached_graph = self.g.sample_neighbors(
            torch.arange(0, self.g.number_of_nodes()),
            fanout_cache_storage,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        print("end init cache")
        return cached_graph

    def OTF_rf_cache(self,layer_id, seed_nodes, fanout_cache_refresh, fanout):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        fanout_cache_remain = self.shared_cache_size-fanout_cache_refresh
        fanout_cache_pr = fanout-fanout_cache_refresh

        all_nodes = torch.arange(0, self.g.number_of_nodes())
        # mask = ~torch.isin(all_nodes, seed_nodes)
        # # 使用布尔掩码来选择不在seed_nodes中的节点
        # unchanged_nodes = all_nodes[mask]

        # unchanged_nodes = torch.arange(0, self.g.number_of_nodes())-seed_nodes
        # the rest node structure remain the same
        unchanged_structure = self.shared_cache.sample_neighbors(
            all_nodes,
            # unchanged_nodes,
            self.shared_cache_size,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        # the OTF node structure should 
        changed_cache_remain = self.shared_cache.sample_neighbors(
            seed_nodes,
            fanout_cache_remain,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        cache_pr = self.shared_cache.sample_neighbors(
            seed_nodes,
            fanout_cache_pr,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        changed_disk_to_add = self.g.sample_neighbors(
            seed_nodes,
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        refreshed_cache = dgl.merge([unchanged_structure, changed_cache_remain, changed_disk_to_add])
        retrieval_cache = dgl.merge([cache_pr, changed_disk_to_add])
        return refreshed_cache, retrieval_cache

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        """
        Samples blocks for GNN layers by combining cached samples with dynamically sampled 
        neighbors. This method also partially refreshes the cache based on specified parameters 
        to balance between sampling efficiency and the freshness of the samples.
        """
        blocks = []
        output_nodes = seed_nodes
        self.cycle += 1
        for i, (fanout) in enumerate(reversed(self.fanouts)):
            fanout_cache_refresh = int(fanout * self.refresh_rate)

            # Refresh cache&disk partially, while retrieval cache&disk partially
            if(self.cycle%self.T==0):
                self.shared_cache, frontier_comp = self.OTF_rf_cache(i, seed_nodes, fanout_cache_refresh, fanout)
            else:
                frontier_comp = self.shared_cache.sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=self.exclude_eids,
            )
            
            # Convert the merged frontier to a block
            block = to_block(frontier_comp, seed_nodes)
            if EID in frontier_comp.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier_comp.edata[EID]
            blocks.append(block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        return seed_nodes,output_nodes, blocks

class NeighborSampler_OTF_struct_PCFPSCR(BlockSampler):
    """
    Implements an on-the-fly (OTF) neighbor sampling strategy for Deep Graph Library (DGL) graphs. 
    This sampler dynamically samples neighbors while balancing efficiency through caching and 
    freshness of samples by periodically refreshing parts of the cache. It supports specifying 
    fanouts, sampling direction, and probabilities, along with cache management parameters to 
    control the trade-offs between sampling efficiency and cache freshness.

    As for the parameters explanations,
    1. amp_rate: sample a larger cache than the original cache to store the local structure
    2. refresh_rate: decide how many portion should be sampled from disk, and the remaining comes out from cache, then combine them as new disk
    3. T: decide how long time will the cache to refresh and store the new structure (refresh mode in OTF is partially refresh)
    """
    
    def __init__(self, g, 
                fanouts, 
                edge_dir='in', 
                amp_rate=1.5, # cache amplification rate (should be bigger than 1 --> to sample for multiple time)
                refresh_rate=0.4, #propotion of cache to be refresh, should be a positive float smaller than 0.5
                T=50, # refresh time, for example
                prob=None, 
                replace=False, 
                output_device=None, 
                exclude_eids=None,
                mask=None,
                prefetch_node_feats=None,
                prefetch_labels=None,
                prefetch_edge_feats=None,
                fused=True,
                 ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.g = g
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.amp_rate = amp_rate
        self.refresh_rate = refresh_rate
        self.replace = replace
        self.output_device = output_device
        self.exclude_eids = exclude_eids

        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.fused = fused
        self.mapping = {}
        self.cache_size = [fanout * amp_rate for fanout in fanouts]
        self.T = T
        self.cached_graph_structures = [self.initialize_cache(cache_size) for cache_size in self.cache_size]
        self.cycle = 0

    def initialize_cache(self, fanout_cache_storage):
        """
        Initializes the cache for each layer with an amplified fanout to pre-sample a larger
        set of neighbors. This pre-sampling helps in reducing the need for dynamic sampling 
        at every iteration, thereby improving efficiency.
        """
        cached_graph = self.g.sample_neighbors(
            torch.arange(0, self.g.number_of_nodes()),
            fanout_cache_storage,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        print("end init cache")
        return cached_graph

    def OTF_rf_cache(self,layer_id, cached_graph_structure, seed_nodes, fanout_cache_refresh, fanout):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        fanout_cache_remain = self.cache_size[layer_id]-fanout_cache_refresh
        fanout_cache_pr = fanout-fanout_cache_refresh
        # unchanged_nodes = range(torch.arange(0, self.g.number_of_nodes()))-seed_nodes
        # the rest node structure remain the same
        all_nodes = torch.arange(0, self.g.number_of_nodes())
        mask = ~torch.isin(all_nodes, seed_nodes)
        # 使用布尔掩码来选择不在seed_nodes中的节点
        unchanged_nodes = all_nodes[mask]
        unchanged_structure = cached_graph_structure.sample_neighbors(
            unchanged_nodes,
            self.cache_size[layer_id],
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        # the OTF node structure should 
        changed_cache_remain = cached_graph_structure.sample_neighbors(
            seed_nodes,
            fanout_cache_remain,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        cache_pr = cached_graph_structure.sample_neighbors(
            seed_nodes,
            fanout_cache_pr,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        changed_disk_to_add = self.g.sample_neighbors(
            seed_nodes,
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        refreshed_cache = dgl.merge([unchanged_structure, changed_cache_remain, changed_disk_to_add])
        retrieval_cache = dgl.merge([cache_pr, changed_disk_to_add])
        del unchanged_structure, changed_cache_remain, cache_pr, changed_disk_to_add
        return refreshed_cache, retrieval_cache

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        """
        Samples blocks for GNN layers by combining cached samples with dynamically sampled 
        neighbors. This method also partially refreshes the cache based on specified parameters 
        to balance between sampling efficiency and the freshness of the samples.
        """
        blocks = []
        output_nodes = seed_nodes
        self.cycle += 1
        for i, (fanout, cached_graph_structure) in enumerate(zip(reversed(self.fanouts), reversed(self.cached_graph_structures))):
            if(self.cycle % self.T) == 0:
                fanout_cache_refresh = int(fanout * self.refresh_rate)

                # Refresh cache&disk partially, while retrieval cache&disk partially
                self.cached_graph_structures[i], frontier_comp = self.OTF_rf_cache(i, cached_graph_structure, seed_nodes, fanout_cache_refresh, fanout)
            else:
                frontier_comp = self.cached_graph_structures[i].sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=self.exclude_eids,
            )
            
            # Convert the merged frontier to a block
            block = to_block(frontier_comp, seed_nodes)
            if EID in frontier_comp.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier_comp.edata[EID]
            blocks.append(block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        return seed_nodes,output_nodes, blocks

class NeighborSampler_OTF_struct_PSCRFCF_SC(BlockSampler):
    """
    Implements an on-the-fly (OTF) neighbor sampling strategy for Deep Graph Library (DGL) graphs. 
    This sampler dynamically samples neighbors while balancing efficiency through caching and 
    freshness of samples by periodically refreshing parts of the cache. It supports specifying 
    fanouts, sampling direction, and probabilities, along with cache management parameters to 
    control the trade-offs between sampling efficiency and cache freshness.

    As for the parameters explanations,
    1. amp_rate: sample a larger cache than the original cache to store the local structure
    2. refresh_rate: decide how many portion should be sampled from disk, and the remaining comes out from cache, then combine them as new disk
    3. T: decide how long time will the cache to refresh and store the new structure (refresh mode in OTF is partially refresh)
    """
    
    def __init__(self, g, 
                fanouts, 
                edge_dir='in', 
                amp_rate=1.5, # cache amplification rate (should be bigger than 1 --> to sample for multiple time)
                refresh_rate=0.4, #propotion of cache to be refresh, should be a positive float smaller than 0.5
                T=50, # refresh time
                prob=None, 
                replace=False, 
                output_device=None, 
                exclude_eids=None,
                mask=None,
                prefetch_node_feats=None,
                prefetch_labels=None,
                prefetch_edge_feats=None,
                fused=True,
                 ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.g = g
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.amp_rate = amp_rate
        self.refresh_rate = refresh_rate
        self.replace = replace
        self.output_device = output_device
        self.exclude_eids = exclude_eids

        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.fused = fused
        self.mapping = {}
        # self.cache_size = [fanout * amp_rate for fanout in fanouts]
        self.T = T
        # self.cached_graph_structures = [self.initialize_cache(cache_size) for cache_size in self.cache_size]
        self.cycle = 0

        self.shared_cache_size = max(self.fanouts)*self.amp_rate
        self.shared_cache = self.initialize_cache(self.shared_cache_size)

    def initialize_cache(self, fanout_cache_storage):
        """
        Initializes the cache for each layer with an amplified fanout to pre-sample a larger
        set of neighbors. This pre-sampling helps in reducing the need for dynamic sampling 
        at every iteration, thereby improving efficiency.
        """
        cached_graph = self.g.sample_neighbors(
            torch.arange(0, self.g.number_of_nodes()),
            fanout_cache_storage,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        print("end init cache")
        return cached_graph

    def OTF_refresh_cache(self,layer_id, cached_graph_structure, seed_nodes, fanout_cache_refresh):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        all_nodes = torch.arange(0, self.g.number_of_nodes())
        mask = ~torch.isin(all_nodes, seed_nodes)
        # use bool mask to select those nodes in all nodes but not in seed_nodes
        unchanged_nodes = all_nodes[mask]
        fanout_cache_sample = self.shared_cache_size-fanout_cache_refresh
        # unchanged_nodes = range(torch.arange(0, self.g.number_of_nodes()))-seed_nodes
        # the rest node structure remain the same
        unchanged_structure = cached_graph_structure.sample_neighbors(
            unchanged_nodes,
            self.shared_cache_size,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        # the OTF node structure should 
        changed_cache_remain = cached_graph_structure.sample_neighbors(
            seed_nodes,
            fanout_cache_sample,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        changed_disk_to_add = self.g.sample_neighbors(
            seed_nodes,
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        refreshed_cache = dgl.merge([unchanged_structure, changed_cache_remain, changed_disk_to_add])
        del unchanged_structure, changed_cache_remain, changed_disk_to_add
        return refreshed_cache

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        """
        Samples blocks for GNN layers by combining cached samples with dynamically sampled 
        neighbors. This method also partially refreshes the cache based on specified parameters 
        to balance between sampling efficiency and the freshness of the samples.
        """
        blocks = []
        output_nodes = seed_nodes
        self.cycle += 1
        for i, (fanout) in enumerate(reversed(self.fanouts)):
            fanout_cache_refresh = int(fanout * self.refresh_rate)

            # Refresh cache partially
            if((self.cycle % self.T) ==0):
                self.shared_cache = self.OTF_refresh_cache(i, self.shared_cache, seed_nodes, fanout_cache_refresh)
            
            # Sample from cache
            frontier_cache = self.shared_cache.sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=self.exclude_eids,
            )

            merged_frontier = frontier_cache
            
            # Convert the merged frontier to a block
            block = to_block(merged_frontier, seed_nodes)
            if EID in merged_frontier.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = merged_frontier.edata[EID]
            blocks.append(block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        return seed_nodes,output_nodes, blocks

class NeighborSampler_OTF_struct_PSCRFCF(BlockSampler):
    """
    Implements an on-the-fly (OTF) neighbor sampling strategy for Deep Graph Library (DGL) graphs. 
    This sampler dynamically samples neighbors while balancing efficiency through caching and 
    freshness of samples by periodically refreshing parts of the cache. It supports specifying 
    fanouts, sampling direction, and probabilities, along with cache management parameters to 
    control the trade-offs between sampling efficiency and cache freshness.

    As for the parameters explanations,
    1. amp_rate: sample a larger cache than the original cache to store the local structure
    2. refresh_rate: decide how many portion should be sampled from disk, and the remaining comes out from cache, then combine them as new disk
    3. T: decide how long time will the cache to refresh and store the new structure (refresh mode in OTF is partially refresh)
    """
    
    def __init__(self, g, 
                fanouts, 
                edge_dir='in', 
                amp_rate=1.5, # cache amplification rate (should be bigger than 1 --> to sample for multiple time)
                refresh_rate=0.4, #propotion of cache to be refresh, should be a positive float smaller than 0.5
                T=50, # refresh time
                prob=None, 
                replace=False, 
                output_device=None, 
                exclude_eids=None,
                mask=None,
                prefetch_node_feats=None,
                prefetch_labels=None,
                prefetch_edge_feats=None,
                fused=True,
                 ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.g = g
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.amp_rate = amp_rate
        self.refresh_rate = refresh_rate
        self.replace = replace
        self.output_device = output_device
        self.exclude_eids = exclude_eids

        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.fused = fused
        self.mapping = {}
        self.cache_size = [fanout * amp_rate for fanout in fanouts]
        self.T = T
        self.cached_graph_structures = [self.initialize_cache(cache_size) for cache_size in self.cache_size]
        self.cycle = 0

    def initialize_cache(self, fanout_cache_storage):
        """
        Initializes the cache for each layer with an amplified fanout to pre-sample a larger
        set of neighbors. This pre-sampling helps in reducing the need for dynamic sampling 
        at every iteration, thereby improving efficiency.
        """
        cached_graph = self.g.sample_neighbors(
            torch.arange(0, self.g.number_of_nodes()),
            fanout_cache_storage,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        print("end init cache")
        return cached_graph

    def OTF_refresh_cache(self,layer_id, cached_graph_structure, seed_nodes, fanout_cache_refresh):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        fanout_cache_sample = self.cache_size[layer_id]-fanout_cache_refresh
        # unchanged_nodes = range(torch.arange(0, self.g.number_of_nodes()))-seed_nodes
        all_nodes = torch.arange(0, self.g.number_of_nodes())
        mask = ~torch.isin(all_nodes, seed_nodes)
        # 使用布尔掩码来选择不在seed_nodes中的节点
        unchanged_nodes = all_nodes[mask]
        # the rest node structure remain the same
        unchanged_structure = cached_graph_structure.sample_neighbors(
            unchanged_nodes,
            self.cache_size[layer_id],
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        # the OTF node structure should 
        changed_cache_remain = cached_graph_structure.sample_neighbors(
            seed_nodes,
            fanout_cache_sample,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        changed_disk_to_add = self.g.sample_neighbors(
            seed_nodes,
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        refreshed_cache = dgl.merge([unchanged_structure, changed_cache_remain, changed_disk_to_add])
        del unchanged_structure, changed_cache_remain, changed_disk_to_add
        return refreshed_cache

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        """
        Samples blocks for GNN layers by combining cached samples with dynamically sampled 
        neighbors. This method also partially refreshes the cache based on specified parameters 
        to balance between sampling efficiency and the freshness of the samples.
        """
        blocks = []
        output_nodes = seed_nodes
        self.cycle += 1
        for i, (fanout, cached_graph_structure) in enumerate(zip(reversed(self.fanouts), reversed(self.cached_graph_structures))):
            fanout_cache_refresh = int(fanout * self.refresh_rate)

            # Refresh cache partially
            if(self.cycle%self.T==0):
                self.cached_graph_structures[i] = self.OTF_refresh_cache(i, cached_graph_structure, seed_nodes, fanout_cache_refresh)
            
            # Sample from cache
            frontier_cache = self.cached_graph_structures[i].sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=self.exclude_eids,
            )

            merged_frontier = frontier_cache
            
            # Convert the merged frontier to a block
            block = to_block(merged_frontier, seed_nodes)
            if EID in merged_frontier.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = merged_frontier.edata[EID]
            blocks.append(block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        return seed_nodes,output_nodes, blocks


class NeighborSampler_FCR_struct_hete(BlockSampler):    
    def __init__(
            self, 
            g,
            fanouts, 
            edge_dir='in', 
            alpha=2, 
            T=20,
            hete_label=None,
            prob=None,
            mask=None,
            replace=False,
            prefetch_node_feats=None,
            prefetch_labels=None,
            prefetch_edge_feats=None,
            output_device=None,
            fused=True,
        ):

        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.g = g
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}

        self.alpha = alpha
        self.cycle = 0  # Initialize sampling cycle counter
        self.amplified_fanouts = [f * alpha for f in fanouts]  # Amplified fanouts for pre-sampling
        self.T = T
        self.Toptim = None # int(self.g.number_of_nodes() / max(self.amplified_fanouts))
        self.cache_struct = []  # Initialize cache structure
        self.hete_label = hete_label
        # self.cache_refresh(self.g)  # Pre-sample and populate the cache

    def cache_refresh(self,g,exclude_eids=None):
        """
        Pre-samples neighborhoods with amplified fanouts and refreshes the cache. This method
        is automatically called upon initialization and after every T sampling iterations to
        ensure that the cache is periodically updated with fresh samples.
        """
        self.cache_struct.clear()  # Clear existing cache
        for fanout in self.amplified_fanouts:
            # Sample neighbors for each layer with amplified fanout
            frontier = g.sample_neighbors(
                {self.hete_label:list(range(0, g.num_nodes(self.hete_label)))},
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids
            )
            # frontier = dgl.add_self_loop(frontier)
            self.cache_struct.append(frontier)  # Update cache with new samples

    def sample_blocks(self, g,seed_nodes, exclude_eids=None):
        """
        Samples blocks from the graph for the specified seed nodes using the cache.

        Parameters
        ----------
        seed_nodes : Tensor
            The nodes for which the neighborhoods are to be sampled.

        Returns
        -------
        tuple
            A tuple containing the seed nodes for the next layer, the output nodes, and
            the list of blocks sampled from the graph.
        """
        output_nodes = seed_nodes

        # refresh cache after a period of time for generalization
        self.Toptim = int(g.number_of_nodes() / max(self.amplified_fanouts))
        if self.cycle % self.Toptim == 0:
            self.cache_refresh(g)  # Refresh cache every T cycles
        
        self.cycle += 1

        blocks = []

        if self.fused and get_num_threads() > 1:
            # print("fused")
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                for ntype in list(seed_nodes.keys()):
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks

        for k in range(len(self.cache_struct)-1,-1,-1):
            cached_structure = self.cache_struct[k]
            fanout = self.fanouts[k]
            frontier = cached_structure.sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids
            )

            # Sample frontier from the cache for acceleration
            block = to_block(frontier, seed_nodes)
            if EID in frontier.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier.edata[EID]
            blocks.insert(0, block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        # output_nodes = seed_nodes
        return seed_nodes, output_nodes, blocks

# class NeighborSampler_FCR_struct_hete(BlockSampler):

#     def __init__(
#         self,
#         g,
#         fanouts,
#         edge_dir="in",
#         alpha = 2,
#         T = 20,
#         hete_label = None,
#         prob=None,
#         mask=None,
#         replace=False,
#         prefetch_node_feats=None,
#         prefetch_labels=None,
#         prefetch_edge_feats=None,
#         output_device=None,
#         fused=True,
#     ):
#         super().__init__(
#             prefetch_node_feats=prefetch_node_feats,
#             prefetch_labels=prefetch_labels,
#             prefetch_edge_feats=prefetch_edge_feats,
#             output_device=output_device,
#         )
#         self.g = g
#         self.fanouts = fanouts
#         self.edge_dir = edge_dir
#         if mask is not None and prob is not None:
#             raise ValueError(
#                 "Mask and probability arguments are mutually exclusive. "
#                 "Consider multiplying the probability with the mask "
#                 "to achieve the same goal."
#             )
#         self.prob = prob or mask
#         self.replace = replace
#         self.fused = fused
#         self.mapping = {}
#         self.g = g
#         self.cycle = 0
#         self.cached_structure = []
#         self.amplified_fanouts = [f * alpha for f in fanouts]  # Amplified fanouts for pre-sampling
#         self.T = T
#         self.Toptim = None #int(self.g[self.hete_label].number_of_nodes() / max(self.amplified_fanouts))
#         self.hete_label = hete_label
#         self.cache_refresh()

#         # print(self.g.number_of_nodes("paper"))
    
#     def cache_refresh(self,exclude_eids=None):
#         for i in range(0,len(self.fanouts)):
#             frontier1 = self.g.sample_neighbors(
#                     {self.hete_label:list(range(0, self.g.num_nodes(self.hete_label)))},
#                     self.amplified_fanouts[i],
#                     edge_dir=self.edge_dir,
#                     prob=self.prob,
#                     replace=self.replace,
#                     output_device=self.output_device,
#                     exclude_edges=exclude_eids,
#             )
#             self.cached_structure.append(frontier1)
        

#     def sample_blocks(self, g, seed_nodes, exclude_eids=None):
#         output_nodes = seed_nodes
#         blocks = []
#         # sample_neighbors_fused function requires multithreading to be more efficient
#         # than sample_neighbors

#         self.Toptim =  int(g.number_of_nodes() / max(self.amplified_fanouts))

#         # self.cycle += 1
#         if(self.cycle%self.T == 0):
#             self.cache_refresh(exclude_eids=exclude_eids) # refresh cache every T cycles

#         self.cycle += 1

#         if self.fused and get_num_threads() > 1:
#             cpu = F.device_type(g.device) == "cpu"
#             if isinstance(seed_nodes, dict):
#                 for ntype in list(seed_nodes.keys()):
#                     print("seed dict",seed_nodes.keys)
#                     if not cpu:
#                         break
#                     cpu = (
#                         cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
#                     )
#             else:
#                 cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
#             if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
#                 if self.g != g:
#                     self.mapping = {}
#                     self.g = g
#                 for fanout in reversed(self.fanouts):
#                     block = self.cached_structure[k].sample_neighbors_fused(
#                         seed_nodes,
#                         fanout,
#                         edge_dir=self.edge_dir,
#                         prob=self.prob,
#                         replace=self.replace,
#                         exclude_edges=exclude_eids,
#                         mapping=self.mapping,
#                     )
#                     seed_nodes = block.srcdata[NID]
#                     blocks.insert(0, block)
#                 return seed_nodes, output_nodes, blocks
        
#         k = len(self.fanouts)-1
#         for fanout in reversed(self.fanouts):
#             print("seeds nodes:",seed_nodes)
#             print("org g:",g)
#             frontier = self.cached_structure[k].sample_neighbors(
#                 seed_nodes,
#                 fanout,
#                 edge_dir=self.edge_dir,
#                 prob=self.prob,
#                 replace=self.replace,
#                 output_device=self.output_device,
#                 exclude_edges=exclude_eids,
#             )
#             k-=1
#             print("sampled frontier:",frontier)
#             block = to_block(frontier, seed_nodes)
#             # If sampled from graphbolt-backed DistGraph, `EID` may not be in
#             # the block.
#             if EID in frontier.edata.keys():
#                 print("--------in this EID code---------")
#                 block.edata[EID] = frontier.edata[EID]
#             seed_nodes = block.srcdata[NID]
#             blocks.insert(0, block)

#         return seed_nodes, output_nodes, blocks

class NeighborSampler_FCR_struct_shared_cache_hete(BlockSampler):    
    def __init__(
            self, 
            g,
            fanouts, 
            edge_dir='in', 
            alpha=2, 
            T=20,
            hete_label=None,
            prob=None,
            mask=None,
            replace=False,
            prefetch_node_feats=None,
            prefetch_labels=None,
            prefetch_edge_feats=None,
            output_device=None,
            fused=True,
        ):

        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.g = g
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}

        self.alpha = alpha
        self.cycle = 0  # Initialize sampling cycle counter
        self.sc_size = max([f * alpha for f in fanouts])  # shared cache_storage size
        self.T = T
        self.Toptim = None # int(self.g.number_of_nodes() / max(self.amplified_fanouts))
        self.shared_cache = None  # Initialize cache structure
        self.hete_label = hete_label
        self.cache_refresh()  # Pre-sample and populate the cache
        self.Toptim = int(self.g.num_nodes(self.hete_label)/ self.sc_size )

    def cache_refresh(self,exclude_eids=None):
        """
        Pre-samples neighborhoods with amplified fanouts and refreshes the cache. This method
        is automatically called upon initialization and after every T sampling iterations to
        ensure that the cache is periodically updated with fresh samples.
        """
        self.shared_cache = self.g.sample_neighbors(
            {self.hete_label:list(range(0, self.g.num_nodes(self.hete_label)))},
            self.sc_size,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=exclude_eids
        )

    def sample_blocks(self, g,seed_nodes, exclude_eids=None):
        """
        Samples blocks from the graph for the specified seed nodes using the cache.

        Parameters
        ----------
        seed_nodes : Tensor
            The nodes for which the neighborhoods are to be sampled.

        Returns
        -------
        tuple
            A tuple containing the seed nodes for the next layer, the output nodes, and
            the list of blocks sampled from the graph.
        """
        output_nodes = seed_nodes

        # refresh cache after a period of time for generalization
        if self.cycle % self.Toptim == 0:
            self.cache_refresh()  # Refresh cache every T cycles
        
        self.cycle += 1

        blocks = []

        if self.fused and get_num_threads() > 1:
            # print("fused")
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                for ntype in list(seed_nodes.keys()):
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = self.g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks

        for k in range(len(self.fanouts)-1,-1,-1):
            frontier = self.shared_cache.sample_neighbors(
                seed_nodes,
                self.fanouts[k],
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids
            )

            # Sample frontier from the cache for acceleration
            block = to_block(frontier, seed_nodes)
            if EID in frontier.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier.edata[EID]
            blocks.insert(0, block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        # output_nodes = seed_nodes
        return seed_nodes, output_nodes, blocks

# class NeighborSampler_OTF_struct_FSCRFCF_hete(BlockSampler):
    
#     def __init__(self, g, 
#                 fanouts, 
#                 edge_dir='in', 
#                 amp_rate=1.5, # cache amplification rate (should be bigger than 1 --> to sample for multiple time)
#                 refresh_rate=0.4, #propotion of cache to be refresh, should be a positive float smaller than 0.5
#                 T=100, # refresh time
#                 hete_label = None,
#                 prob=None, 
#                 replace=False, 
#                 output_device=None, 
#                 exclude_eids=None,
#                 mask=None,
#                 prefetch_node_feats=None,
#                 prefetch_labels=None,
#                 prefetch_edge_feats=None,
#                 fused=True,
#                  ):
#         super().__init__(
#             prefetch_node_feats=prefetch_node_feats,
#             prefetch_labels=prefetch_labels,
#             prefetch_edge_feats=prefetch_edge_feats,
#             output_device=output_device,
#         )
#         self.g = g
#         self.fanouts = fanouts
#         self.edge_dir = edge_dir
#         self.amp_rate = amp_rate
#         self.refresh_rate = refresh_rate
#         self.replace = replace
#         self.output_device = output_device
#         self.exclude_eids = exclude_eids
#         self.cycle = 0

#         if mask is not None and prob is not None:
#             raise ValueError(
#                 "Mask and probability arguments are mutually exclusive. "
#                 "Consider multiplying the probability with the mask "
#                 "to achieve the same goal."
#             )
#         self.prob = prob or mask
#         self.fused = fused
#         self.mapping = {}
#         self.cache_size = [int(fanout * amp_rate) for fanout in fanouts]
#         self.T = T
#         self.hete_label = hete_label
#         self.Toptim = int(self.g.num_nodes(self.hete_label) / (max(self.cache_size))*self.amp_rate)
#         self.cached_graph_structures = [self.initialize_cache(cache_size) for cache_size in self.cache_size]

#     def initialize_cache(self, fanout_cache_storage):
#         """
#         Initializes the cache for each layer with an amplified fanout to pre-sample a larger
#         set of neighbors. This pre-sampling helps in reducing the need for dynamic sampling 
#         at every iteration, thereby improving efficiency.
#         """
#         cached_graph = self.g.sample_neighbors(
#             # torch.arange(0, self.g.number_of_nodes()),
#             {self.hete_label:list(range(0, self.g.num_nodes(self.hete_label)))},
#             fanout_cache_storage,
#             edge_dir=self.edge_dir,
#             prob=self.prob,
#             replace=self.replace,
#             output_device=self.output_device,
#             exclude_edges=self.exclude_eids,
#         )
#         print("end init cache")
#         return cached_graph

#     def refresh_cache(self,layer_id, cached_graph_structure, fanout_cache_refresh):
#         """
#         Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
#         cached edges with new samples from the graph. This method ensures the cache remains 
#         relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
#         """
#         fanout_cache_sample = self.cache_size[layer_id]-fanout_cache_refresh
#         cache_remain = cached_graph_structure.sample_neighbors(
#             # torch.arange(0, self.g.number_of_nodes()),
#             {self.hete_label:list(range(0, self.g.num_nodes(self.hete_label)))},
#             fanout_cache_sample,
#             edge_dir=self.edge_dir,
#             prob=self.prob,
#             replace=self.replace,
#             output_device=self.output_device,
#             exclude_edges=self.exclude_eids,
#         )

#         disk_to_add = self.g.sample_neighbors(
#             # torch.arange(0, self.g.number_of_nodes()),
#             {self.hete_label:list(range(0, self.g.num_nodes(self.hete_label)))},
#             fanout_cache_refresh,
#             edge_dir=self.edge_dir,
#             prob=self.prob,
#             replace=self.replace,
#             output_device=self.output_device,
#             exclude_edges=self.exclude_eids,
#         )

#         refreshed_cache = dgl.merge([cache_remain, disk_to_add])
#         print("end refresh cache")
#         return refreshed_cache

#     def sample_blocks(self, g, seed_nodes, exclude_eids=None):
#         """
#         Samples blocks for GNN layers by combining cached samples with dynamically sampled 
#         neighbors. This method also partially refreshes the cache based on specified parameters 
#         to balance between sampling efficiency and the freshness of the samples.
#         """
#         blocks = []
#         output_nodes = seed_nodes
#         self.cycle += 1
#         if((self.cycle % self.Toptim)==0):
#             for i in range(0,len(self.cached_graph_structures)):
#                 # Refresh cache partially
#                 fanout_cache_refresh = int(self.cache_size[i] * self.refresh_rate)
#                 self.cached_graph_structures[i]=self.refresh_cache(i, self.cached_graph_structures[i], fanout_cache_refresh)
            
#         for i, (fanout, cached_graph_structure) in enumerate(zip(reversed(self.fanouts), reversed(self.cached_graph_structures))):            
#             # Sample from cache
#             frontier_from_cache = self.cached_graph_structures[i].sample_neighbors(
#                 seed_nodes,
#                 #fanout_cache_retrieval,
#                 fanout,
#                 edge_dir=self.edge_dir,
#                 prob=self.prob,
#                 replace=self.replace,
#                 output_device=self.output_device,
#                 exclude_edges=self.exclude_eids,
#             )

#             # Convert the merged frontier to a block
#             block = to_block(frontier_from_cache, seed_nodes)
#             if EID in frontier_from_cache.edata.keys():
#                 print("--------in this EID code---------")
#                 block.edata[EID] = frontier_from_cache.edata[EID]
#             blocks.append(block)
#             seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer
        
#         return seed_nodes,output_nodes, blocks

class NeighborSampler_OTF_struct_FSCRFCF_shared_cache_hete(BlockSampler):
    
    def __init__(self, g, 
                fanouts, 
                edge_dir='in', 
                amp_rate=1.5, # cache amplification rate (should be bigger than 1 --> to sample for multiple time)
                refresh_rate=0.4, #propotion of cache to be refresh, should be a positive float smaller than 0.5
                T=100, # refresh time
                hete_label = None,
                prob=None, 
                replace=False, 
                output_device=None, 
                exclude_eids=None,
                mask=None,
                prefetch_node_feats=None,
                prefetch_labels=None,
                prefetch_edge_feats=None,
                fused=True,
                 ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.g = g
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.amp_rate = amp_rate
        self.refresh_rate = refresh_rate
        self.replace = replace
        self.output_device = output_device
        self.exclude_eids = exclude_eids
        self.cycle = 0

        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.fused = fused
        self.mapping = {}
        self.amp_cache_size = [fanout * amp_rate for fanout in fanouts]
        self.hete_label = hete_label
        # self.Toptim = int(self.g.number_of_nodes() / (max(self.amp_cache_size))*self.amp_rate)
        self.T = T
        # self.cached_graph_structures = [self.initialize_cache(cache_size) for cache_size in self.cache_size]

        self.shared_cache_size = max(self.amp_cache_size)
        self.shared_cache = self.initialize_cache(self.shared_cache_size)
        self.Toptim = int(self.g.num_nodes(self.hete_label) / (self.shared_cache_size*self.amp_rate))

    def initialize_cache(self, fanout_cache_storage):
        """
        Initializes the cache for each layer with an amplified fanout to pre-sample a larger
        set of neighbors. This pre-sampling helps in reducing the need for dynamic sampling 
        at every iteration, thereby improving efficiency.
        """
        cached_graph = self.g.sample_neighbors(
            # torch.arange(0, self.g.number_of_nodes()),
            {self.hete_label:list(range(0, self.g.num_nodes(self.hete_label)))},
            fanout_cache_storage,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
            # mappings=self.mapping
        )
        print("end init cache")
        return cached_graph

    def refresh_cache(self, fanout_cache_refresh):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        fanout_cache_sample = self.shared_cache_size-fanout_cache_refresh
        cache_remain = self.shared_cache.sample_neighbors(
            # torch.arange(0, self.g.number_of_nodes()),
            {self.hete_label:list(range(0, self.g.num_nodes(self.hete_label)))},
            fanout_cache_sample,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )

        disk_to_add = self.g.sample_neighbors(
            # torch.arange(0, self.g.number_of_nodes()),
            {self.hete_label:list(range(0, self.g.num_nodes(self.hete_label)))},
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )

        self.shared_cache = dgl.merge([cache_remain, disk_to_add])
        del cache_remain
        del disk_to_add
        print("end refresh cache")

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        """
        Samples blocks for GNN layers by combining cached samples with dynamically sampled 
        neighbors. This method also partially refreshes the cache based on specified parameters 
        to balance between sampling efficiency and the freshness of the samples.
        """
        self.cycle += 1
        blocks = []
        output_nodes = seed_nodes
        if((self.cycle % self.Toptim)==0):
            # Refresh cache partially
            fanout_cache_refresh = int(self.shared_cache_size * self.refresh_rate)
            self.refresh_cache(fanout_cache_refresh)
            
        for i, (fanout) in enumerate(reversed(self.fanouts)):            
            # Sample from cache
            frontier_from_cache = self.shared_cache.sample_neighbors(
                seed_nodes,
                #fanout_cache_retrieval,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=self.exclude_eids,
            )

            # Convert the merged frontier to a block
            block = to_block(frontier_from_cache, seed_nodes)
            if EID in frontier_from_cache.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier_from_cache.edata[EID]
            blocks.append(block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer
        
        return seed_nodes,output_nodes, blocks
    

class NeighborSampler_OTF_refresh_struct_hete(BlockSampler):
    def __init__(
            self, 
            g,
            fanouts, 
            edge_dir='in', 
            alpha=2, 
            T=20,
            refresh_rate=0.4,
            hete_label=None,
            prob=None,
            mask=None,
            replace=False,
            prefetch_node_feats=None,
            prefetch_labels=None,
            prefetch_edge_feats=None,
            output_device=None,
            fused=True,
        ):

        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.g = g
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}

        self.alpha = alpha
        self.cycle = 0  # Initialize sampling cycle counter
        self.cache_size = [f * alpha for f in fanouts]  # Amplified fanouts for pre-sampling
        self.refresh_rate = refresh_rate
        self.T = T
        self.Toptim = None # int(self.g.number_of_nodes() / max(self.amplified_fanouts))
        # self.cache_struct = []  # Initialize cache structure
        self.hete_label = hete_label
        # self.cache_refresh(self.g)  # Pre-sample and populate the cache
        self.cached_struct = [self.initialize_cache(cache_size) for cache_size in self.cache_size]
    
    def initialize_cache(self, fanout_cache_storage, exclude_eids=None):
        """
        Initializes the cache for each layer with an amplified fanout to pre-sample a larger
        set of neighbors. This pre-sampling helps in reducing the need for dynamic sampling 
        at every iteration, thereby improving efficiency.
        """
        cached_graph = self.g.sample_neighbors(
            # torch.arange(0, self.g.number_of_nodes()),
            {self.hete_label:list(range(0, self.g.num_nodes(self.hete_label)))},
            fanout_cache_storage,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=exclude_eids,
        )
        print("end init cache")
        return cached_graph

    def refresh_cache(self,layer_id,fanout_cache_refresh,exclude_eids=None):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        fanout_cache_sample = self.cache_size[layer_id]-fanout_cache_refresh
        cache_remain = self.cached_struct[layer_id].sample_neighbors(
            # torch.arange(0, self.g.number_of_nodes()),
            {self.hete_label:list(range(0, self.g.num_nodes(self.hete_label)))},
            fanout_cache_sample,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=exclude_eids,
        )

        disk_to_add = self.g.sample_neighbors(
            # torch.arange(0, self.g.number_of_nodes()),
            {self.hete_label:list(range(0, self.g.num_nodes(self.hete_label)))},
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=exclude_eids,
        )

        refreshed_cache = dgl.merge([cache_remain, disk_to_add])
        print("end refresh cache")
        return refreshed_cache

    def sample_blocks(self, g,seed_nodes, exclude_eids=None):
        """
        Samples blocks from the graph for the specified seed nodes using the cache.

        Parameters
        ----------
        seed_nodes : Tensor
            The nodes for which the neighborhoods are to be sampled.

        Returns
        -------
        tuple
            A tuple containing the seed nodes for the next layer, the output nodes, and
            the list of blocks sampled from the graph.
        """
        output_nodes = seed_nodes

        # refresh cache after a period of time for generalization
        self.Toptim = int(g.number_of_nodes() / max(self.cache_size))

        self.cycle += 1
        # if self.cycle % self.Toptim == 0:
        #     self.refresh_cache(g)  # Refresh cache every T cycles
        if((self.cycle % self.Toptim)==0):
            for i in range(0,len(self.cached_struct)):
                # Refresh cache partially
                fanout_cache_refresh = int(self.cache_size[i] * self.refresh_rate)
                self.cached_struct[i]=self.refresh_cache(i, fanout_cache_refresh)

        blocks = []

        if self.fused and get_num_threads() > 1:
            # print("fused")
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                for ntype in list(seed_nodes.keys()):
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = self.g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks

        for k in range(len(self.cached_struct)-1,-1,-1):
            cached_structure = self.cached_struct[k]
            fanout = self.fanouts[k]
            frontier = cached_structure.sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids
            )

            # Sample frontier from the cache for acceleration
            block = to_block(frontier, seed_nodes)
            if EID in frontier.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier.edata[EID]
            blocks.insert(0, block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        # output_nodes = seed_nodes
        return seed_nodes, output_nodes, blocks
    

class NeighborSampler_OTF_refresh_struct_shared_cache_hete(BlockSampler):    
    def __init__(
            self, 
            g,
            fanouts, 
            edge_dir='in', 
            alpha=2, 
            T=20,
            refresh_rate=0.4,
            hete_label=None,
            prob=None,
            mask=None,
            replace=False,
            prefetch_node_feats=None,
            prefetch_labels=None,
            prefetch_edge_feats=None,
            output_device=None,
            fused=True,
        ):

        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.g = g
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}

        self.alpha = alpha
        self.cycle = 0  # Initialize sampling cycle counter
        self.sc_size = max([f * alpha for f in fanouts])  # Amplified fanouts for pre-sampling
        self.refresh_rate = refresh_rate
        self.T = T
        self.Toptim = None # int(self.g.number_of_nodes() / max(self.amplified_fanouts))
        # self.cache_struct = []  # Initialize cache structure
        self.hete_label = hete_label
        # self.cache_refresh(self.g)  # Pre-sample and populate the cache
        self.shared_cache = self.initialize_cache(self.sc_size)

    def initialize_cache(self, fanout_cache_storage, exclude_eids=None):
        """
        Initializes the cache for each layer with an amplified fanout to pre-sample a larger
        set of neighbors. This pre-sampling helps in reducing the need for dynamic sampling 
        at every iteration, thereby improving efficiency.
        """
        cached_graph = self.g.sample_neighbors(
            # torch.arange(0, self.g.number_of_nodes()),
            {self.hete_label:list(range(0, self.g.num_nodes(self.hete_label)))},
            fanout_cache_storage,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=exclude_eids,
        )
        print("end init cache")
        return cached_graph

    def refresh_cache(self,fanout_cache_refresh,exclude_eids=None):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        fanout_cache_sample = self.sc_size-fanout_cache_refresh
        cache_remain = self.shared_cache.sample_neighbors(
            # torch.arange(0, self.g.number_of_nodes()),
            {self.hete_label:list(range(0, self.g.num_nodes(self.hete_label)))},
            fanout_cache_sample,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=exclude_eids,
        )

        disk_to_add = self.g.sample_neighbors(
            # torch.arange(0, self.g.number_of_nodes()),
            {self.hete_label:list(range(0, self.g.num_nodes(self.hete_label)))},
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=exclude_eids,
        )

        refreshed_cache = dgl.merge([cache_remain, disk_to_add])
        print("end refresh cache")
        return refreshed_cache

    def sample_blocks(self, g,seed_nodes, exclude_eids=None):
        """
        Samples blocks from the graph for the specified seed nodes using the cache.

        Parameters
        ----------
        seed_nodes : Tensor
            The nodes for which the neighborhoods are to be sampled.

        Returns
        -------
        tuple
            A tuple containing the seed nodes for the next layer, the output nodes, and
            the list of blocks sampled from the graph.
        """
        output_nodes = seed_nodes

        # refresh cache after a period of time for generalization
        self.Toptim = int(g.number_of_nodes() / self.sc_size)

        self.cycle += 1
        # if self.cycle % self.Toptim == 0:
        #     self.refresh_cache(g)  # Refresh cache every T cycles
        if((self.cycle % self.Toptim)==0):
            fanout_cache_refresh = int(self.sc_size * self.refresh_rate)
            self.shared_cache=self.refresh_cache(fanout_cache_refresh)

        blocks = []

        if self.fused and get_num_threads() > 1:
            # print("fused")
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                for ntype in list(seed_nodes.keys()):
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = self.g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks

        for k in range(len(self.fanouts)-1,-1,-1):
            cached_structure = self.shared_cache
            fanout = self.fanouts[k]
            frontier = cached_structure.sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids
            )

            # Sample frontier from the cache for acceleration
            block = to_block(frontier, seed_nodes)
            if EID in frontier.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier.edata[EID]
            blocks.insert(0, block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        # output_nodes = seed_nodes
        return seed_nodes, output_nodes, blocks


# class NeighborSampler_OTF_fetch_struct_shared_cache_hete(BlockSampler):
    
#     def __init__(self, g, 
#                 fanouts, 
#                 edge_dir='in', 
#                 amp_rate=1.5, # cache amplification rate (should be bigger than 1 --> to sample for multiple time)
#                 fetch_rate=0.4, #propotion of cache to be fetch from cache, should be a positive float smaller than 0.5
#                 T_fetch=3, # fetch period of time
#                 T_refresh=None, # refresh time
#                 hete_label=None,
#                 prob=None, 
#                 replace=False, 
#                 output_device=None, 
#                 exclude_eids=None,
#                 mask=None,
#                 prefetch_node_feats=None,
#                 prefetch_labels=None,
#                 prefetch_edge_feats=None,
#                 fused=True,
#                  ):
#         super().__init__(
#             prefetch_node_feats=prefetch_node_feats,
#             prefetch_labels=prefetch_labels,
#             prefetch_edge_feats=prefetch_edge_feats,
#             output_device=output_device,
#         )
#         self.g = g
#         self.fanouts = fanouts
#         self.edge_dir = edge_dir
#         self.amp_rate = amp_rate
#         self.fetch_rate = fetch_rate
#         self.hete_label = hete_label
#         self.replace = replace
#         self.output_device = output_device
#         self.exclude_eids = exclude_eids

#         if mask is not None and prob is not None:
#             raise ValueError(
#                 "Mask and probability arguments are mutually exclusive. "
#                 "Consider multiplying the probability with the mask "
#                 "to achieve the same goal."
#             )
#         self.prob = prob or mask
#         self.fused = fused
#         self.mapping = {}
#         self.amp_cache_size = [fanout * amp_rate for fanout in fanouts]
#         if T_refresh!=None:
#             self.T_refresh = T_refresh
#         else:
#             self.T_refresh = int(self.g.number_of_nodes()/max(self.fanouts) *self.amp_rate)
#         self.T_fetch = T_fetch
#         # self.cached_graph_structures = None
#         self.cycle = 0

#         self.shared_cache_size = max(self.amp_cache_size)
#         self.shared_cache = self.full_cache_refresh(self.shared_cache_size)
#         self.hete_label = hete_label

#     def full_cache_refresh(self, fanout_cache_storage):
#         cached_graph = self.g.sample_neighbors(
#             # torch.arange(0, self.g.number_of_nodes()),
#             {self.hete_label:list(range(0, self.g.num_nodes(self.hete_label)))},
#             fanout_cache_storage,
#             edge_dir=self.edge_dir,
#             prob=self.prob,
#             replace=self.replace,
#             output_device=self.output_device,
#             exclude_edges=self.exclude_eids,
#         )
#         print("cache refresh")
#         return cached_graph

#     def OTF_fetch(self,layer_id,  seed_nodes, fanout_cache_fetch):
#         print("OTF fetch cache")
#         if(fanout_cache_fetch==self.fanouts[layer_id]):
#             cache_fetch = self.shared_cache.sample_neighbors(
#             seed_nodes,
#             fanout_cache_fetch,
#             edge_dir=self.edge_dir,
#             prob=self.prob,
#             replace=self.replace,
#             output_device=self.output_device,
#             exclude_edges=self.exclude_eids,
#             )
#             return cache_fetch
#         else:
#             fanout_disk_fetch = self.fanouts[layer_id]-fanout_cache_fetch
#             cache_fetch = self.shared_cache.sample_neighbors(
#                 seed_nodes,
#                 fanout_cache_fetch,
#                 edge_dir=self.edge_dir,
#                 prob=self.prob,
#                 replace=self.replace,
#                 output_device=self.output_device,
#                 exclude_edges=self.exclude_eids,
#             )

#             disk_fetch = self.g.sample_neighbors(
#                 seed_nodes,
#                 fanout_disk_fetch,
#                 edge_dir=self.edge_dir,
#                 prob=self.prob,
#                 replace=self.replace,
#                 output_device=self.output_device,
#                 exclude_edges=self.exclude_eids,
#             )

#             OTF_fetch_res = dgl.merge([cache_fetch, disk_fetch])
#             return OTF_fetch_res

#     def sample_blocks(self, g, seed_nodes, exclude_eids=None):
#         blocks = []
#         output_nodes = seed_nodes

#         self.cycle += 1
#         print("self.T_refresh=",self.T_refresh)
#         # refresh full cache after a period of time
#         if((self.cycle%self.T_refresh)==0):
#             self.shared_cache = self.full_cache_refresh(self.shared_cache_size)
#             # self.cached_graph_structures = [self.full_cache_refresh(cache_size) for cache_size in self.cache_size]
        
#         for i, (fanout) in enumerate(reversed(self.fanouts)):
#             fanout_cache_fetch = int(fanout * self.fetch_rate)

#             # fetch cache partially
#             if((self.cycle%self.T_fetch)==0):
#                 frontier_OTF = self.OTF_fetch(i, seed_nodes, fanout_cache_fetch)
#             else:
#                 #frontier_OTF = self.OTF_fetch(i, seed_nodes, self.fanouts[i])
#                 frontier_OTF = self.shared_cache.sample_neighbors(
#                     seed_nodes,
#                     fanout,
#                     edge_dir=self.edge_dir,
#                     prob=self.prob,
#                     replace=self.replace,
#                     output_device=self.output_device,
#                     exclude_edges=self.exclude_eids,
#                 )
            
#             # Convert the merged frontier to a block
#             block = to_block(frontier_OTF, seed_nodes)
#             if EID in frontier_OTF.edata.keys():
#                 print("--------in this EID code---------")
#                 block.edata[EID] = frontier_OTF.edata[EID]
#             blocks.append(block)
#             seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

#         return seed_nodes,output_nodes, blocks
    
class NeighborSampler_OTF_fetch_struct_hete(BlockSampler):    
    def __init__(
            self, 
            g,
            fanouts, 
            edge_dir='in', 
            amp_rate=2, 
            fetch_rate = 0.4,
            T_refresh=None,
            T_fetch=3, # fetch period of time
            hete_label=None,
            prob=None,
            mask=None,
            replace=False,
            prefetch_node_feats=None,
            prefetch_labels=None,
            prefetch_edge_feats=None,
            output_device=None,
            fused=True,
        ):

        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.g = g
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}
        self.exclude_eids = None

        self.alpha = amp_rate
        self.cycle = 0  # Initialize sampling cycle counter
        self.cache_size = [f * amp_rate for f in fanouts]  # Amplified fanouts for pre-sampling
        if T_refresh!=None:
            self.T_refresh = T_refresh
        else:
            self.T_refresh = int(self.g.number_of_nodes()/max(self.fanouts) *self.amp_rate)
        self.T_fetch = T_fetch
        self.Toptim = None # int(self.g.number_of_nodes() / max(self.amplified_fanouts))
        self.fetch_rate = fetch_rate
        # self.cache_struct = []  # Initialize cache structure
        self.hete_label = hete_label
        # self.cache_refresh(self.g)  # Pre-sample and populate the cache
        self.cache_struct = [self.full_cache_refresh(cache_size) for cache_size in self.cache_size]
    
    def full_cache_refresh(self, fanout_cache_storage, exclude_eids = None):
        cached_graph = self.g.sample_neighbors(
            # torch.arange(0, self.g.number_of_nodes()),
            {self.hete_label:list(range(0, self.g.num_nodes(self.hete_label)))},
            fanout_cache_storage,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=exclude_eids,
        )
        print("cache refresh")
        return cached_graph

    def OTF_fetch(self,layer_id,  seed_nodes, fanout_cache_fetch, exclude_eids = None):
        print("OTF fetch cache")
        if(fanout_cache_fetch==self.fanouts[layer_id]):
            cache_fetch = self.cache_struct[layer_id].sample_neighbors(
            seed_nodes,
            fanout_cache_fetch,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=exclude_eids,
            )
            return cache_fetch
        else:
            fanout_disk_fetch = self.fanouts[layer_id]-fanout_cache_fetch
            cache_fetch = self.cache_struct[layer_id].sample_neighbors(
                seed_nodes,
                fanout_cache_fetch,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=self.exclude_eids,
            )

            disk_fetch = self.g.sample_neighbors(
                seed_nodes,
                fanout_disk_fetch,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=self.exclude_eids,
            )

            OTF_fetch_res = dgl.merge([cache_fetch, disk_fetch])
            return OTF_fetch_res

    def sample_blocks(self, g,seed_nodes, exclude_eids=None):
        """
        Samples blocks from the graph for the specified seed nodes using the cache.

        Parameters
        ----------
        seed_nodes : Tensor
            The nodes for which the neighborhoods are to be sampled.

        Returns
        -------
        tuple
            A tuple containing the seed nodes for the next layer, the output nodes, and
            the list of blocks sampled from the graph.
        """
        output_nodes = seed_nodes
        
        self.cycle += 1
        print("self.T_refresh=",self.T_refresh)
        # refresh full cache after a period of time
        if((self.cycle%self.T_refresh)==0):
            for i in range(len(self.fanouts)):
                self.cache_struct[i] = self.full_cache_refresh(self.cache_size[i])

        blocks = []

        if self.fused and get_num_threads() > 1:
            # print("fused")
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                for ntype in list(seed_nodes.keys()):
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = self.g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks

        for k in range(len(self.fanouts)-1,-1,-1):
            fanout = self.fanouts[k]

            fanout_cache_fetch = int(fanout * self.fetch_rate)

            # fetch cache partially
            if((self.cycle%self.T_fetch)==0):
                frontier_OTF = self.OTF_fetch(k, seed_nodes, fanout_cache_fetch)
            else:
                #frontier_OTF = self.OTF_fetch(i, seed_nodes, self.fanouts[i])
                frontier_OTF = self.cache_struct[k].sample_neighbors(
                    seed_nodes,
                    fanout,
                    edge_dir=self.edge_dir,
                    prob=self.prob,
                    replace=self.replace,
                    output_device=self.output_device,
                    exclude_edges=self.exclude_eids,
                )

            # Sample frontier from the cache for acceleration
            block = to_block(frontier_OTF, seed_nodes)
            if EID in frontier_OTF.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier_OTF.edata[EID]
            blocks.insert(0, block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        # output_nodes = seed_nodes
        return seed_nodes, output_nodes, blocks

class NeighborSampler_OTF_fetch_struct_shared_cache_hete(BlockSampler):    
    def __init__(
            self, 
            g,
            fanouts, 
            edge_dir='in', 
            amp_rate=2, 
            fetch_rate = 0.4,
            T_refresh=None,
            T_fetch=3, # fetch period of time
            hete_label=None,
            prob=None,
            mask=None,
            replace=False,
            prefetch_node_feats=None,
            prefetch_labels=None,
            prefetch_edge_feats=None,
            output_device=None,
            fused=True,
        ):

        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.g = g
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}
        self.exclude_eids = None

        self.alpha = amp_rate
        self.cycle = 0  # Initialize sampling cycle counter
        self.sc_size = max([f * amp_rate for f in fanouts])  # Amplified fanouts for pre-sampling
        if T_refresh!=None:
            self.T_refresh = T_refresh
        else:
            self.T_refresh = int(self.g.number_of_nodes()/max(self.fanouts) *self.amp_rate)
        self.T_fetch = T_fetch
        self.Toptim = None # int(self.g.number_of_nodes() / max(self.amplified_fanouts))
        self.fetch_rate = fetch_rate
        # self.cache_struct = []  # Initialize cache structure
        self.hete_label = hete_label
        # self.cache_refresh(self.g)  # Pre-sample and populate the cache
        self.shared_cache = self.full_cache_refresh(self.sc_size)
    
    def full_cache_refresh(self, fanout_cache_storage, exclude_eids = None):
        cached_graph = self.g.sample_neighbors(
            # torch.arange(0, self.g.number_of_nodes()),
            {self.hete_label:list(range(0, self.g.num_nodes(self.hete_label)))},
            fanout_cache_storage,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=exclude_eids,
        )
        print("cache refresh")
        return cached_graph

    def OTF_fetch(self,layer_id,  seed_nodes, fanout_cache_fetch, exclude_eids = None):
        print("OTF fetch cache")
        if(fanout_cache_fetch==self.fanouts[layer_id]):
            cache_fetch = self.shared_cache.sample_neighbors(
            seed_nodes,
            fanout_cache_fetch,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=exclude_eids,
            )
            return cache_fetch
        else:
            fanout_disk_fetch = self.fanouts[layer_id]-fanout_cache_fetch
            cache_fetch = self.shared_cache.sample_neighbors(
                seed_nodes,
                fanout_cache_fetch,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=self.exclude_eids,
            )

            disk_fetch = self.g.sample_neighbors(
                seed_nodes,
                fanout_disk_fetch,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=self.exclude_eids,
            )

            OTF_fetch_res = dgl.merge([cache_fetch, disk_fetch])
            return OTF_fetch_res

    def sample_blocks(self, g,seed_nodes, exclude_eids=None):
        """
        Samples blocks from the graph for the specified seed nodes using the cache.

        Parameters
        ----------
        seed_nodes : Tensor
            The nodes for which the neighborhoods are to be sampled.

        Returns
        -------
        tuple
            A tuple containing the seed nodes for the next layer, the output nodes, and
            the list of blocks sampled from the graph.
        """
        output_nodes = seed_nodes
        
        self.cycle += 1
        print("self.T_refresh=",self.T_refresh)
        # refresh full cache after a period of time
        if((self.cycle%self.T_refresh)==0):
            self.shared_cache = self.full_cache_refresh(self.sc_size)

        blocks = []

        if self.fused and get_num_threads() > 1:
            # print("fused")
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                for ntype in list(seed_nodes.keys()):
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = self.g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks

        for k in range(len(self.fanouts)-1,-1,-1):
            fanout = self.fanouts[k]

            fanout_cache_fetch = int(fanout * self.fetch_rate)

            # fetch cache partially
            if((self.cycle%self.T_fetch)==0):
                frontier_OTF = self.OTF_fetch(k, seed_nodes, fanout_cache_fetch)
            else:
                #frontier_OTF = self.OTF_fetch(i, seed_nodes, self.fanouts[i])
                frontier_OTF = self.shared_cache.sample_neighbors(
                    seed_nodes,
                    fanout,
                    edge_dir=self.edge_dir,
                    prob=self.prob,
                    replace=self.replace,
                    output_device=self.output_device,
                    exclude_edges=self.exclude_eids,
                )

            # Sample frontier from the cache for acceleration
            block = to_block(frontier_OTF, seed_nodes)
            if EID in frontier_OTF.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier_OTF.edata[EID]
            blocks.insert(0, block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        # output_nodes = seed_nodes
        return seed_nodes, output_nodes, blocks


# class NeighborSampler_OTF_struct_PCFPSCR_hete(BlockSampler):
    
#     def __init__(self, g, 
#                 fanouts, 
#                 edge_dir='in', 
#                 amp_rate=1.5, # cache amplification rate (should be bigger than 1 --> to sample for multiple time)
#                 refresh_rate=0.4, #propotion of cache to be refresh, should be a positive float smaller than 0.5
#                 T=50, # refresh time, for example
#                 hete_label = None,
#                 prob=None, 
#                 replace=False, 
#                 output_device=None, 
#                 exclude_eids=None,
#                 mask=None,
#                 prefetch_node_feats=None,
#                 prefetch_labels=None,
#                 prefetch_edge_feats=None,
#                 fused=True,
#                  ):
#         super().__init__(
#             prefetch_node_feats=prefetch_node_feats,
#             prefetch_labels=prefetch_labels,
#             prefetch_edge_feats=prefetch_edge_feats,
#             output_device=output_device,
#         )
#         self.g = g
#         self.fanouts = fanouts
#         self.edge_dir = edge_dir
#         self.amp_rate = amp_rate
#         self.refresh_rate = refresh_rate
#         self.hete_label = hete_label
#         self.replace = replace
#         self.output_device = output_device
#         self.exclude_eids = exclude_eids

#         if mask is not None and prob is not None:
#             raise ValueError(
#                 "Mask and probability arguments are mutually exclusive. "
#                 "Consider multiplying the probability with the mask "
#                 "to achieve the same goal."
#             )
#         self.prob = prob or mask
#         self.fused = fused
#         self.mapping = {}
#         self.cache_size = [fanout * amp_rate for fanout in fanouts]
#         self.T = T
#         self.cached_graph_structures = [self.initialize_cache(cache_size) for cache_size in self.cache_size]

#     def initialize_cache(self, fanout_cache_storage):
#         """
#         Initializes the cache for each layer with an amplified fanout to pre-sample a larger
#         set of neighbors. This pre-sampling helps in reducing the need for dynamic sampling 
#         at every iteration, thereby improving efficiency.
#         """
#         cached_graph = self.g.sample_neighbors(
#             # torch.arange(0, self.g.number_of_nodes()),
#             {self.hete_label:list(range(0, self.g.num_nodes(self.hete_label)))},
#             fanout_cache_storage,
#             edge_dir=self.edge_dir,
#             prob=self.prob,
#             replace=self.replace,
#             output_device=self.output_device,
#             exclude_edges=self.exclude_eids,
#         )
#         print("end init cache")
#         return cached_graph

#     def OTF_rf_cache(self,layer_id, cached_graph_structure, seed_nodes, fanout_cache_refresh, fanout):
#         """
#         Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
#         cached edges with new samples from the graph. This method ensures the cache remains 
#         relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
#         """
#         fanout_cache_remain = self.cache_size[layer_id]-fanout_cache_refresh
#         fanout_cache_pr = fanout-fanout_cache_refresh
#         # unchanged_nodes = range(torch.arange(0, self.g.number_of_nodes()))-seed_nodes
#         # the rest node structure remain the same
#         all_nodes = torch.arange(0,  self.g.num_nodes(self.hete_label))
#         print("seed nodes:",seed_nodes)
#         print("all nodes",all_nodes)
#         mask = ~torch.isin(all_nodes, seed_nodes[self.hete_label])
#         # bool mask to select those nodes do not in seed_nodes
#         unchanged_nodes = {self.hete_label: all_nodes[mask]}
#         unchanged_structure = cached_graph_structure.sample_neighbors(
#             unchanged_nodes,
#             self.cache_size[layer_id],
#             edge_dir=self.edge_dir,
#             prob=self.prob,
#             replace=self.replace,
#             output_device=self.output_device,
#             exclude_edges=self.exclude_eids,
#         )
#         # the OTF node structure should 
#         changed_cache_remain = cached_graph_structure.sample_neighbors(
#             seed_nodes,
#             fanout_cache_remain,
#             edge_dir=self.edge_dir,
#             prob=self.prob,
#             replace=self.replace,
#             output_device=self.output_device,
#             exclude_edges=self.exclude_eids,
#         )
#         cache_pr = cached_graph_structure.sample_neighbors(
#             seed_nodes,
#             fanout_cache_pr,
#             edge_dir=self.edge_dir,
#             prob=self.prob,
#             replace=self.replace,
#             output_device=self.output_device,
#             exclude_edges=self.exclude_eids,
#         )
#         changed_disk_to_add = self.g.sample_neighbors(
#             seed_nodes,
#             fanout_cache_refresh,
#             edge_dir=self.edge_dir,
#             prob=self.prob,
#             replace=self.replace,
#             output_device=self.output_device,
#             exclude_edges=self.exclude_eids,
#         )
#         refreshed_cache = dgl.merge([unchanged_structure, changed_cache_remain, changed_disk_to_add])
#         retrieval_cache = dgl.merge([cache_pr, changed_disk_to_add])
#         return refreshed_cache, retrieval_cache

#     def sample_blocks(self, g, seed_nodes, exclude_eids=None):
#         """
#         Samples blocks for GNN layers by combining cached samples with dynamically sampled 
#         neighbors. This method also partially refreshes the cache based on specified parameters 
#         to balance between sampling efficiency and the freshness of the samples.
#         """
#         blocks = []
#         output_nodes = seed_nodes
#         for i, (fanout, cached_graph_structure) in enumerate(zip(reversed(self.fanouts), reversed(self.cached_graph_structures))):
#             fanout_cache_refresh = int(fanout * self.refresh_rate)

#             # Refresh cache&disk partially, while retrieval cache&disk partially
#             self.cached_graph_structures[i], frontier_comp = self.OTF_rf_cache(i, cached_graph_structure, seed_nodes, fanout_cache_refresh, fanout)
            
#             # Convert the merged frontier to a block
#             block = to_block(frontier_comp, seed_nodes)
#             if EID in frontier_comp.edata.keys():
#                 print("--------in this EID code---------")
#                 block.edata[EID] = frontier_comp.edata[EID]
#             blocks.append(block)
#             seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

#         return seed_nodes,output_nodes, blocks


class NeighborSampler_OTF_struct_PCFPSCR_hete(BlockSampler):    
    def __init__(
            self, 
            g,
            fanouts, 
            edge_dir='in', 
            amp_rate=2, 
            T=20,
            refresh_rate=0.4,
            hete_label=None,
            prob=None,
            mask=None,
            replace=False,
            prefetch_node_feats=None,
            prefetch_labels=None,
            prefetch_edge_feats=None,
            output_device=None,
            fused=True,
        ):

        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.g = g
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}
        self.exclude_eids = None

        self.amp_rate = amp_rate
        self.hete_label = hete_label
        self.cycle = 0  # Initialize sampling cycle counter
        self.amplified_fanouts = [f * self.amp_rate for f in fanouts]  # Amplified fanouts for pre-sampling
        self.T = T
        self.refresh_rate = refresh_rate
        self.Toptim = None # int(self.g.number_of_nodes() / max(self.amplified_fanouts))
        self.cache_struct = [self.initialize_cache(fanout_cache_storage=ampf) for ampf in self.amplified_fanouts]  # Initialize cache structure
        # self.cache_refresh(self.g)  # Pre-sample and populate the cache

    def initialize_cache(self, fanout_cache_storage):
        """
        Initializes the cache for each layer with an amplified fanout to pre-sample a larger
        set of neighbors. This pre-sampling helps in reducing the need for dynamic sampling 
        at every iteration, thereby improving efficiency.
        """
        cached_graph = self.g.sample_neighbors(
            # torch.arange(0, self.g.number_of_nodes()),
            {self.hete_label:list(range(0, self.g.num_nodes(self.hete_label)))},
            fanout_cache_storage,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        print("end init cache")
        return cached_graph

    def OTF_rf_cache(self,layer_id, cached_graph_structure, seed_nodes, fanout_cache_refresh, fanout):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        fanout_cache_remain = self.amplified_fanouts[layer_id]-fanout_cache_refresh
        fanout_cache_pr = fanout-fanout_cache_refresh
        # unchanged_nodes = range(torch.arange(0, self.g.number_of_nodes()))-seed_nodes
        # the rest node structure remain the same
        all_nodes = torch.arange(0,  self.g.num_nodes(self.hete_label))
        print("seed nodes:",seed_nodes)
        print("all nodes",all_nodes)
        mask = ~torch.isin(all_nodes, seed_nodes[self.hete_label])
        # bool mask to select those nodes do not in seed_nodes
        unchanged_nodes = {self.hete_label: all_nodes[mask]}
        unchanged_structure = cached_graph_structure.sample_neighbors(
            unchanged_nodes,
            self.amplified_fanouts[layer_id],
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        # the OTF node structure should 
        changed_cache_remain = cached_graph_structure.sample_neighbors(
            seed_nodes,
            fanout_cache_remain,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        cache_pr = cached_graph_structure.sample_neighbors(
            seed_nodes,
            fanout_cache_pr,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        changed_disk_to_add = self.g.sample_neighbors(
            seed_nodes,
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        refreshed_cache = dgl.merge([unchanged_structure, changed_cache_remain, changed_disk_to_add])
        retrieval_cache = dgl.merge([cache_pr, changed_disk_to_add])
        return refreshed_cache, retrieval_cache

    def sample_blocks(self, g,seed_nodes, exclude_eids=None):
        """
        Samples blocks from the graph for the specified seed nodes using the cache.

        Parameters
        ----------
        seed_nodes : Tensor
            The nodes for which the neighborhoods are to be sampled.

        Returns
        -------
        tuple
            A tuple containing the seed nodes for the next layer, the output nodes, and
            the list of blocks sampled from the graph.
        """
        output_nodes = seed_nodes

        # # refresh cache after a period of time for generalization
        # self.Toptim = int(g.number_of_nodes() / max(self.amplified_fanouts))
        # if self.cycle % self.Toptim == 0:
        #     self.cache_refresh(g)  # Refresh cache every T cycles
        
        self.cycle += 1

        blocks = []

        if self.fused and get_num_threads() > 1:
            # print("fused")
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                for ntype in list(seed_nodes.keys()):
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks

        for k in range(len(self.cache_struct)-1,-1,-1):
            fanout_cache_refresh = int(self.fanouts[k] * self.refresh_rate)

            # Refresh cache&disk partially, while retrieval cache&disk partially
            self.cache_struct[k], frontier_comp = self.OTF_rf_cache(k, self.cache_struct[k], seed_nodes, fanout_cache_refresh, self.fanouts[k])

            # Sample frontier from the cache for acceleration
            block = to_block(frontier_comp, seed_nodes)
            if EID in frontier_comp.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier_comp.edata[EID]
            blocks.insert(0, block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        # output_nodes = seed_nodes
        return seed_nodes, output_nodes, blocks


class NeighborSampler_OTF_struct_PCFPSCR_shared_cache_hete(BlockSampler):    
    def __init__(
            self, 
            g,
            fanouts, 
            edge_dir='in', 
            amp_rate=2, 
            T=20,
            refresh_rate=0.4,
            hete_label=None,
            prob=None,
            mask=None,
            replace=False,
            prefetch_node_feats=None,
            prefetch_labels=None,
            prefetch_edge_feats=None,
            output_device=None,
            fused=True,
        ):

        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.g = g
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}
        self.exclude_eids = None

        self.amp_rate = amp_rate
        self.hete_label = hete_label
        self.cycle = 0  # Initialize sampling cycle counter
        self.sc_size = max([f * self.amp_rate for f in fanouts])  # Amplified fanouts for pre-sampling
        self.T = T
        self.refresh_rate = refresh_rate
        self.Toptim = None # int(self.g.number_of_nodes() / max(self.amplified_fanouts))
        self.shared_cache = self.initialize_cache(fanout_cache_storage=self.sc_size)  # Initialize cache structure
        # self.cache_refresh(self.g)  # Pre-sample and populate the cache

    def initialize_cache(self, fanout_cache_storage):
        """
        Initializes the cache for each layer with an amplified fanout to pre-sample a larger
        set of neighbors. This pre-sampling helps in reducing the need for dynamic sampling 
        at every iteration, thereby improving efficiency.
        """
        cached_graph = self.g.sample_neighbors(
            # torch.arange(0, self.g.number_of_nodes()),
            {self.hete_label:list(range(0, self.g.num_nodes(self.hete_label)))},
            fanout_cache_storage,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        print("end init cache")
        return cached_graph

    def OTF_rf_cache(self,seed_nodes, fanout_cache_refresh, fanout):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        fanout_cache_remain = self.sc_size-fanout_cache_refresh
        fanout_cache_pr = fanout-fanout_cache_refresh
        # unchanged_nodes = range(torch.arange(0, self.g.number_of_nodes()))-seed_nodes
        # the rest node structure remain the same
        all_nodes = torch.arange(0,  self.g.num_nodes(self.hete_label))
        print("seed nodes:",seed_nodes)
        print("all nodes",all_nodes)
        mask = ~torch.isin(all_nodes, seed_nodes[self.hete_label])
        # bool mask to select those nodes do not in seed_nodes
        unchanged_nodes = {self.hete_label: all_nodes[mask]}
        unchanged_structure = self.shared_cache.sample_neighbors(
            unchanged_nodes,
            self.sc_size,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        # the OTF node structure should 
        changed_cache_remain = self.shared_cache.sample_neighbors(
            seed_nodes,
            fanout_cache_remain,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        cache_pr = self.shared_cache.sample_neighbors(
            seed_nodes,
            fanout_cache_pr,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        changed_disk_to_add = self.g.sample_neighbors(
            seed_nodes,
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        refreshed_cache = dgl.merge([unchanged_structure, changed_cache_remain, changed_disk_to_add])
        retrieval_cache = dgl.merge([cache_pr, changed_disk_to_add])
        return refreshed_cache, retrieval_cache

    def sample_blocks(self, g,seed_nodes, exclude_eids=None):
        """
        Samples blocks from the graph for the specified seed nodes using the cache.

        Parameters
        ----------
        seed_nodes : Tensor
            The nodes for which the neighborhoods are to be sampled.

        Returns
        -------
        tuple
            A tuple containing the seed nodes for the next layer, the output nodes, and
            the list of blocks sampled from the graph.
        """
        output_nodes = seed_nodes

        # # refresh cache after a period of time for generalization
        # self.Toptim = int(g.number_of_nodes() / max(self.amplified_fanouts))
        # if self.cycle % self.Toptim == 0:
        #     self.cache_refresh(g)  # Refresh cache every T cycles
        
        self.cycle += 1

        blocks = []

        if self.fused and get_num_threads() > 1:
            # print("fused")
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                for ntype in list(seed_nodes.keys()):
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks

        for k in range(len(self.fanouts)-1,-1,-1):
            fanout_cache_refresh = int(self.fanouts[k] * self.refresh_rate)

            # Refresh cache&disk partially, while retrieval cache&disk partially
            self.shared_cache, frontier_comp = self.OTF_rf_cache( seed_nodes, fanout_cache_refresh, self.fanouts[k])

            # Sample frontier from the cache for acceleration
            block = to_block(frontier_comp, seed_nodes)
            if EID in frontier_comp.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier_comp.edata[EID]
            blocks.insert(0, block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        # output_nodes = seed_nodes
        return seed_nodes, output_nodes, blocks


class NeighborSampler_OTF_struct_PSCRFCF_hete(BlockSampler):    
    def __init__(
            self, 
            g,
            fanouts, 
            edge_dir='in', 
            amp_rate=2, 
            T=20,
            refresh_rate=0.4,
            hete_label=None,
            prob=None,
            mask=None,
            replace=False,
            prefetch_node_feats=None,
            prefetch_labels=None,
            prefetch_edge_feats=None,
            output_device=None,
            fused=True,
        ):

        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.g = g
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}
        self.exclude_eids = None

        self.amp_rate = amp_rate
        self.hete_label = hete_label
        self.cycle = 0  # Initialize sampling cycle counter
        self.cache_size = [f * self.amp_rate for f in fanouts]  # Amplified fanouts for pre-sampling
        self.T = T
        self.refresh_rate = refresh_rate
        self.Toptim = None # int(self.g.number_of_nodes() / max(self.amplified_fanouts))
        self.cache_struct = [self.initialize_cache(fanout_cache_storage=ampf) for ampf in self.cache_size]  # Initialize cache structure
        # self.cache_refresh(self.g)  # Pre-sample and populate the cache

    def initialize_cache(self, fanout_cache_storage):
        """
        Initializes the cache for each layer with an amplified fanout to pre-sample a larger
        set of neighbors. This pre-sampling helps in reducing the need for dynamic sampling 
        at every iteration, thereby improving efficiency.
        """
        cached_graph = self.g.sample_neighbors(
            # torch.arange(0, self.g.number_of_nodes()),
            {self.hete_label:list(range(0, self.g.num_nodes(self.hete_label)))},
            fanout_cache_storage,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        print("end init cache")
        return cached_graph

    def OTF_refresh_cache(self,layer_id, cached_graph_structure, seed_nodes, fanout_cache_refresh):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        fanout_cache_sample = self.cache_size[layer_id]-fanout_cache_refresh
        all_nodes = torch.arange(0,  self.g.num_nodes(self.hete_label))
        print("seed nodes:",seed_nodes)
        print("all nodes",all_nodes)
        mask = ~torch.isin(all_nodes, seed_nodes[self.hete_label])
        # bool mask to select those nodes do not in seed_nodes
        unchanged_nodes = {self.hete_label: all_nodes[mask]}
        # the rest node structure remain the same
        unchanged_structure = cached_graph_structure.sample_neighbors(
            unchanged_nodes,
            self.cache_size[layer_id],
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        # the OTF node structure should 
        changed_cache_remain = cached_graph_structure.sample_neighbors(
            seed_nodes,
            fanout_cache_sample,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        changed_disk_to_add = self.g.sample_neighbors(
            seed_nodes,
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        refreshed_cache = dgl.merge([unchanged_structure, changed_cache_remain, changed_disk_to_add])
        return refreshed_cache

    def sample_blocks(self, g,seed_nodes, exclude_eids=None):
        """
        Samples blocks from the graph for the specified seed nodes using the cache.

        Parameters
        ----------
        seed_nodes : Tensor
            The nodes for which the neighborhoods are to be sampled.

        Returns
        -------
        tuple
            A tuple containing the seed nodes for the next layer, the output nodes, and
            the list of blocks sampled from the graph.
        """
        output_nodes = seed_nodes

        # # refresh cache after a period of time for generalization
        # self.Toptim = int(g.number_of_nodes() / max(self.amplified_fanouts))
        # if self.cycle % self.Toptim == 0:
        #     self.cache_refresh(g)  # Refresh cache every T cycles
        
        self.cycle += 1

        blocks = []

        if self.fused and get_num_threads() > 1:
            # print("fused")
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                for ntype in list(seed_nodes.keys()):
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks

        for k in range(len(self.cache_struct)-1,-1,-1):
            fanout_cache_refresh = int(self.fanouts[k] * self.refresh_rate)

            # Refresh cache&disk partially, while retrieval cache&disk partially
            self.cache_struct[k] = self.OTF_refresh_cache(k, self.cache_struct[k], seed_nodes, fanout_cache_refresh)

            # Sample from cache
            frontier_cache = self.cache_struct[k].sample_neighbors(
                seed_nodes,
                self.fanouts[k],
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=self.exclude_eids,
            )

            # Sample frontier from the cache for acceleration
            block = to_block(frontier_cache, seed_nodes)
            if EID in frontier_cache.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier_cache.edata[EID]
            blocks.insert(0, block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        # output_nodes = seed_nodes
        return seed_nodes, output_nodes, blocks


class NeighborSampler_OTF_struct_PSCRFCF_shared_cache_hete(BlockSampler):    
    def __init__(
            self, 
            g,
            fanouts, 
            edge_dir='in', 
            amp_rate=2, 
            T=20,
            refresh_rate=0.4,
            hete_label=None,
            prob=None,
            mask=None,
            replace=False,
            prefetch_node_feats=None,
            prefetch_labels=None,
            prefetch_edge_feats=None,
            output_device=None,
            fused=True,
        ):

        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.g = g
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}
        self.exclude_eids = None

        self.amp_rate = amp_rate
        self.hete_label = hete_label
        self.cycle = 0  # Initialize sampling cycle counter
        self.sc_size = max([f * self.amp_rate for f in fanouts])  # Amplified fanouts for pre-sampling
        self.T = T
        self.refresh_rate = refresh_rate
        self.Toptim = None # int(self.g.number_of_nodes() / max(self.amplified_fanouts))
        self.shared_cache = self.initialize_cache(fanout_cache_storage=self.sc_size)  # Initialize cache structure
        # self.cache_refresh(self.g)  # Pre-sample and populate the cache

    def initialize_cache(self, fanout_cache_storage):
        """
        Initializes the cache for each layer with an amplified fanout to pre-sample a larger
        set of neighbors. This pre-sampling helps in reducing the need for dynamic sampling 
        at every iteration, thereby improving efficiency.
        """
        cached_graph = self.g.sample_neighbors(
            # torch.arange(0, self.g.number_of_nodes()),
            {self.hete_label:list(range(0, self.g.num_nodes(self.hete_label)))},
            fanout_cache_storage,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        print("end init cache")
        return cached_graph

    def OTF_refresh_cache(self, seed_nodes, fanout_cache_refresh):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        fanout_cache_sample = self.sc_size-fanout_cache_refresh
        all_nodes = torch.arange(0,  self.g.num_nodes(self.hete_label))
        print("seed nodes:",seed_nodes)
        print("all nodes",all_nodes)
        mask = ~torch.isin(all_nodes, seed_nodes[self.hete_label])
        # bool mask to select those nodes do not in seed_nodes
        unchanged_nodes = {self.hete_label: all_nodes[mask]}
        # the rest node structure remain the same
        unchanged_structure = self.shared_cache.sample_neighbors(
            unchanged_nodes,
            self.sc_size,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        # the OTF node structure should 
        changed_cache_remain = self.shared_cache.sample_neighbors(
            seed_nodes,
            fanout_cache_sample,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        changed_disk_to_add = self.g.sample_neighbors(
            seed_nodes,
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        refreshed_cache = dgl.merge([unchanged_structure, changed_cache_remain, changed_disk_to_add])
        return refreshed_cache

    def sample_blocks(self, g,seed_nodes, exclude_eids=None):
        """
        Samples blocks from the graph for the specified seed nodes using the cache.

        Parameters
        ----------
        seed_nodes : Tensor
            The nodes for which the neighborhoods are to be sampled.

        Returns
        -------
        tuple
            A tuple containing the seed nodes for the next layer, the output nodes, and
            the list of blocks sampled from the graph.
        """
        output_nodes = seed_nodes
        
        self.cycle += 1

        blocks = []

        if self.fused and get_num_threads() > 1:
            # print("fused")
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                for ntype in list(seed_nodes.keys()):
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks

        for k in range(len(self.fanouts)-1,-1,-1):
            fanout_cache_refresh = int(self.fanouts[k] * self.refresh_rate)

            # Refresh cache&disk partially, while retrieval cache&disk partially
            self.shared_cache = self.OTF_refresh_cache(seed_nodes, fanout_cache_refresh)

            # Sample from cache
            frontier_cache = self.shared_cache.sample_neighbors(
                seed_nodes,
                self.fanouts[k],
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=self.exclude_eids,
            )

            # Sample frontier from the cache for acceleration
            block = to_block(frontier_cache, seed_nodes)
            if EID in frontier_cache.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier_cache.edata[EID]
            blocks.insert(0, block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        # output_nodes = seed_nodes
        return seed_nodes, output_nodes, blocks
