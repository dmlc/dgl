# pylint: disable=global-variable-undefined, invalid-name
"""Multiprocess dataloader for distributed training"""
import inspect
from abc import ABC, abstractmethod
from collections.abc import Mapping

from .. import backend as F, transforms, utils
from ..base import EID, NID
from ..convert import heterograph
from .dist_context import get_sampler_pool

__all__ = [
    "NodeCollator",
    "EdgeCollator",
    "DistDataLoader",
    "DistNodeDataLoader",
    "DistEdgeDataLoader",
]

DATALOADER_ID = 0


class DistDataLoader:
    """DGL customized multiprocessing dataloader.

    DistDataLoader provides a similar interface to Pytorch's DataLoader to generate mini-batches
    with multiprocessing. It utilizes the worker processes created by
    :func:`dgl.distributed.initialize` to parallelize sampling.

    Parameters
    ----------
    dataset: a tensor
        Tensors of node IDs or edge IDs.
    batch_size: int
        The number of samples per batch to load.
    shuffle: bool, optional
        Set to ``True`` to have the data reshuffled at every epoch (default: ``False``).
    collate_fn: callable, optional
        The function is typically used to sample neighbors of the nodes in a batch
        or the endpoint nodes of the edges in a batch.
    drop_last: bool, optional
        Set to ``True`` to drop the last incomplete batch, if the dataset size is not
        divisible by the batch size. If ``False`` and the size of dataset is not divisible
        by the batch size, then the last batch will be smaller. (default: ``False``)
    queue_size: int, optional
        Size of multiprocessing queue

    Examples
    --------
    >>> g = dgl.distributed.DistGraph('graph-name')
    >>> def sample(seeds):
    ...     seeds = th.LongTensor(np.asarray(seeds))
    ...     frontier = dgl.distributed.sample_neighbors(g, seeds, 10)
    ...     return dgl.to_block(frontier, seeds)
    >>> dataloader = dgl.distributed.DistDataLoader(dataset=nodes, batch_size=1000,
                                                    collate_fn=sample, shuffle=True)
    >>> for block in dataloader:
    ...     feat = g.ndata['features'][block.srcdata[dgl.NID]]
    ...     labels = g.ndata['labels'][block.dstdata[dgl.NID]]
    ...     pred = model(block, feat)

    Note
    ----
    When performing DGL's distributed sampling with multiprocessing, users have to use this class
    instead of Pytorch's DataLoader because DGL's RPC requires that all processes establish
    connections with servers before invoking any DGL's distributed API. Therefore, this dataloader
    uses the worker processes created in :func:`dgl.distributed.initialize`.

    Note
    ----
    This dataloader does not guarantee the iteration order. For example,
    if dataset = [1, 2, 3, 4], batch_size = 2 and shuffle = False, the order of [1, 2]
    and [3, 4] is not guaranteed.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=False,
        collate_fn=None,
        drop_last=False,
        queue_size=None,
    ):
        self.pool, self.num_workers = get_sampler_pool()
        if queue_size is None:
            queue_size = self.num_workers * 4 if self.num_workers > 0 else 4
        self.queue_size = queue_size  # prefetch size
        self.batch_size = batch_size
        self.num_pending = 0
        self.collate_fn = collate_fn
        self.current_pos = 0
        self.queue = []  # Only used when pool is None
        self.drop_last = drop_last
        self.recv_idxs = 0
        self.shuffle = shuffle
        self.is_closed = False

        self.dataset = dataset
        self.data_idx = F.arange(0, len(dataset))
        self.expected_idxs = len(dataset) // self.batch_size
        if not self.drop_last and len(dataset) % self.batch_size != 0:
            self.expected_idxs += 1

        # We need to have a unique ID for each data loader to identify itself
        # in the sampler processes.
        global DATALOADER_ID
        self.name = "dataloader-" + str(DATALOADER_ID)
        DATALOADER_ID += 1

        if self.pool is not None:
            self.pool.set_collate_fn(self.collate_fn, self.name)

    def __del__(self):
        # When the process exits, the process pool may have been closed. We should try
        # and get the process pool again and see if we need to clean up the process pool.
        self.pool, self.num_workers = get_sampler_pool()
        if self.pool is not None:
            self.pool.delete_collate_fn(self.name)

    def __next__(self):
        if self.pool is None:
            num_reqs = 1
        else:
            num_reqs = self.queue_size - self.num_pending
        for _ in range(num_reqs):
            self._request_next_batch()
        if self.recv_idxs < self.expected_idxs:
            result = self._get_data_from_result_queue()
            self.recv_idxs += 1
            self.num_pending -= 1
            return result
        else:
            assert self.num_pending == 0
            raise StopIteration

    def _get_data_from_result_queue(self, timeout=1800):
        if self.pool is None:
            ret = self.queue.pop(0)
        else:
            ret = self.pool.get_result(self.name, timeout=timeout)
        return ret

    def __iter__(self):
        if self.shuffle:
            self.data_idx = F.rand_shuffle(self.data_idx)
        self.recv_idxs = 0
        self.current_pos = 0
        self.num_pending = 0
        return self

    def _request_next_batch(self):
        next_data = self._next_data()
        if next_data is None:
            return
        elif self.pool is not None:
            self.pool.submit_task(self.name, next_data)
        else:
            result = self.collate_fn(next_data)
            self.queue.append(result)
        self.num_pending += 1

    def _next_data(self):
        if self.current_pos == len(self.dataset):
            return None

        end_pos = 0
        if self.current_pos + self.batch_size > len(self.dataset):
            if self.drop_last:
                return None
            else:
                end_pos = len(self.dataset)
        else:
            end_pos = self.current_pos + self.batch_size
        idx = self.data_idx[self.current_pos : end_pos].tolist()
        ret = [self.dataset[i] for i in idx]
        # Sharing large number of tensors between processes will consume too many
        # file descriptors, so let's convert each tensor to scalar value beforehand.
        if isinstance(ret[0], tuple):
            ret = [(type, F.as_scalar(id)) for (type, id) in ret]
        else:
            ret = [F.as_scalar(id) for id in ret]
        self.current_pos = end_pos
        return ret


# [Note] As implementation of ``dgl.distributed.DistDataLoader`` is independent
# of ``dgl.dataloading.DataLoader`` currently, dedicated collators are defined
# here instead of using ``dgl.dataloading.CollateWrapper``.


def _find_exclude_eids_with_reverse_id(g, eids, reverse_eid_map):
    if isinstance(eids, Mapping):
        eids = {g.to_canonical_etype(k): v for k, v in eids.items()}
        exclude_eids = {
            k: F.cat([v, F.gather_row(reverse_eid_map[k], v)], 0)
            for k, v in eids.items()
        }
    else:
        exclude_eids = F.cat([eids, F.gather_row(reverse_eid_map, eids)], 0)
    return exclude_eids


def _find_exclude_eids_with_reverse_types(g, eids, reverse_etype_map):
    exclude_eids = {g.to_canonical_etype(k): v for k, v in eids.items()}
    reverse_etype_map = {
        g.to_canonical_etype(k): g.to_canonical_etype(v)
        for k, v in reverse_etype_map.items()
    }
    exclude_eids.update(
        {reverse_etype_map[k]: v for k, v in exclude_eids.items()}
    )
    return exclude_eids


def _find_exclude_eids(g, exclude_mode, eids, **kwargs):
    """Find all edge IDs to exclude according to :attr:`exclude_mode`.

    Parameters
    ----------
    g : DGLGraph
        The graph.
    exclude_mode : str, optional
        Can be either of the following,

        None (default)
            Does not exclude any edge.

        'self'
            Exclude the given edges themselves but nothing else.

        'reverse_id'
            Exclude all edges specified in ``eids``, as well as their reverse edges
            of the same edge type.

            The mapping from each edge ID to its reverse edge ID is specified in
            the keyword argument ``reverse_eid_map``.

            This mode assumes that the reverse of an edge with ID ``e`` and type
            ``etype`` will have ID ``reverse_eid_map[e]`` and type ``etype``.

        'reverse_types'
            Exclude all edges specified in ``eids``, as well as their reverse
            edges of the corresponding edge types.

            The mapping from each edge type to its reverse edge type is specified
            in the keyword argument ``reverse_etype_map``.

            This mode assumes that the reverse of an edge with ID ``e`` and type ``etype``
            will have ID ``e`` and type ``reverse_etype_map[etype]``.
    eids : Tensor or dict[etype, Tensor]
        The edge IDs.
    reverse_eid_map : Tensor or dict[etype, Tensor]
        The mapping from edge ID to its reverse edge ID.
    reverse_etype_map : dict[etype, etype]
        The mapping from edge etype to its reverse edge type.
    """
    if exclude_mode is None:
        return None
    elif exclude_mode == "self":
        if isinstance(eids, Mapping):
            eids = {g.to_canonical_etype(k): v for k, v in eids.items()}
        return eids
    elif exclude_mode == "reverse_id":
        return _find_exclude_eids_with_reverse_id(
            g, eids, kwargs["reverse_eid_map"]
        )
    elif exclude_mode == "reverse_types":
        return _find_exclude_eids_with_reverse_types(
            g, eids, kwargs["reverse_etype_map"]
        )
    else:
        raise ValueError("unsupported mode {}".format(exclude_mode))


class Collator(ABC):
    """Abstract DGL collator for training GNNs on downstream tasks stochastically.

    Provides a :attr:`dataset` object containing the collection of all nodes or edges,
    as well as a :attr:`collate` method that combines a set of items from
    :attr:`dataset` and obtains the message flow graphs (MFGs).

    Notes
    -----
    For the concept of MFGs, please refer to
    :ref:`User Guide Section 6 <guide-minibatch>` and
    :doc:`Minibatch Training Tutorials <tutorials/large/L0_neighbor_sampling_overview>`.
    """

    @property
    @abstractmethod
    def dataset(self):
        """Returns the dataset object of the collator."""
        raise NotImplementedError

    @abstractmethod
    def collate(self, items):
        """Combines the items from the dataset object and obtains the list of MFGs.

        Parameters
        ----------
        items : list[str, int]
            The list of node or edge IDs or type-ID pairs.

        Notes
        -----
        For the concept of MFGs, please refer to
        :ref:`User Guide Section 6 <guide-minibatch>` and
        :doc:`Minibatch Training Tutorials <tutorials/large/L0_neighbor_sampling_overview>`.
        """
        raise NotImplementedError

    @staticmethod
    def add_edge_attribute_to_graph(g, data_name):
        """Add data into the graph as an edge attribute.

        For some cases such as prob/mask-based sampling on GraphBolt partitions,
        we need to prepare such data beforehand. This is because data are
        usually saved in DistGraph.ndata/edata, but such data is not in the
        format that GraphBolt partitions require. And in GraphBolt, such data
        are saved as edge attributes. So we need to add such data into the graph
        before any sampling is kicked off.

        Parameters
        ----------
        g : DistGraph
            The graph.
        data_name : str
            The name of data that's stored in DistGraph.ndata/edata.
        """
        if g._use_graphbolt and data_name:
            g.add_edge_attribute(data_name)


class NodeCollator(Collator):
    """DGL collator to combine nodes and their computation dependencies within a minibatch for
    training node classification or regression on a single graph with neighborhood sampling.

    Parameters
    ----------
    g : DGLGraph
        The graph.
    nids : Tensor or dict[ntype, Tensor]
        The node set to compute outputs.
    graph_sampler : dgl.dataloading.BlockSampler
        The neighborhood sampler.

    Examples
    --------
    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from all neighbors (assume
    the backend is PyTorch):

    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    >>> collator = dgl.dataloading.NodeCollator(g, train_nid, sampler)
    >>> dataloader = torch.utils.data.DataLoader(
    ...     collator.dataset, collate_fn=collator.collate,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, output_nodes, blocks in dataloader:
    ...     train_on(input_nodes, output_nodes, blocks)

    Notes
    -----
    For the concept of MFGs, please refer to
    :ref:`User Guide Section 6 <guide-minibatch>` and
    :doc:`Minibatch Training Tutorials <tutorials/large/L0_neighbor_sampling_overview>`.
    """

    def __init__(self, g, nids, graph_sampler):
        self.g = g
        if not isinstance(nids, Mapping):
            assert (
                len(g.ntypes) == 1
            ), "nids should be a dict of node type and ids for graph with multiple node types"
        self.graph_sampler = graph_sampler

        self.nids = utils.prepare_tensor_or_dict(g, nids, "nids")
        self._dataset = utils.maybe_flatten_dict(self.nids)

        # Add prob/mask into graphbolt partition's edge attributes if needed.
        if hasattr(self.graph_sampler, "prob"):
            Collator.add_edge_attribute_to_graph(
                self.g, self.graph_sampler.prob
            )

    @property
    def dataset(self):
        return self._dataset

    def collate(self, items):
        """Find the list of MFGs necessary for computing the representation of given
        nodes for a node classification/regression task.

        Parameters
        ----------
        items : list[int] or list[tuple[str, int]]
            Either a list of node IDs (for homogeneous graphs), or a list of node type-ID
            pairs (for heterogeneous graphs).

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
        MFGs : list[DGLGraph]
            The list of MFGs necessary for computing the representation.
        """
        if isinstance(items[0], tuple):
            # returns a list of pairs: group them by node types into a dict
            items = utils.group_as_dict(items)
        items = utils.prepare_tensor_or_dict(self.g, items, "items")

        input_nodes, output_nodes, blocks = self.graph_sampler.sample_blocks(
            self.g, items
        )

        return input_nodes, output_nodes, blocks


class EdgeCollator(Collator):
    """DGL collator to combine edges and their computation dependencies within a minibatch for
    training edge classification, edge regression, or link prediction on a single graph
    with neighborhood sampling.

    Given a set of edges, the collate function will yield

    * A tensor of input nodes necessary for computing the representation on edges, or
      a dictionary of node type names and such tensors.

    * A subgraph that contains only the edges in the minibatch and their incident nodes.
      Note that the graph has an identical metagraph with the original graph.

    * If a negative sampler is given, another graph that contains the "negative edges",
      connecting the source and destination nodes yielded from the given negative sampler.

    * A list of MFGs necessary for computing the representation of the incident nodes
      of the edges in the minibatch.

    Parameters
    ----------
    g : DGLGraph
        The graph from which the edges are iterated in minibatches and the subgraphs
        are generated.
    eids : Tensor or dict[etype, Tensor]
        The edge set in graph :attr:`g` to compute outputs.
    graph_sampler : dgl.dataloading.BlockSampler
        The neighborhood sampler.
    g_sampling : DGLGraph, optional
        The graph where neighborhood sampling and message passing is performed.

        Note that this is not necessarily the same as :attr:`g`.

        If None, assume to be the same as :attr:`g`.
    exclude : str, optional
        Whether and how to exclude dependencies related to the sampled edges in the
        minibatch.  Possible values are

        * None, which excludes nothing.

        * ``'self'``, which excludes the sampled edges themselves but nothing else.

        * ``'reverse_id'``, which excludes the reverse edges of the sampled edges.  The said
          reverse edges have the same edge type as the sampled edges.  Only works
          on edge types whose source node type is the same as its destination node type.

        * ``'reverse_types'``, which excludes the reverse edges of the sampled edges.  The
          said reverse edges have different edge types from the sampled edges.

        If ``g_sampling`` is given, ``exclude`` is ignored and will be always ``None``.
    reverse_eids : Tensor or dict[etype, Tensor], optional
        A tensor of reverse edge ID mapping.  The i-th element indicates the ID of
        the i-th edge's reverse edge.

        If the graph is heterogeneous, this argument requires a dictionary of edge
        types and the reverse edge ID mapping tensors.

        Required and only used when ``exclude`` is set to ``reverse_id``.

        For heterogeneous graph this will be a dict of edge type and edge IDs.  Note that
        only the edge types whose source node type is the same as destination node type
        are needed.
    reverse_etypes : dict[etype, etype], optional
        The mapping from the edge type to its reverse edge type.

        Required and only used when ``exclude`` is set to ``reverse_types``.
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
        :ref:`the negative sampling module <api-dataloading-negative-sampling>`.

    Examples
    --------
    The following example shows how to train a 3-layer GNN for edge classification on a
    set of edges ``train_eid`` on a homogeneous undirected graph. Each node takes
    messages from all neighbors.

    Say that you have an array of source node IDs ``src`` and another array of destination
    node IDs ``dst``.  One can make it bidirectional by adding another set of edges
    that connects from ``dst`` to ``src``:

    >>> g = dgl.graph((torch.cat([src, dst]), torch.cat([dst, src])))

    One can then know that the ID difference of an edge and its reverse edge is ``|E|``,
    where ``|E|`` is the length of your source/destination array.  The reverse edge
    mapping can be obtained by

    >>> E = len(src)
    >>> reverse_eids = torch.cat([torch.arange(E, 2 * E), torch.arange(0, E)])

    Note that the sampled edges as well as their reverse edges are removed from
    computation dependencies of the incident nodes.  This is a common trick to avoid
    information leakage.

    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    >>> collator = dgl.dataloading.EdgeCollator(
    ...     g, train_eid, sampler, exclude='reverse_id',
    ...     reverse_eids=reverse_eids)
    >>> dataloader = torch.utils.data.DataLoader(
    ...     collator.dataset, collate_fn=collator.collate,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, pair_graph, blocks in dataloader:
    ...     train_on(input_nodes, pair_graph, blocks)

    To train a 3-layer GNN for link prediction on a set of edges ``train_eid`` on a
    homogeneous graph where each node takes messages from all neighbors (assume the
    backend is PyTorch), with 5 uniformly chosen negative samples per edge:

    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    >>> neg_sampler = dgl.dataloading.negative_sampler.Uniform(5)
    >>> collator = dgl.dataloading.EdgeCollator(
    ...     g, train_eid, sampler, exclude='reverse_id',
    ...     reverse_eids=reverse_eids, negative_sampler=neg_sampler)
    >>> dataloader = torch.utils.data.DataLoader(
    ...     collator.dataset, collate_fn=collator.collate,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, pos_pair_graph, neg_pair_graph, blocks in dataloader:
    ...     train_on(input_nodse, pair_graph, neg_pair_graph, blocks)

    For heterogeneous graphs, the reverse of an edge may have a different edge type
    from the original edge.  For instance, consider that you have an array of
    user-item clicks, representated by a user array ``user`` and an item array ``item``.
    You may want to build a heterogeneous graph with a user-click-item relation and an
    item-clicked-by-user relation.

    >>> g = dgl.heterograph({
    ...     ('user', 'click', 'item'): (user, item),
    ...     ('item', 'clicked-by', 'user'): (item, user)})

    To train a 3-layer GNN for edge classification on a set of edges ``train_eid`` with
    type ``click``, you can write

    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    >>> collator = dgl.dataloading.EdgeCollator(
    ...     g, {'click': train_eid}, sampler, exclude='reverse_types',
    ...     reverse_etypes={'click': 'clicked-by', 'clicked-by': 'click'})
    >>> dataloader = torch.utils.data.DataLoader(
    ...     collator.dataset, collate_fn=collator.collate,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, pair_graph, blocks in dataloader:
    ...     train_on(input_nodes, pair_graph, blocks)

    To train a 3-layer GNN for link prediction on a set of edges ``train_eid`` with type
    ``click``, you can write

    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    >>> neg_sampler = dgl.dataloading.negative_sampler.Uniform(5)
    >>> collator = dgl.dataloading.EdgeCollator(
    ...     g, train_eid, sampler, exclude='reverse_types',
    ...     reverse_etypes={'click': 'clicked-by', 'clicked-by': 'click'},
    ...     negative_sampler=neg_sampler)
    >>> dataloader = torch.utils.data.DataLoader(
    ...     collator.dataset, collate_fn=collator.collate,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, pos_pair_graph, neg_pair_graph, blocks in dataloader:
    ...     train_on(input_nodes, pair_graph, neg_pair_graph, blocks)

    Notes
    -----
    For the concept of MFGs, please refer to
    :ref:`User Guide Section 6 <guide-minibatch>` and
    :doc:`Minibatch Training Tutorials <tutorials/large/L0_neighbor_sampling_overview>`.
    """

    def __init__(
        self,
        g,
        eids,
        graph_sampler,
        g_sampling=None,
        exclude=None,
        reverse_eids=None,
        reverse_etypes=None,
        negative_sampler=None,
    ):
        self.g = g
        if not isinstance(eids, Mapping):
            assert (
                len(g.etypes) == 1
            ), "eids should be a dict of etype and ids for graph with multiple etypes"
        self.graph_sampler = graph_sampler

        # One may wish to iterate over the edges in one graph while perform sampling in
        # another graph.  This may be the case for iterating over validation and test
        # edge set while perform neighborhood sampling on the graph formed by only
        # the training edge set.
        # See GCMC for an example usage.
        if g_sampling is not None:
            self.g_sampling = g_sampling
            self.exclude = None
        else:
            self.g_sampling = self.g
            self.exclude = exclude

        self.reverse_eids = reverse_eids
        self.reverse_etypes = reverse_etypes
        self.negative_sampler = negative_sampler

        self.eids = utils.prepare_tensor_or_dict(g, eids, "eids")
        self._dataset = utils.maybe_flatten_dict(self.eids)

        # Add prob/mask into graphbolt partition's edge attributes if needed.
        if hasattr(self.graph_sampler, "prob"):
            Collator.add_edge_attribute_to_graph(
                self.g, self.graph_sampler.prob
            )

    @property
    def dataset(self):
        return self._dataset

    def _collate(self, items):
        if isinstance(items[0], tuple):
            # returns a list of pairs: group them by node types into a dict
            items = utils.group_as_dict(items)
        items = utils.prepare_tensor_or_dict(self.g_sampling, items, "items")

        pair_graph = self.g.edge_subgraph(items)
        seed_nodes = pair_graph.ndata[NID]

        exclude_eids = _find_exclude_eids(
            self.g_sampling,
            self.exclude,
            items,
            reverse_eid_map=self.reverse_eids,
            reverse_etype_map=self.reverse_etypes,
        )

        input_nodes, _, blocks = self.graph_sampler.sample_blocks(
            self.g_sampling, seed_nodes, exclude_eids=exclude_eids
        )

        return input_nodes, pair_graph, blocks

    def _collate_with_negative_sampling(self, items):
        if isinstance(items[0], tuple):
            # returns a list of pairs: group them by node types into a dict
            items = utils.group_as_dict(items)
        items = utils.prepare_tensor_or_dict(self.g_sampling, items, "items")

        pair_graph = self.g.edge_subgraph(items, relabel_nodes=False)
        induced_edges = pair_graph.edata[EID]

        neg_srcdst = self.negative_sampler(self.g, items)
        if not isinstance(neg_srcdst, Mapping):
            assert len(self.g.etypes) == 1, (
                "graph has multiple or no edge types; "
                "please return a dict in negative sampler."
            )
            neg_srcdst = {self.g.canonical_etypes[0]: neg_srcdst}
        # Get dtype from a tuple of tensors
        dtype = F.dtype(list(neg_srcdst.values())[0][0])
        ctx = F.context(pair_graph)
        neg_edges = {
            etype: neg_srcdst.get(
                etype,
                (
                    F.copy_to(F.tensor([], dtype), ctx),
                    F.copy_to(F.tensor([], dtype), ctx),
                ),
            )
            for etype in self.g.canonical_etypes
        }
        neg_pair_graph = heterograph(
            neg_edges,
            {ntype: self.g.num_nodes(ntype) for ntype in self.g.ntypes},
        )

        pair_graph, neg_pair_graph = transforms.compact_graphs(
            [pair_graph, neg_pair_graph]
        )
        pair_graph.edata[EID] = induced_edges

        seed_nodes = pair_graph.ndata[NID]

        exclude_eids = _find_exclude_eids(
            self.g_sampling,
            self.exclude,
            items,
            reverse_eid_map=self.reverse_eids,
            reverse_etype_map=self.reverse_etypes,
        )

        input_nodes, _, blocks = self.graph_sampler.sample_blocks(
            self.g_sampling, seed_nodes, exclude_eids=exclude_eids
        )

        return input_nodes, pair_graph, neg_pair_graph, blocks

    def collate(self, items):
        """Combines the sampled edges into a minibatch for edge classification, edge
        regression, and link prediction tasks.

        Parameters
        ----------
        items : list[int] or list[tuple[str, int]]
            Either a list of edge IDs (for homogeneous graphs), or a list of edge type-ID
            pairs (for heterogeneous graphs).

        Returns
        -------
        Either ``(input_nodes, pair_graph, blocks)``, or
        ``(input_nodes, pair_graph, negative_pair_graph, blocks)`` if negative sampling is
        enabled.

        input_nodes : Tensor or dict[ntype, Tensor]
            The input nodes necessary for computation in this minibatch.

            If the original graph has multiple node types, return a dictionary of
            node type names and node ID tensors.  Otherwise, return a single tensor.
        pair_graph : DGLGraph
            The graph that contains only the edges in the minibatch as well as their incident
            nodes.

            Note that the metagraph of this graph will be identical to that of the original
            graph.
        negative_pair_graph : DGLGraph
            The graph that contains only the edges connecting the source and destination nodes
            yielded from the given negative sampler, if negative sampling is enabled.

            Note that the metagraph of this graph will be identical to that of the original
            graph.
        blocks : list[DGLGraph]
            The list of MFGs necessary for computing the representation of the edges.
        """
        if self.negative_sampler is None:
            return self._collate(items)
        else:
            return self._collate_with_negative_sampling(items)


def _remove_kwargs_dist(kwargs):
    if "num_workers" in kwargs:
        del kwargs["num_workers"]
    if "pin_memory" in kwargs:
        del kwargs["pin_memory"]
        print("Distributed DataLoaders do not support pin_memory.")
    return kwargs


class DistNodeDataLoader(DistDataLoader):
    """Sampled graph data loader over nodes for distributed graph storage.

    It wraps an iterable over a set of nodes, generating the list
    of message flow graphs (MFGs) as computation dependency of the said minibatch, on
    a distributed graph.

    All the arguments have the same meaning as the single-machine counterpart
    :class:`dgl.dataloading.DataLoader` except the first argument
    :attr:`g` which must be a :class:`dgl.distributed.DistGraph`.

    Parameters
    ----------
    g : DistGraph
        The distributed graph.

    nids, graph_sampler, device, kwargs :
        See :class:`dgl.dataloading.DataLoader`.

    See also
    --------
    dgl.dataloading.DataLoader
    """

    def __init__(self, g, nids, graph_sampler, device=None, **kwargs):
        collator_kwargs = {}
        dataloader_kwargs = {}
        _collator_arglist = inspect.getfullargspec(NodeCollator).args
        for k, v in kwargs.items():
            if k in _collator_arglist:
                collator_kwargs[k] = v
            else:
                dataloader_kwargs[k] = v
        if device is None:
            # for the distributed case default to the CPU
            device = "cpu"
        assert (
            device == "cpu"
        ), "Only cpu is supported in the case of a DistGraph."
        # Distributed DataLoader currently does not support heterogeneous graphs
        # and does not copy features.  Fallback to normal solution
        self.collator = NodeCollator(g, nids, graph_sampler, **collator_kwargs)
        _remove_kwargs_dist(dataloader_kwargs)
        super().__init__(
            self.collator.dataset,
            collate_fn=self.collator.collate,
            **dataloader_kwargs
        )
        self.device = device


class DistEdgeDataLoader(DistDataLoader):
    """Sampled graph data loader over edges for distributed graph storage.

    It wraps an iterable over a set of edges, generating the list
    of message flow graphs (MFGs) as computation dependency of the said minibatch for
    edge classification, edge regression, and link prediction, on a distributed
    graph.

    All the arguments have the same meaning as the single-machine counterpart
    :class:`dgl.dataloading.DataLoader` except the first argument
    :attr:`g` which must be a :class:`dgl.distributed.DistGraph`.

    Parameters
    ----------
    g : DistGraph
        The distributed graph.

    eids, graph_sampler, device, kwargs :
        See :class:`dgl.dataloading.DataLoader`.

    See also
    --------
    dgl.dataloading.DataLoader
    """

    def __init__(self, g, eids, graph_sampler, device=None, **kwargs):
        collator_kwargs = {}
        dataloader_kwargs = {}
        _collator_arglist = inspect.getfullargspec(EdgeCollator).args
        for k, v in kwargs.items():
            if k in _collator_arglist:
                collator_kwargs[k] = v
            else:
                dataloader_kwargs[k] = v

        if device is None:
            # for the distributed case default to the CPU
            device = "cpu"
        assert (
            device == "cpu"
        ), "Only cpu is supported in the case of a DistGraph."
        # Distributed DataLoader currently does not support heterogeneous graphs
        # and does not copy features.  Fallback to normal solution
        self.collator = EdgeCollator(g, eids, graph_sampler, **collator_kwargs)
        _remove_kwargs_dist(dataloader_kwargs)
        super().__init__(
            self.collator.dataset,
            collate_fn=self.collator.collate,
            **dataloader_kwargs
        )

        self.device = device
