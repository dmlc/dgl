"""Base classes and functionalities for dataloaders"""
from collections import Mapping
import inspect
from ..base import NID, EID
from ..convert import heterograph
from .. import backend as F
from ..transforms import compact_graphs
from ..frame import LazyFeature
from ..utils import recursive_apply

def _set_lazy_features(x, xdata, feature_names):
    if feature_names is None:
        return
    if not isinstance(feature_names, Mapping):
        xdata.update({k: LazyFeature(k) for k in feature_names})
    else:
        for type_, names in feature_names.items():
            x[type_].data.update({k: LazyFeature(k) for k in names})

def set_node_lazy_features(g, feature_names):
    """Assign LazyFeature objects to ``g.ndata`` for given feature names for prefetching
    with :class:`dgl.dataloading.NodeDataLoader` and :class:`dgl.dataloading.EdgeDataLoader`.

    If the graph is homogeneous, this is equivalent to:

    .. code:: python

       g.ndata.update({k: LazyFeature(k, g.ndata[dgl.NID]) for k in feature_names})

    If the graph is heterogeneous, this is equivalent to:

    .. code:: python

        for type_, names in feature_names.items():
            g.nodes[type_].data.update(
                {k: LazyFeature(k, g.nodes[type_].data[dgl.NID]) for k in names})

    Parameters
    ----------
    g : DGLGraph
        The graph.
    feature_names : list[str] or dict[ntype, list[str]]
        The feature names to prefetch.

    See also
    --------
    dgl.frame.LazyFeature
    """
    return _set_lazy_features(g.nodes, g.ndata, feature_names)

def set_edge_lazy_features(g, feature_names):
    """Assign LazyFeature objects to ``g.edata`` for given feature names for prefetching
    with :class:`dgl.dataloading.NodeDataLoader` and :class:`dgl.dataloading.EdgeDataLoader`.

    If the graph is homogeneous, this is equivalent to:

    .. code:: python

       g.edata.update({k: LazyFeature(k, g.edata[dgl.EID]) for k in feature_names})

    If the graph is heterogeneous, this is equivalent to:

    .. code:: python

        for type_, names in feature_names.items():
            g.edges[type_].data.update(
                {k: LazyFeature(k, g.edges[type_].data[dgl.EID]) for k in names})

    Parameters
    ----------
    g : DGLGraph
        The graph.
    feature_names : list[str] or dict[etype, list[str]]
        The feature names to prefetch.

    See also
    --------
    dgl.frame.LazyFeature
    """
    return _set_lazy_features(g.edges, g.edata, feature_names)

def set_src_lazy_features(g, feature_names):
    """Assign LazyFeature objects to ``g.srcdata`` for given feature names for prefetching
    with :class:`dgl.dataloading.NodeDataLoader` and :class:`dgl.dataloading.EdgeDataLoader`.

    If the graph is homogeneous, this is equivalent to:

    .. code:: python

       g.srcdata.update({k: LazyFeature(k, g.srcdata[dgl.NID]) for k in feature_names})

    If the graph is heterogeneous, this is equivalent to:

    .. code:: python

        for type_, names in feature_names.items():
            g.srcnodes[type_].data.update(
                {k: LazyFeature(k, g.srcnodes[type_].data[dgl.NID]) for k in names})

    Parameters
    ----------
    g : DGLGraph
        The graph.
    feature_names : list[str] or dict[ntype, list[str]]
        The feature names to prefetch.

    See also
    --------
    dgl.frame.LazyFeature
    """
    return _set_lazy_features(g.srcnodes, g.srcdata, feature_names)

def set_dst_lazy_features(g, feature_names):
    """Assign LazyFeature objects to ``g.srcdata`` for given feature names for prefetching
    with :class:`dgl.dataloading.NodeDataLoader` and :class:`dgl.dataloading.EdgeDataLoader`.

    If the graph is homogeneous, this is equivalent to:

    .. code:: python

       g.dstdata.update({k: LazyFeature(k, g.dstdata[dgl.NID]) for k in feature_names})

    If the graph is heterogeneous, this is equivalent to:

    .. code:: python

        for type_, names in feature_names.items():
            g.dstnodes[type_].data.update(
                {k: LazyFeature(k, g.dstnodes[type_].data[dgl.NID]) for k in names})

    Parameters
    ----------
    g : DGLGraph
        The graph.
    feature_names : list[str] or dict[ntype, list[str]]
        The feature names to prefetch.

    See also
    --------
    dgl.frame.LazyFeature
    """
    return _set_lazy_features(g.dstnodes, g.dstdata, feature_names)

class Sampler(object):
    """Abstract sampler class."""
    def sample(self, g, indices):
        """Abstract sample method.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        indices : object
            Any object representing the indices selected in the current minibatch.
        """
        raise NotImplementedError

class BlockSampler(Sampler):
    """Abstract class that assumes to take in a set of nodes whose
    outputs are to compute, and returns a list of MFGs.

    Moreover, it assumes that the input node features will be put in the first MFG's
    ``srcdata``, the output node labels will be put in the last MFG's ``dstdata``, and
    the edge data will be put in all the MFGs' ``edata``.

    Parameters
    ----------
    prefetch_node_feats : list[str] or dict[ntype, list[str]], optional
        The node data to prefetch for the first MFG.

        DGL will populate the first layer's MFG's ``srcnodes`` and ``srcdata`` with
        the node data of the given names from the original graph.
    prefetch_labels : list[str] or dict[ntype, list[str]], optional
        The node data to prefetch for the last MFG.

        DGL will populate the last layer's MFG's ``dstnodes`` and ``dstdata`` with
        the node data of the given names from the original graph.
    prefetch_edge_feats : list[str] or dict[etype, list[str]], optional
        The edge data names to prefetch for all the MFGs.

        DGL will populate every MFG's ``edges`` and ``edata`` with the edge data
        of the given names from the original graph.
    output_device : device, optional
        The device of the output subgraphs or MFGs.  Default is the same as the
        minibatch of seed nodes.
    """
    def __init__(self, prefetch_node_feats=None, prefetch_labels=None,
                 prefetch_edge_feats=None, output_device=None):
        super().__init__()
        self.prefetch_node_feats = prefetch_node_feats or []
        self.prefetch_labels = prefetch_labels or []
        self.prefetch_edge_feats = prefetch_edge_feats or []
        self.output_device = output_device

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        """Generates a list of blocks from the given seed nodes.

        This function must return a triplet where the first element is the input node IDs
        for the first GNN layer (a tensor or a dict of tensors for heterogeneous graphs),
        the second element is the output node IDs for the last GNN layer, and the third
        element is the said list of blocks.
        """
        raise NotImplementedError

    def assign_lazy_features(self, result):
        """Assign lazy features for prefetching."""
        # A LazyFeature is a placeholder telling the dataloader where and which IDs
        # to prefetch.  It has the signature LazyFeature(name, id_).  id_ can be None
        # if the LazyFeature is set into one of the subgraph's ``xdata``, in which case the
        # dataloader will infer from the subgraph's ``xdata[dgl.NID]`` (or ``xdata[dgl.EID]``
        # if the LazyFeature is set as edge features).
        #
        # If you want to prefetch things other than ndata and edata, you can also
        # return a LazyFeature(name, id_).  If a LazyFeature is returned in places other than
        # in a graph's ndata/edata/srcdata/dstdata, the DataLoader will prefetch it
        # from its dictionary ``other_data``.
        # For instance, you can run
        #
        #     return blocks, LazyFeature('other_feat', id_)
        #
        # To make it work with the sampler returning the stuff above, your dataloader
        # needs to have the following
        #
        #     dataloader.attach_data('other_feat', tensor)
        #
        # Then you can run
        #
        #     for blocks, other_feat in dataloader:
        #         train_on(blocks, other_feat)
        input_nodes, output_nodes, blocks = result
        set_src_lazy_features(blocks[0], self.prefetch_node_feats)
        set_dst_lazy_features(blocks[-1], self.prefetch_labels)
        for block in blocks:
            set_edge_lazy_features(block, self.prefetch_edge_feats)
        return input_nodes, output_nodes, blocks

    def sample(self, g, seed_nodes, exclude_eids=None):     # pylint: disable=arguments-differ
        """Sample a list of blocks from the given seed nodes."""
        result = self.sample_blocks(g, seed_nodes, exclude_eids=exclude_eids)
        return self.assign_lazy_features(result)


def _find_exclude_eids_with_reverse_id(g, eids, reverse_eid_map):
    if isinstance(eids, Mapping):
        eids = {g.to_canonical_etype(k): v for k, v in eids.items()}
        exclude_eids = {
            k: F.cat([v, F.gather_row(reverse_eid_map[k], v)], 0)
            for k, v in eids.items()}
    else:
        exclude_eids = F.cat([eids, F.gather_row(reverse_eid_map, eids)], 0)
    return exclude_eids

def _find_exclude_eids_with_reverse_types(g, eids, reverse_etype_map):
    exclude_eids = {g.to_canonical_etype(k): v for k, v in eids.items()}
    reverse_etype_map = {
        g.to_canonical_etype(k): g.to_canonical_etype(v)
        for k, v in reverse_etype_map.items()}
    exclude_eids.update({reverse_etype_map[k]: v for k, v in exclude_eids.items()})
    return exclude_eids

def _find_exclude_eids(g, exclude_mode, eids, **kwargs):
    if exclude_mode is None:
        return None
    elif callable(exclude_mode):
        return exclude_mode(eids)
    elif F.is_tensor(exclude_mode) or (
            isinstance(exclude_mode, Mapping) and
            all(F.is_tensor(v) for v in exclude_mode.values())):
        return exclude_mode
    elif exclude_mode == 'self':
        return eids
    elif exclude_mode == 'reverse_id':
        return _find_exclude_eids_with_reverse_id(g, eids, kwargs['reverse_eid_map'])
    elif exclude_mode == 'reverse_types':
        return _find_exclude_eids_with_reverse_types(g, eids, kwargs['reverse_etype_map'])
    else:
        raise ValueError('unsupported mode {}'.format(exclude_mode))

def find_exclude_eids(g, seed_edges, exclude, reverse_eids=None, reverse_etypes=None,
                      output_device=None):
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

        callable
            Any function that takes in a single argument :attr:`seed_edges` and returns
            a tensor or dict of tensors.
    eids : Tensor or dict[etype, Tensor]
        The edge IDs.
    reverse_eids : Tensor or dict[etype, Tensor]
        The mapping from edge ID to its reverse edge ID.
    reverse_etypes : dict[etype, etype]
        The mapping from edge etype to its reverse edge type.
    output_device : device
        The device of the output edge IDs.
    """
    exclude_eids = _find_exclude_eids(
        g,
        exclude,
        seed_edges,
        reverse_eid_map=reverse_eids,
        reverse_etype_map=reverse_etypes)
    if exclude_eids is not None and output_device is not None:
        exclude_eids = recursive_apply(exclude_eids, lambda x: F.copy_to(x, output_device))
    return exclude_eids

class EdgePredictionSampler(Sampler):
    """Sampler class that wraps an existing sampler for node classification into another
    one for edge classification or link prediction.

    See also
    --------
    as_edge_prediction_sampler
    """
    def __init__(self, sampler, exclude=None, reverse_eids=None,
                 reverse_etypes=None, negative_sampler=None, prefetch_labels=None):
        super().__init__()
        # Check if the sampler's sample method has an optional third argument.
        argspec = inspect.getfullargspec(sampler.sample)
        if len(argspec.args) < 4:       # ['self', 'g', 'indices', 'exclude_eids']
            raise TypeError(
                "This sampler does not support edge or link prediction; please add an"
                "optional third argument for edge IDs to exclude in its sample() method.")
        self.reverse_eids = reverse_eids
        self.reverse_etypes = reverse_etypes
        self.exclude = exclude
        self.sampler = sampler
        self.negative_sampler = negative_sampler
        self.prefetch_labels = prefetch_labels or []
        self.output_device = sampler.output_device

    def _build_neg_graph(self, g, seed_edges):
        neg_srcdst = self.negative_sampler(g, seed_edges)
        if not isinstance(neg_srcdst, Mapping):
            assert len(g.canonical_etypes) == 1, \
                'graph has multiple or no edge types; '\
                'please return a dict in negative sampler.'
            neg_srcdst = {g.canonical_etypes[0]: neg_srcdst}

        dtype = F.dtype(list(neg_srcdst.values())[0][0])
        neg_edges = {
            etype: neg_srcdst.get(etype, (F.tensor([], dtype), F.tensor([], dtype)))
            for etype in g.canonical_etypes}
        neg_pair_graph = heterograph(
            neg_edges, {ntype: g.num_nodes(ntype) for ntype in g.ntypes})
        return neg_pair_graph

    def assign_lazy_features(self, result):
        """Assign lazy features for prefetching."""
        pair_graph = result[1]
        set_edge_lazy_features(pair_graph, self.prefetch_labels)
        # In-place updates
        return result

    def sample(self, g, seed_edges):    # pylint: disable=arguments-differ
        """Samples a list of blocks, as well as a subgraph containing the sampled
        edges from the original graph.

        If :attr:`negative_sampler` is given, also returns another graph containing the
        negative pairs as edges.
        """
        if isinstance(seed_edges, Mapping):
            seed_edges = {g.to_canonical_etype(k): v for k, v in seed_edges.items()}
        exclude = self.exclude
        pair_graph = g.edge_subgraph(
            seed_edges, relabel_nodes=False, output_device=self.output_device)
        eids = pair_graph.edata[EID]

        if self.negative_sampler is not None:
            neg_graph = self._build_neg_graph(g, seed_edges)
            pair_graph, neg_graph = compact_graphs([pair_graph, neg_graph])
        else:
            pair_graph = compact_graphs(pair_graph)

        pair_graph.edata[EID] = eids
        seed_nodes = pair_graph.ndata[NID]

        exclude_eids = find_exclude_eids(
            g, seed_edges, exclude, self.reverse_eids, self.reverse_etypes,
            self.output_device)

        input_nodes, _, blocks = self.sampler.sample(g, seed_nodes, exclude_eids)

        if self.negative_sampler is None:
            return self.assign_lazy_features((input_nodes, pair_graph, blocks))
        else:
            return self.assign_lazy_features((input_nodes, pair_graph, neg_graph, blocks))

def as_edge_prediction_sampler(
        sampler, exclude=None, reverse_eids=None, reverse_etypes=None, negative_sampler=None,
        prefetch_labels=None):
    """Wrap a sampler for node classification/regression into another sampler for
    edge classification/regression or link prediction.

    For each iteration, the sampler will yield

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
    sampler : Sampler
        The sampler object.  It additionally requires that the :attr:`sample` method
        must have an optional third argument :attr:`exclude_eids` representing the
        edge IDs to exclude from neighborhood.  The argument will be either a tensor
        for homogeneous graphs or a dict of edge types and tensors for heterogeneous
        graphs.
    exclude : str, optional
        Whether and how to exclude dependencies related to the sampled edges in the
        minibatch.  Possible values are

        * None, for not excluding any edges.

        * ``self``, for excluding only the edges sampled as seed edges in this minibatch.

        * ``reverse_id``, for excluding not only the edges sampled in the minibatch but
          also their reverse edges of the same edge type.  Requires the argument
          :attr:`reverse_eids`.

        * ``reverse_types``, for excluding not only the edges sampled in the minibatch
          but also their reverse edges of different types but with the same IDs.
          Requires the argument :attr:`reverse_etypes`.

        * A callable which takes in a tensor or a dictionary of tensors and their
          canonical edge types and returns a tensor or dictionary of tensors to
          exclude.
    reverse_eids : Tensor or dict[etype, Tensor], optional
        A tensor of reverse edge ID mapping.  The i-th element indicates the ID of
        the i-th edge's reverse edge.

        If the graph is heterogeneous, this argument requires a dictionary of edge
        types and the reverse edge ID mapping tensors.

        See the description of the argument with the same name in the docstring of
        :class:`~dgl.dataloading.EdgeCollator` for more details.
    reverse_etypes : dict[etype, etype], optional
        The mapping from the original edge types to their reverse edge types.

        See the description of the argument with the same name in the docstring of
        :class:`~dgl.dataloading.EdgeCollator` for more details.
    negative_sampler : callable, optional
        The negative sampler.

        See the description of the argument with the same name in the docstring of
        :class:`~dgl.dataloading.EdgeCollator` for more details.
    prefetch_labels : list[str] or dict[etype, list[str]], optional
        The node data to prefetch for the returned positive pair graph.

        DGL will populate the pair graph's ``edges`` and ``edata`` with
        the edge data of the given names from the original graph.

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
    computation dependencies of the incident nodes.  That is, the edge will not
    involve in neighbor sampling and message aggregation.  This is a common trick
    to avoid information leakage.

    >>> sampler = dgl.dataloading.NeighborSampler([15, 10, 5])
    >>> sampler = dgl.dataloading.as_edge_prediction_sampler(
    ...     sampler, exclude='reverse_id', reverse_eids=reverse_eids)
    >>> dataloader = dgl.dataloading.EdgeDataLoader(
    ...     g, train_eid, sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, pair_graph, blocks in dataloader:
    ...     train_on(input_nodes, pair_graph, blocks)

    To train a 3-layer GNN for link prediction on a set of edges ``train_eid`` on a
    homogeneous graph where each node takes messages from all neighbors (assume the
    backend is PyTorch), with 5 uniformly chosen negative samples per edge:

    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    >>> neg_sampler = dgl.dataloading.negative_sampler.Uniform(5)
    >>> sampler = dgl.dataloading.as_edge_prediction_sampler(
    ...     sampler, exclude='reverse_id', reverse_eids=reverse_eids,
    ...     negative_sampler=neg_sampler)
    >>> dataloader = dgl.dataloading.EdgeDataLoader(
    ...     g, train_eid, sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, pos_pair_graph, neg_pair_graph, blocks in dataloader:
    ...     train_on(input_nodes, pair_graph, neg_pair_graph, blocks)

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
    >>> sampler = dgl.dataloading.as_edge_prediction_sampler(
    ...     sampler, exclude='reverse_types',
    ...     reverse_etypes={'click': 'clicked-by', 'clicked-by': 'click'})
    >>> dataloader = dgl.dataloading.EdgeDataLoader(
    ...     g, {'click': train_eid}, sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, pair_graph, blocks in dataloader:
    ...     train_on(input_nodes, pair_graph, blocks)

    To train a 3-layer GNN for link prediction on a set of edges ``train_eid`` with type
    ``click``, you can write

    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    >>> neg_sampler = dgl.dataloading.negative_sampler.Uniform(5)
    >>> sampler = dgl.dataloading.as_edge_prediction_sampler(
    ...     sampler, exclude='reverse_types',
    ...     reverse_etypes={'click': 'clicked-by', 'clicked-by': 'click'},
    ...     negative_sampler=neg_sampler)
    >>> dataloader = dgl.dataloading.EdgeDataLoader(
    ...     g, train_eid, sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, pos_pair_graph, neg_pair_graph, blocks in dataloader:
    ...     train_on(input_nodes, pair_graph, neg_pair_graph, blocks)
    """
    return EdgePredictionSampler(
        sampler, exclude=exclude, reverse_eids=reverse_eids, reverse_etypes=reverse_etypes,
        negative_sampler=negative_sampler, prefetch_labels=prefetch_labels)
