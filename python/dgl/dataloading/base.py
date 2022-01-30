"""Base classes and functionalities for dataloaders"""
from collections import Mapping
from ..base import NID, EID
from ..convert import heterograph
from .. import backend as F
from ..transform import compact_graphs
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
    """Set lazy features for ``g.ndata`` if :attr:`feature_names` is a list of strings,
    or ``g.nodes[ntype].data`` if :attr:`feature_names` is a dict of list of strings.
    """
    return _set_lazy_features(g.nodes, g.ndata, feature_names)

def set_edge_lazy_features(g, feature_names):
    """Set lazy features for ``g.edata`` if :attr:`feature_names` is a list of strings,
    or ``g.edges[etype].data`` if :attr:`feature_names` is a dict of list of strings.
    """
    return _set_lazy_features(g.edges, g.edata, feature_names)

def set_src_lazy_features(g, feature_names):
    """Set lazy features for ``g.srcdata`` if :attr:`feature_names` is a list of strings,
    or ``g.srcnodes[srctype].data`` if :attr:`feature_names` is a dict of list of strings.
    """
    return _set_lazy_features(g.srcnodes, g.srcdata, feature_names)

def set_dst_lazy_features(g, feature_names):
    """Set lazy features for ``g.dstdata`` if :attr:`feature_names` is a list of strings,
    or ``g.dstnodes[dsttype].data`` if :attr:`feature_names` is a dict of list of strings.
    """
    return _set_lazy_features(g.dstnodes, g.dstdata, feature_names)

class BlockSampler(object):
    """BlockSampler is an abstract class assuming to take in a set of nodes whose
    outputs are to compute, and return a list of blocks.

    Moreover, it assumes that the input node features will be put in the first block's
    ``srcdata``, the output node labels will be put in the last block's ``dstdata``, and
    the edge data will be put in all the blocks' ``edata``.
    """
    def __init__(self, prefetch_node_feats=None, prefetch_labels=None,
                 prefetch_edge_feats=None, output_device=None):
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

    def sample(self, g, seed_nodes):
        """Sample a list of blocks from the given seed nodes."""
        result = self.sample_blocks(g, seed_nodes)
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

        Tensor or dict[etype, Tensor]
            Exclude the given edge IDs.

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
    if exclude_eids is not None:
        exclude_eids = recursive_apply(
            exclude_eids, lambda x: x.to(output_device))
    return exclude_eids


class EdgeBlockSampler(object):
    """Adapts a :class:`BlockSampler` object's :attr:`sample` method for edge
    classification and link prediction.
    """
    def __init__(self, block_sampler, exclude=None, reverse_eids=None,
                 reverse_etypes=None, negative_sampler=None, prefetch_node_feats=None,
                 prefetch_labels=None, prefetch_edge_feats=None):
        self.reverse_eids = reverse_eids
        self.reverse_etypes = reverse_etypes
        self.exclude = exclude
        self.block_sampler = block_sampler
        self.negative_sampler = negative_sampler
        self.prefetch_node_feats = prefetch_node_feats or []
        self.prefetch_labels = prefetch_labels or []
        self.prefetch_edge_feats = prefetch_edge_feats or []
        self.output_device = block_sampler.output_device

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
        blocks = result[-1]

        set_src_lazy_features(blocks[0], self.prefetch_node_feats)
        set_edge_lazy_features(pair_graph, self.prefetch_labels)
        for block in blocks:
            set_edge_lazy_features(block, self.prefetch_edge_feats)
        # In-place updates
        return result

    def sample(self, g, seed_edges):
        """Samples a list of blocks, as well as a subgraph containing the sampled
        edges from the original graph.

        If :attr:`negative_sampler` is given, also returns another graph containing the
        negative pairs as edges.
        """
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

        input_nodes, _, blocks = self.block_sampler.sample_blocks(g, seed_nodes, exclude_eids)

        if self.negative_sampler is None:
            return self.assign_lazy_features((input_nodes, pair_graph, blocks))
        else:
            return self.assign_lazy_features((input_nodes, pair_graph, neg_graph, blocks))
