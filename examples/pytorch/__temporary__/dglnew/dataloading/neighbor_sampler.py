import dgl
from .sampler import Sampler

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
    elif exclude_mode == 'self':
        return eids
    elif exclude_mode == 'reverse_id':
        return _find_exclude_eids_with_reverse_id(g, eids, kwargs['reverse_eid_map'])
    elif exclude_mode == 'reverse_types':
        return _find_exclude_eids_with_reverse_types(g, eids, kwargs['reverse_etype_map'])
    else:
        raise ValueError('unsupported mode {}'.format(exclude_mode))

class NeighborSampler(object):
    def __init__(self, fanouts, edge_dir='in', prob=None, replace=False, output_device=None, **kwargs):
        super().__init__()
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.prob = prob
        self.replace = replace
        self.output_device = output_device

        self.inputs = []
        self.outputs = []
        self.edata = []

    def sample(self, g, seed_nodes, exclude_edges=None):
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes, fanout, edge_dir=self.edge_dir, prob=self.prob,
                replace=self.replace, output_device=self.output_device,
                exclude_edges=exclude_edges)
            eid = frontier.edata[dgl.EID]
            block = dgl.to_block(frontier, seed_nodes)
            block.edata[dgl.EID] = eid
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)

        blocks[0].srcdata.mark(self.inputs)
        blocks[-1].dstdata.mark(self.outputs)
        for block in blocks:
            block.edata.mark(self.edata)

        return blocks

    def add_input(self, name):
        self.inputs.append(name)

    def add_output(self, name):
        self.outputs.append(name)

    def add_edata(self, edata):
        self.edata.append(name)

def find_exclude_eids(seed_edges, exclude, g_sampling, reverse_eids=None,
                      reverse_etypes=None, output_device=None):
    # find the edges to exclude
    if exclude is not None:
        exclude_eids = _find_exclude_eids(
            g_sampling,
            exclude,
            seed_edges,
            reverse_eid_map=reverse_eids,
            reverse_etype_map=reverse_etypes)
        exclude_eids = recursive_apply(
            exclude_eids, lambda x: x.to(output_device))

    return exclude_eids


class EdgeWrapper(object):
    def __init__(self, neighbor_sampler, exclude=None, reverse_eids=None,
                 reverse_etypes=None):
        self.reverse_eids = reverse_eids
        self.reverse_etypes = reverse_etypes
        self.exclude = exclude
        self.neighbor_sampler = neighbor_sampler

        self.inputs = []
        self.outputs = []
        self.edata = []

    def sample(self, g, seed_edges, g_sampling=None):
        exclude = self.exclude if g_sampling is None else None
        g_sampling = g_sampling or g
        pair_graph = g.edge_subgraph(seed_edges, self.output_device)
        eids = pair_graph.edata[dgl.EID]
        pair_graph = dgl.compact_graphs(pair_graph)
        pair_graph.edata[dgl.EID] = eids
        seed_nodes = pair_graph.ndata[dgl.NID]

        exclude_eids = find_exclude_eids(
            g, seed_edges, g_sampling, self.reverse_eids, self.reverse_etypes,
            self.output_device)

        blocks = self.neighbor_sampler(g_sampling, seed_nodes, exclude_eids)

        blocks[0].srcdata.mark(self.inputs)
        pair_graph.edata.mark(self.outputs)
        for block in blocks:
            block.edata.mark(self.edata)
        return pair_graph, blocks


class LinkWrapper(object):
    def __init__(self, neighbor_sampler, exclude=None, reverse_eids=None,
                 reverse_etypes=None, negative_sampler=None):
        self.reverse_eids = reverse_eids
        self.reverse_etypes = reverse_etypes
        self.exclude = exclude
        self.negative_sampler = negative_sampler
        self.neighbor_sampler = neighbor_sampler

        self.inputs = []
        self.edata = []

    def _build_neg_graph(self, g, seed_edges):
        neg_srcdst = self.negative_sampler(g, seed_edges)
        if not isinstance(neg_srcdst, Mapping):
            assert len(g.etypes) == 1, \
                'graph has multiple or no edge types; '\
                'please return a dict in negative sampler.'
            neg_srcdst = {g.canonical_etypes[0]: neg_srcdst}

        dtype = F.dtype(list(neg_srcdst.values())[0][0])
        neg_edges = {
            etype: neg_srcdst.get(etype, (F.tensor([], dtype), F.tensor([], dtype)))
            for etype in g.canonical_etypes}
        neg_pair_graph = heterograph(
            neg_edges, {ntype: g.number_of_nodes(ntype) for ntype in g.ntypes})
        return neg_pair_graph

    def sample(self, g, seed_edges, g_sampling=None):
        exclude = self.exclude if g_sampling is None else None
        g_sampling = g_sampling or g
        pair_graph = g.edge_subgraph(seed_edges, self.output_device)
        eids = pair_graph.edata[dgl.EID]

        neg_graph = self._build_neg_graph(g, seed_edges)

        pair_graph, neg_graph = dgl.compact_graphs([pair_graph, neg_graph])
        pair_graph.edata[dgl.EID] = eids
        seed_nodes = pair_graph.ndata[dgl.NID]

        exclude_eids = find_exclude_eids(
            g, seed_edges, g_sampling, self.reverse_eids, self.reverse_etypes,
            self.output_device)

        blocks = self.neighbor_sampler(g_sampling, seed_nodes, exclude_eids)

        blocks[0].srcdata.mark(self.inputs)
        for block in blocks:
            block.edata.mark(self.edata)
        return pair_graph, neg_graph, blocks
