class BlockSampler(object):
    """BlockSampler is an abstract class assuming to take in a set of nodes whose
    outputs are to compute, and return a list of blocks.

    Moreover, it assumes that the input node features will be put in the first block's
    ``srcdata``, the output node labels will be put in the last block's ``dstdata``, and
    the edge data will be put in all the blocks' ``edata``.
    """
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.edata = []
        self._frozen = False

    def _freeze(self):
        self._frozen = True

    def sample_blocks(self, g, seed_nodes, exclude_edges=None):
        raise NotImplementedError

    def sample(self, g, seed_nodes):
        blocks = self.sample_blocks(g, seed_nodes)

        blocks[0].srcdata.mark(self.inputs)
        # Or equivalently
        #
        #     for name in self.inputs:
        #         blocks[0].srcdata[name] = Marker()
        #
        # Marker's signature is
        #
        #     Marker(name: str, id_: tensor)
        #
        # Both arguments can be None.  In this case, the name and ID will be inferred
        # from where it is assigned to.  E.g.
        #
        #     blocks[0].srcdata[name] = Marker()
        #
        # is equivalent to
        #
        #     blocks[0].srcdata[name] = Marker(name, blocks[0].srcdata[dgl.NID])
        blocks[-1].dstdata.mark(self.outputs)
        for block in blocks:
            block.edata.mark(self.edata)
        # If you want to prefetch things other than ndata and edata, you can also
        # return a Marker(name, id_).  If a Marker is returned in places other than
        # in a graph's ndata/edata/srcdata/dstdata, the DataLoader will prefetch it
        # from its dictionary ``other_data``.
        # For instance, you can run
        #
        #     return blocks, Marker('other_feat', id_)
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
        return blocks

    def add_input(self, name):
        assert not self._frozen, \
            "Cannot call this method after the dataloader has been created.  "
            "Please call this method before the creation of dataloader."
        self.inputs.append(name)

    def add_output(self, name):
        assert not self._frozen, \
            "Cannot call this method after the dataloader has been created.  "
            "Please call this method before the creation of dataloader."
        self.outputs.append(name)

    def add_edata(self, edata):
        assert not self._frozen, \
            "Cannot call this method after the dataloader has been created.  "
            "Please call this method before the creation of dataloader."
        self.edata.append(name)


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
    reverse_eid_map : Tensor or dict[etype, Tensor]
        The mapping from edge ID to its reverse edge ID.
    reverse_etype_map : dict[etype, etype]
        The mapping from edge etype to its reverse edge type.
    """
    if exclude_mode is None:
        return None
    elif F.is_tensor(exclude_mode) or all(recursive_apply(exclude_mode, F.is_tensor)):
        return exclude_mode
    elif exclude_mode == 'self':
        return eids
    elif exclude_mode == 'reverse_id':
        return _find_exclude_eids_with_reverse_id(g, eids, kwargs['reverse_eid_map'])
    elif exclude_mode == 'reverse_types':
        return _find_exclude_eids_with_reverse_types(g, eids, kwargs['reverse_etype_map'])
    else:
        raise ValueError('unsupported mode {}'.format(exclude_mode))

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
    """EdgeWrapper adapts a node-wise BlockSampler for edge classification.

    It assumes that the neighbor sampler will return a list of blocks, and it will return

    * A subgraph that contains only the edges in the minibatch and their incident nodes.
      Note that the graph has an identical metagraph with the original graph.

    * The list of blocks returned by the neighbor sampler.

    It assumes that the input node features will be put in the first block's
    ``srcdata``, the output edge labels will be put in the last block's ``dstdata``, and
    the edge data will be put in all the blocks' ``edata``.
    """
    def __init__(self, neighbor_sampler, exclude=None, reverse_eids=None,
                 reverse_etypes=None):
        self.reverse_eids = reverse_eids
        self.reverse_etypes = reverse_etypes
        self.exclude = exclude
        self.neighbor_sampler = neighbor_sampler

        self.inputs = neighbor_sampler.inputs
        self.outputs = neighbor_sampler.outputs
        self.edata = neighbor_sampler.edata
        self._frozen = False

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

        blocks = self.neighbor_sampler.sample_blocks(g_sampling, seed_nodes, exclude_eids)

        blocks[0].srcdata.mark(self.inputs)
        pair_graph.edata.mark(self.outputs)
        for block in blocks:
            block.edata.mark(self.edata)
        return pair_graph, blocks

    def _freeze(self):
        self._frozen = True

    def add_input(self, name):
        assert not self._frozen, \
            "Cannot call this method after the dataloader has been created.  "
            "Please call this method before the creation of dataloader."
        self.inputs.append(name)

    def add_output(self, name):
        assert not self._frozen, \
            "Cannot call this method after the dataloader has been created.  "
            "Please call this method before the creation of dataloader."
        self.outputs.append(name)

    def add_edata(self, edata):
        assert not self._frozen, \
            "Cannot call this method after the dataloader has been created.  "
            "Please call this method before the creation of dataloader."
        self.edata.append(name)


class LinkWrapper(object):
    """LinkWrapper adapts a node-wise BlockSampler for link prediction.

    It assumes that the neighbor sampler will return a list of blocks, and it will return

    * A subgraph that contains only the edges in the minibatch and their incident nodes.
      Note that the graph has an identical metagraph with the original graph.

    * Another graph that contains the "negative edges", connecting the source and
      destination nodes yielded from the given negative sampler.

    * The list of blocks returned by the neighbor sampler.

    It assumes that the input node features will be put in the first block's
    ``srcdata``, and the edge data will be put in all the blocks' ``edata``.
    """
    def __init__(self, neighbor_sampler, exclude=None, reverse_eids=None,
                 reverse_etypes=None, negative_sampler=None):
        self.reverse_eids = reverse_eids
        self.reverse_etypes = reverse_etypes
        self.exclude = exclude
        self.negative_sampler = negative_sampler
        self.neighbor_sampler = neighbor_sampler

        self.inputs = neighbor_sampler.inputs
        self.edata = neighbor_sampler.edata
        self._frozen = True

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

        blocks = self.neighbor_sampler.sample_blocks(g_sampling, seed_nodes, exclude_eids)

        blocks[0].srcdata.mark(self.inputs)
        for block in blocks:
            block.edata.mark(self.edata)
        return pair_graph, neg_graph, blocks

    def _freeze(self):
        self._frozen = True

    def add_input(self, name):
        assert not self._frozen, \
            "Cannot call this method after the dataloader has been created.  "
            "Please call this method before the creation of dataloader."
        self.inputs.append(name)

    def add_edata(self, edata):
        assert not self._frozen, \
            "Cannot call this method after the dataloader has been created.  "
            "Please call this method before the creation of dataloader."
        self.edata.append(name)
