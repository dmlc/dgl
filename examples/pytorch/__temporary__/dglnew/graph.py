import dgl

class GraphStorage(object):
    @property
    def ndata(self):
        pass

    @property
    def ntypes(self):
        pass

    def sample_neighbors(self, seed_nodes, fanout, edge_dir='in', prob=None, replace=False,
                         output_device=None):
        pass

    def edge_subgraph(self, edges):
        pass

class _EidExcluder():
    def __init__(self, exclude_eids):
        device = None
        if isinstance(exclude_eids, Mapping):
            for _, v in exclude_eids.items():
                if device is None:
                    device = F.context(v)
                    break
        else:
            device = F.context(exclude_eids)
        self._exclude_eids = None
        self._filter = None

        if device == F.cpu():
            # TODO(nv-dlasalle): Once Filter is implemented for the CPU, we
            # should just use that irregardless of the device.
            self._exclude_eids = (
                _tensor_or_dict_to_numpy(exclude_eids) if exclude_eids is not None else None)
        else:
            if isinstance(exclude_eids, Mapping):
                self._filter = {k: utils.Filter(v) for k, v in exclude_eids.items()}
            else:
                self._filter = utils.Filter(exclude_eids)

    def _find_indices(self, parent_eids):
        """ Find the set of edge indices to remove.
        """
        if self._exclude_eids is not None:
            parent_eids_np = _tensor_or_dict_to_numpy(parent_eids)
            return _locate_eids_to_exclude(parent_eids_np, self._exclude_eids)
        else:
            assert self._filter is not None
            if isinstance(parent_eids, Mapping):
                located_eids = {k: self._filter[k].find_included_indices(parent_eids[k])
                                for k, v in parent_eids.items() if k in self._filter}
            else:
                located_eids = self._filter.find_included_indices(parent_eids)
            return located_eids

    def __call__(self, frontier):
        parent_eids = frontier.edata[EID]
        located_eids = self._find_indices(parent_eids)

        if not isinstance(located_eids, Mapping):
            # (BarclayII) If frontier already has a EID field and located_eids is empty,
            # the returned graph will keep EID intact.  Otherwise, EID will change
            # to the mapping from the new graph to the old frontier.
            # So we need to test if located_eids is empty, and do the remapping ourselves.
            if len(located_eids) > 0:
                frontier = transform.remove_edges(
                    frontier, located_eids, store_ids=True)
                frontier.edata[EID] = F.gather_row(parent_eids, frontier.edata[EID])
        else:
            # (BarclayII) remove_edges only accepts removing one type of edges,
            # so I need to keep track of the edge IDs left one by one.
            new_eids = parent_eids.copy()
            for k, v in located_eids.items():
                if len(v) > 0:
                    frontier = transform.remove_edges(
                        frontier, v, etype=k, store_ids=True)
                    new_eids[k] = F.gather_row(parent_eids[k], frontier.edges[k].data[EID])
            frontier.edata[EID] = new_eids
        return frontier

class DGLGraphStorage(GraphStorage):
    # A thin wrapper of DGLGraph that makes it a GraphStorage
    def __init__(self, g):
        self.g = g

    @property
    def ntypes(self):
        return self.g.ntypes

    @property
    def ndata(self):
        return self.g.ndata

    def sample_neighbors(self, seed_nodes, fanout, edge_dir='in', prob=None, replace=False,
                         exclude_edges=None, output_device=None):
        if self.g.device == 'cpu':
            frontier = dgl.sampling.sample_neighbors(
                self.g, seed_nodes, fanout, edge_dir=edge_dir, prob=prob, replace=replace,
                exclude_edges=exclude_edges)
        else:
            frontier = dgl.sampling.sample_neighbors(
                self.g, seed_nodes, fanout, edge_dir=edge_dir, prob=prob, replace=replace)
            if exclude_edges is not None:
                eid_excluder = _EidExcluder(exclude_edges)
                frontier = eid_excluder(frontier)
        return frontier.to(output_device)

    def edge_subgraph(self, edges, output_device=None):
        return self.g.edge_subgraph(edges, relabel_nodes=False).to(output_device)
