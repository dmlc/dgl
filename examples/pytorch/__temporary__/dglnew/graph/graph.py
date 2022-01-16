import dgl
from collections import Mapping
from dgl import backend as F
from dgl import utils, transform
from dgl.base import EID
from dgl.utils import recursive_apply, recursive_apply_pair
import numpy as np

def _locate_eids_to_exclude(frontier_parent_eids, exclude_eids):
    """Find the edges whose IDs in parent graph appeared in exclude_eids.

    Note that both arguments are numpy arrays or numpy dicts.
    """
    func = lambda x, y: np.isin(x, y).nonzero()[0]
    result = recursive_apply_pair(frontier_parent_eids, exclude_eids, func)
    return recursive_apply(result, F.zerocopy_from_numpy)


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
                recursive_apply(exclude_eids, F.zerocopy_to_numpy)
                if exclude_eids is not None else None)
        else:
            self._filter = recursive_apply(exclude_eids, utils.Filter)

    def _find_indices(self, parent_eids):
        """ Find the set of edge indices to remove.
        """
        if self._exclude_eids is not None:
            parent_eids_np = recursive_apply(parent_eids, F.zerocopy_to_numpy)
            return _locate_eids_to_exclude(parent_eids_np, self._exclude_eids)
        else:
            assert self._filter is not None
            func = lambda x, y: x.find_included_indices(y)
            return recursive_apply_pair(self._filter, parent_eids, func)

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

class DGLGraphStorage(object):
    # A thin wrapper of DGLGraph that makes it a GraphStorage
    def __init__(self, g):
        self.g = g

    @property
    def ntypes(self):
        return self.g.ntypes

    @property
    def ndata(self):
        return self.g.ndata

    # Required in Link Prediction
    @property
    def etypes(self):
        return self.g.etypes

    # Required in Link Prediction
    @property
    def canonical_etypes(self):
        return self.g.canonical_etypes

    @property
    def edata(self):
        return self.g.edata

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
        return frontier if output_device is None else frontier.to(output_device)

    # Required in Link Prediction
    def edge_subgraph(self, edges, output_device=None):
        subg = self.g.edge_subgraph(edges, relabel_nodes=False)
        return subg if output_device is None else subg.to(output_device)

    # Required in Link Prediction negative sampler
    def find_edges(self, edges, etype=None, output_device=None):
        src, dst = self.g.find_edges(edges, etype=etype)
        if output_device is None:
            return src, dst
        else:
            return src.to(output_device), dst.to(output_device)

    # Required in Link Prediction negative sampler
    def num_nodes(self, ntype):
        return self.g.num_nodes(ntype)
