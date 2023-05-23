"""Temporal neighbor sampling APIs"""

from ... import backend as F, DGLGraph, ndarray as nd
from ..._ffi.function import _init_api

__all__ = ["temporal_sample_neighbors"]


def _to_nd_dict(pt_dict):
    """Convert a dictionary of torch tensors to a dictionary of DGL NDArrays."""
    return {key: F.to_dgl_nd(tensor) for key, tensor in pt_dict.items()}


def temporal_sample_neighbors(
    g, nodes, fanout, timestamp, replace=False, return_eid=False
):
    """Sample neighbors of the given nodes constrained by node timestamps.

    For each node, only the neighbors with smaller or equal timestamp will be sampled.

    Parameters
    ----------
    g : DGLGraph
        The graph to sample on.
    nodes : torch.Tensor or dict[ntype, torch.Tensor]
        Seed nodes.
    fanout : int or dict[etype, int]
        The number of neighbors to be sampled for each edge type.
    timestamp : torch.Tensor or dict[ntype, torch.Tensor]
        Node timestamp tensor. Timestamps are stored in long integers using
        the Unix timestamp format.
    replace : bool, optional
        If True, sample with replacement.
    return_eid : bool, optional
        If True, return the ID of the sampled edges.
    """
    # Handle non-dict input.
    if F.is_tensor(nodes):
        assert len(g.ntypes) == 1
        nodes = {g.ntypes[0]: nodes}
    if F.is_tensor(timestamp):
        assert len(g.ntypes) == 1
        timestamp = {g.ntypes[0]: timestamp}

    if isinstance(fanout, int):
        fanout_list = [fanout] * len(g.canonical_etypes)
    else:
        fanout_list = [fanout[etype] for etype in g.canonical_etypes]

    subgidx = _CAPI_DGLTemporalSampleNeighbors(
        g._graph,
        g.ntypes,
        _to_nd_dict(nodes),
        fanout_list,
        _to_nd_dict(timestamp),
        replace,
    )
    induced_edges = subgidx.induced_edges
    ret = DGLGraph(subgidx.graph, g.ntypes, g.etypes)

    if return_eid:
        return ret, induced_edges
    else:
        return ret


_init_api("dgl.contrib.sampling.temporal", __name__)
