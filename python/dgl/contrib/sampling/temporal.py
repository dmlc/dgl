"""Temporal neighbor sampling APIs"""

from ... import backend as F
from ... import ndarray as nd
from ..._ffi.function import _init_api

def _to_nd_dict(pt_dict):
    """Convert a dictionary of torch tensors to a dictionary of DGL NDArrays."""
    return {key : F.to_dgl_nd(tensor) for key, tensor in pt_dict.items()}

def temporal_sample_neighbors(
    g,
    nodes,
    fanout,
    timestamp,
    replace=False
):
    """Sample neighbors of the given nodes constrained by node timestamps.

    For each node, only the neighbors with smaller or equal timestamp will be sampled.

    Parameters
    ----------
    g : DGLGraph
    nodes : torch.Tensor or dict[ntype, torch.Tensor]
    fanout : int or dict[etype, int]
        The number of neighbors to be sampled for each edge type.
    timestamp : torch.Tensor or dict[ntype, torch.Tensor]
        Node timestamp tensor. Timestamps are stored in long integers using
        the Unix timestamp format.
    replace : bool, optional
        If True, sample with replacement.
    """
    # Handle non-dict input.
    if F.is_tensor(nodes):
        assert len(g.ntypes) == 1
        nodes = {g.ntypes[0] : nodes}
    if F.is_tensor(timestamp):
        assert len(g.ntypes) == 1
        timestamp = {g.ntypes[0] : timestamp}

    if isinstance(fanout, int):
        fanout_list = [fanout] * len(g.canonical_etypes)
    else:
        fanout_list = [fanout[etype] for etype in g.canonical_etypes]

    sample_rst = _CAPI_DGLTemporalSampleNeighbors(
        g._graph,
        g.ntypes,
        _to_nd_dict(nodes),
        fanout_list,
        _to_nd_dict(timestamp),
        replace
    )

_init_api("dgl.contrib.sampling.temporal", __name__)
