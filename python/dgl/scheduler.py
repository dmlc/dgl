"""Schedule policies for graph computation."""
from __future__ import absolute_import

import numpy as np

import dgl.backend as F
import dgl.utils as utils

def degree_bucketing(cached_graph, v):
    """Create degree bucketing scheduling policy.

    Parameters
    ----------
    cached_graph : dgl.cached_graph.CachedGraph
        the graph
    v : dgl.utils.Index
        the nodes to gather messages

    Returns
    -------
    unique_degrees : list of int
        list of unique degrees
    v_bkt : list of dgl.utils.Index
        list of node id buckets; nodes belong to the same bucket have
        the same degree
    """
    degrees = F.asnumpy(cached_graph.in_degrees(v).totensor())
    unique_degrees = list(np.unique(degrees))
    v_np = np.array(v.tolist())
    v_bkt = []
    for deg in unique_degrees:
        idx = np.where(degrees == deg)
        v_bkt.append(utils.Index(v_np[idx]))
    #print('degree-bucketing:', unique_degrees, [len(b) for b in v_bkt])
    return unique_degrees, v_bkt
