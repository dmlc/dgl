"""Schedule policies for graph computation."""
from __future__ import absolute_import

import dgl.backend as F

def degree_bucketing(cached_graph, v):
    degrees = cached_graph.in_degrees(v)
    unique_degrees = list(F.asnumpy(F.unique(degrees)))
    v_bkt = []
    for deg in unique_degrees:
        idx = F.squeeze(F.nonzero(F.eq_scalar(degrees, deg)), 1)
        v_bkt.append(v[idx])
    return unique_degrees, v_bkt
