"""Schedule policies for graph computation."""
from __future__ import absolute_import

import dgl.backend as F
import numpy as np

def degree_bucketing(cached_graph, v):
    degrees = F.asnumpy(cached_graph.in_degrees(v))
    unique_degrees = list(np.unique(degrees))
    v_bkt = []
    for deg in unique_degrees:
        idx = np.where(degrees == deg)
        v_bkt.append(v[idx])
    return unique_degrees, v_bkt
