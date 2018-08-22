"""Built-in functors."""
from __future__ import absolute_import

import dgl.backend as F

def message_from_src(src, edge):
    return src

def reduce_sum(node, msgs):
    if isinstance(msgs, list):
        if isinstance(msgs[0], dict):
            return {k : sum(m[k] for m in msgs) for k in msgs[0].keys()}
        else:
            return sum(msgs)
    else:
        return F.sum(msgs, 1)

def reduce_max(node, msgs):
    if isinstance(msgs, list):
        return max(msgs)
    else:
        return F.max(msgs, 1)
