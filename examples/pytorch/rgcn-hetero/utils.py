"""
Utility functions for link prediction
"""

import numpy as np
import torch
import dgl


def build_graph_from_triplets(num_nodes, num_rels, edge_lists, reverse=True):
    """ Create a DGL Hetero graph. 
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """
    src = []
    rel = []
    dst = []
    raw_subg = {}

    # here there is noly one node type
    s_type = "node"
    d_type = "node"
    for edges in edge_lists:
        for edge in edges:
            s, r, d = edge
            r_type = str(r)
            e_type = (s_type, r_type, d_type)

            if raw_subg.get(e_type, None) is None:
                raw_subg[e_type] = ([], [])
            raw_subg[e_type][0].append(s)
            raw_subg[e_type][1].append(d)

            if reverse is True:
                r_type = str(r + num_rels)
                re_type = (d_type, r_type, s_type)
                raw_subg[re_type] = ([], [])
                raw_subg[re_type][0].append(d)
                raw_subg[re_type][1].append(s)

    subg = []
    for e_type, val in raw_subg.items():
        s_type, r_type, d_type = e_type
        s, d = val
        s = np.asarray(s)
        d = np.asarray(d)

        subg.append(dgl.graph((s, d),
                              s_type,
                              r_type,
                              num_nodes=num_nodes))
    g = dgl.hetero_from_relations(subg)

    return g

# According to https://github.com/pytorch/pytorch/issues/17199, this decorator
# is necessary to make fork() and openmp work together.
def thread_wrapped_func(func):
    """
    Wraps a process entry point to make it work with OpenMP.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function
