"""tf modules for graph related softmax."""
# pylint: disable= no-member, arguments-differ
import tensorflow as tf

from ... import function as fn
from ...base import ALL, is_all

__all__ = ['edge_softmax']


def edge_softmax_real(graph, score, eids=ALL):
    """Edge Softmax function"""
    if not is_all(eids):
        graph = graph.edge_subgraph(tf.cast(eids, tf.int64))
    g = graph.local_var()
    g.edata['s'] = score
    g.update_all(fn.copy_e('s', 'm'), fn.max('m', 'smax'))
    g.apply_edges(fn.e_sub_v('s', 'smax', 'out'))
    g.edata['out'] = tf.math.exp(g.edata['out'])
    g.update_all(fn.copy_e('out', 'm'), fn.sum('m', 'out_sum'))
    g.apply_edges(fn.e_div_v('out', 'out_sum', 'out'))
    out = g.edata['out']

    def edge_softmax_backward(grad_out):
        g = graph.local_var()
        # clear backward cache explicitly
        g.edata['out'] = out
        g.edata['grad_s'] = out * grad_out
        g.update_all(fn.copy_e('grad_s', 'm'), fn.sum('m', 'accum'))
        g.apply_edges(fn.e_mul_v('out', 'accum', 'out'))
        grad_score = g.edata['grad_s'] - g.edata['out']
        return grad_score

    return out, edge_softmax_backward


def edge_softmax(graph, logits, eids=ALL):
    """Closure for tf.custom_gradient"""

    @tf.custom_gradient
    def _lambda(logits):
        return edge_softmax_real(graph, logits, eids=eids)

    return _lambda(logits)
