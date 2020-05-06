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
    with graph.local_scope():
        graph.edata['s'] = score
        graph.update_all(fn.copy_e('s', 'm'), fn.max('m', 'smax'))
        graph.apply_edges(fn.e_sub_v('s', 'smax', 'out'))
        graph.edata['out'] = tf.math.exp(graph.edata['out'])
        graph.update_all(fn.copy_e('out', 'm'), fn.sum('m', 'out_sum'))
        graph.apply_edges(fn.e_div_v('out', 'out_sum', 'out'))
        out = graph.edata['out']

    def edge_softmax_backward(grad_out):
        with graph.local_scope():
            # clear backward cache explicitly
            graph.edata['out'] = out
            graph.edata['grad_s'] = out * grad_out
            graph.update_all(fn.copy_e('grad_s', 'm'), fn.sum('m', 'accum'))
            graph.apply_edges(fn.e_mul_v('out', 'accum', 'out'))
            grad_score = graph.edata['grad_s'] - graph.edata['out']
            return grad_score

    return out, edge_softmax_backward


def edge_softmax(graph, logits, eids=ALL):
    """Closure for tf.custom_gradient"""

    @tf.custom_gradient
    def _lambda(logits):
        return edge_softmax_real(graph, logits, eids=eids)

    return _lambda(logits)
