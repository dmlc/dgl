"""tf modules for graph related softmax."""
# pylint: disable= no-member, arguments-differ
import tensorflow as tf

from ...sparse import _gspmm, _gsddmm
from ...base import ALL, is_all

__all__ = ['edge_softmax']


def edge_softmax_real(graph, score, eids=ALL):
    """Edge Softmax function"""
    if not is_all(eids):
        graph = graph.edge_subgraph(tf.cast(eids, graph.idtype), preserve_nodes=True)
    gidx = graph._graph
    score_max = _gspmm(gidx, 'copy_rhs', 'max', None, score)[0]
    score = tf.math.exp(_gsddmm(gidx, 'sub', score, score_max, 'e', 'v'))
    score_sum = _gspmm(gidx, 'copy_rhs', 'sum', None, score)[0]
    out = _gsddmm(gidx, 'div', score, score_sum, 'e', 'v')

    def edge_softmax_backward(grad_out):
        sds = out * grad_out
        accum = _gspmm(gidx, 'copy_rhs', 'sum', None, sds)[0]
        grad_score = sds - _gsddmm(gidx, 'mul', out, accum, 'e', 'v')
        return grad_score

    return out, edge_softmax_backward


def edge_softmax(graph, logits, eids=ALL):
    """Closure for tf.custom_gradient"""

    @tf.custom_gradient
    def _lambda(logits):
        return edge_softmax_real(graph, logits, eids=eids)

    return _lambda(logits)
