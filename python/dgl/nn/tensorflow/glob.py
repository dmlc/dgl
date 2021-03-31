"""Tensorflow modules for graph global pooling."""
# pylint: disable= no-member, arguments-differ, invalid-name, W0235
import tensorflow as tf
from tensorflow.keras import layers


from ...readout import sum_nodes, mean_nodes, max_nodes, \
    softmax_nodes, topk_nodes


__all__ = ['SumPooling', 'AvgPooling',
           'MaxPooling', 'SortPooling', 'WeightAndSum', 'GlobalAttentionPooling']


class SumPooling(layers.Layer):
    r"""Apply sum pooling over the nodes in the graph.

    .. math::
        r^{(i)} = \sum_{k=1}^{N_i} x^{(i)}_k
    """

    def __init__(self):
        super(SumPooling, self).__init__()

    def call(self, graph, feat):
        r"""Compute sum pooling.


        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : tf.Tensor
            The input feature with shape :math:`(N, *)` where
            :math:`N` is the number of nodes in the graph.

        Returns
        -------
        tf.Tensor
            The output feature with shape :math:`(B, *)`, where
            :math:`B` refers to the batch size.
        """
        with graph.local_scope():
            graph.ndata['h'] = feat
            readout = sum_nodes(graph, 'h')
            return readout


class AvgPooling(layers.Layer):
    r"""Apply average pooling over the nodes in the graph.

    .. math::
        r^{(i)} = \frac{1}{N_i}\sum_{k=1}^{N_i} x^{(i)}_k
    """

    def __init__(self):
        super(AvgPooling, self).__init__()

    def call(self, graph, feat):
        r"""Compute average pooling.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : tf.Tensor
            The input feature with shape :math:`(N, *)` where
            :math:`N` is the number of nodes in the graph.

        Returns
        -------
        tf.Tensor
            The output feature with shape :math:`(B, *)`, where
            :math:`B` refers to the batch size.
        """
        with graph.local_scope():
            graph.ndata['h'] = feat
            readout = mean_nodes(graph, 'h')
            return readout


class MaxPooling(layers.Layer):
    r"""Apply max pooling over the nodes in the graph.

    .. math::
        r^{(i)} = \max_{k=1}^{N_i}\left( x^{(i)}_k \right)
    """

    def __init__(self):
        super(MaxPooling, self).__init__()

    def call(self, graph, feat):
        r"""Compute max pooling.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : tf.Tensor
            The input feature with shape :math:`(N, *)` where
            :math:`N` is the number of nodes in the graph.

        Returns
        -------
        tf.Tensor
            The output feature with shape :math:`(B, *)`, where
            :math:`B` refers to the batch size.
        """
        with graph.local_scope():
            graph.ndata['h'] = feat
            readout = max_nodes(graph, 'h')
            return readout


class SortPooling(layers.Layer):
    r"""Apply Sort Pooling (`An End-to-End Deep Learning Architecture for Graph Classification
    <https://www.cse.wustl.edu/~ychen/public/DGCNN.pdf>`__) over the nodes in the graph.

    Parameters
    ----------
    k : int
        The number of nodes to hold for each graph.
    """

    def __init__(self, k):
        super(SortPooling, self).__init__()
        self.k = k

    def call(self, graph, feat):
        r"""Compute sort pooling.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : tf.Tensor
            The input feature with shape :math:`(N, D)` where
            :math:`N` is the number of nodes in the graph.

        Returns
        -------
        tf.Tensor
            The output feature with shape :math:`(B, k * D)`, where
            :math:`B` refers to the batch size.
        """
        with graph.local_scope():
            # Sort the feature of each node in ascending order.
            feat = tf.sort(feat, -1)
            graph.ndata['h'] = feat
            # Sort nodes according to their last features.
            ret = tf.reshape(topk_nodes(graph, 'h', self.k, sortby=-1)[0], (
                -1, self.k * feat.shape[-1]))
            return ret


class GlobalAttentionPooling(layers.Layer):
    r"""Apply Global Attention Pooling (`Gated Graph Sequence Neural Networks
    <https://arxiv.org/abs/1511.05493.pdf>`__) over the nodes in the graph.

    .. math::
        r^{(i)} = \sum_{k=1}^{N_i}\mathrm{softmax}\left(f_{gate}
        \left(x^{(i)}_k\right)\right) f_{feat}\left(x^{(i)}_k\right)

    Parameters
    ----------
    gate_nn : tf.layers.Layer
        A neural network that computes attention scores for each feature.
    feat_nn : tf.layers.Layer, optional
        A neural network applied to each feature before combining them
        with attention scores.
    """

    def __init__(self, gate_nn, feat_nn=None):
        super(GlobalAttentionPooling, self).__init__()
        self.gate_nn = gate_nn
        self.feat_nn = feat_nn

    def call(self, graph, feat):
        r"""Compute global attention pooling.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : tf.Tensor
            The input feature with shape :math:`(N, D)` where
            :math:`N` is the number of nodes in the graph.

        Returns
        -------
        tf.Tensor
            The output feature with shape :math:`(B, *)`, where
            :math:`B` refers to the batch size.
        """
        with graph.local_scope():
            gate = self.gate_nn(feat)
            assert gate.shape[-1] == 1, "The output of gate_nn should have size 1 at the last axis."
            feat = self.feat_nn(feat) if self.feat_nn else feat

            graph.ndata['gate'] = gate
            gate = softmax_nodes(graph, 'gate')
            graph.ndata.pop('gate')

            graph.ndata['r'] = feat * gate
            readout = sum_nodes(graph, 'r')
            graph.ndata.pop('r')

            return readout


class WeightAndSum(layers.Layer):
    """Compute importance weights for atoms and perform a weighted sum.

    Parameters
    ----------
    in_feats : int
        Input atom feature size
    """

    def __init__(self, in_feats):
        super(WeightAndSum, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = tf.keras.Sequential(
            layers.Dense(1),
            layers.Activation(tf.nn.sigmoid)
        )

    def call(self, g, feats):
        """Compute molecule representations out of atom representations

        Parameters
        ----------
        g : DGLGraph
            DGLGraph with batch size B for processing multiple molecules in parallel
        feats : FloatTensor of shape (N, self.in_feats)
            Representations for all atoms in the molecules
            * N is the total number of atoms in all molecules

        Returns
        -------
        FloatTensor of shape (B, self.in_feats)
            Representations for B molecules
        """
        with g.local_scope():
            g.ndata['h'] = feats
            g.ndata['w'] = self.atom_weighting(g.ndata['h'])
            h_g_sum = sum_nodes(g, 'h', 'w')

        return h_g_sum
