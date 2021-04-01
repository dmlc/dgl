"""Tensorflow modules for EdgeConv Layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import tensorflow as tf
from tensorflow.keras import layers

from .... import function as fn
from ....base import DGLError
from ....utils import expand_as_pair


class EdgeConv(layers.Layer):
    r"""
    Description
    -----------
    EdgeConv layer.
    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:
    .. math::
       h_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} (
       \Theta \cdot (h_j^{(l)} - h_i^{(l)}) + \Phi \cdot h_i^{(l)})
    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    :math:`\Theta` and :math:`\Phi` are linear layers.
    .. note::
       The original formulation includes a ReLU inside the maximum operator.
       This is equivalent to first applying a maximum operator then applying
       the ReLU.
    Parameters
    ----------
    in_feat : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    out_feat : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    batch_norm : bool
        Whether to include batch normalization on messages. Default: ``False``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Default: ``False``.
    Note
    ----
    Zero in-degree nodes will lead to invalid output value. This is because no message
    will be passed to those nodes, the aggregation function will be appied on empty input.
    A common practice to avoid this is to add a self-loop for each node in the graph if
    it is homogeneous, which can be achieved by:
    >>> g = ... # a DGLGraph
    >>> g = dgl.add_self_loop(g)
    Calling ``add_self_loop`` will not work for some graphs, for example, heterogeneous graph
    since the edge type can not be decided for self_loop edges. Set ``allow_zero_in_degree``
    to ``True`` for those cases to unblock the code and handle zere-in-degree nodes manually.
    A common practise to handle this is to filter out the nodes with zere-in-degree when use
    after conv.
    """
    def __init__(self,
                 out_feats,
                 batch_norm=False,
                 allow_zero_in_degree=False):
        super(EdgeConv, self).__init__()
        self.batch_norm = batch_norm
        self._allow_zero_in_degree = allow_zero_in_degree

        self.theta = layers.Dense(out_feats)
        self.phi = layers.Dense(out_feats)
        if batch_norm:
            self.bn = layers.BatchNormalization()

    def set_allow_zero_in_degree(self, set_value):
        r"""
        Description
        -----------
        Set allow_zero_in_degree flag.
        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def call(self, g, feat):
        """
        Description
        -----------
        Forward computation
        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : tf.Tensor or pair of tf.Tensor
            :math:`(N, D)` where :math:`N` is the number of nodes and
            :math:`D` is the number of feature dimensions.
            If a pair of tensors is given, the graph must be a uni-bipartite graph
            with only one edge type, and the two tensors must have the same
            dimensionality on all except the first axis.
        Returns
        -------
        tf.Tensor or pair of tf.Tensor
            New node features.
        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with g.local_scope():
            if not self._allow_zero_in_degree:
                if  tf.math.count_nonzero(g.in_degrees() == 0) > 0:
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            h_src, h_dst = expand_as_pair(feat, g)
            g.srcdata['x'] = h_src
            g.dstdata['x'] = h_dst
            g.apply_edges(fn.v_sub_u('x', 'x', 'theta'))
            g.edata['theta'] = self.theta(g.edata['theta'])
            g.dstdata['phi'] = self.phi(g.dstdata['x'])
            if not self.batch_norm:
                g.update_all(fn.e_add_v('theta', 'phi', 'e'), fn.max('e', 'x'))
            else:
                g.apply_edges(fn.e_add_v('theta', 'phi', 'e'))
                # for more comments on why global batch norm instead
                # of batch norm within EdgeConv go to
                # https://github.com/dmlc/dgl/blob/master/python/dgl/nn/pytorch/conv/edgeconv.py
                g.edata['e'] = self.bn(g.edata['e'])
                g.update_all(fn.copy_e('e', 'e'), fn.max('e', 'x'))
            return g.dstdata['x']
