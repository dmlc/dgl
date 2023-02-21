"""Torch Module for EdgeConv Layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
from torch import nn

from .... import function as fn

from ....base import DGLError
from ....utils import expand_as_pair


class EdgeConv(nn.Module):
    r"""EdgeConv layer from `Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__

    It can be described as follows:

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
    to ``True`` for those cases to unblock the code and handle zero-in-degree nodes manually.
    A common practise to handle this is to filter out the nodes with zero-in-degree when use
    after conv.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import EdgeConv

    >>> # Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = th.ones(6, 10)
    >>> conv = EdgeConv(10, 2)
    >>> res = conv(g, feat)
    >>> res
    tensor([[-0.2347,  0.5849],
            [-0.2347,  0.5849],
            [-0.2347,  0.5849],
            [-0.2347,  0.5849],
            [-0.2347,  0.5849],
            [-0.2347,  0.5849]], grad_fn=<CopyReduceBackward>)

    >>> # Case 2: Unidirectional bipartite graph
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> g = dgl.heterograph({('_N', '_E', '_N'):(u, v)})
    >>> u_fea = th.rand(2, 5)
    >>> v_fea = th.rand(4, 5)
    >>> conv = EdgeConv(5, 2, 3)
    >>> res = conv(g, (u_fea, v_fea))
    >>> res
    tensor([[ 1.6375,  0.2085],
            [-1.1925, -1.2852],
            [ 0.2101,  1.3466],
            [ 0.2342, -0.9868]], grad_fn=<CopyReduceBackward>)
    """

    def __init__(
        self, in_feat, out_feat, batch_norm=False, allow_zero_in_degree=False
    ):
        super(EdgeConv, self).__init__()
        self.batch_norm = batch_norm
        self._allow_zero_in_degree = allow_zero_in_degree

        self.theta = nn.Linear(in_feat, out_feat)
        self.phi = nn.Linear(in_feat, out_feat)

        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feat)

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

    def forward(self, g, feat):
        """

        Description
        -----------
        Forward computation

        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : Tensor or pair of tensors
            :math:`(N, D)` where :math:`N` is the number of nodes and
            :math:`D` is the number of feature dimensions.

            If a pair of tensors is given, the graph must be a uni-bipartite graph
            with only one edge type, and the two tensors must have the same
            dimensionality on all except the first axis.

        Returns
        -------
        torch.Tensor
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
                if (g.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            h_src, h_dst = expand_as_pair(feat, g)
            g.srcdata["x"] = h_src
            g.dstdata["x"] = h_dst
            g.apply_edges(fn.v_sub_u("x", "x", "theta"))
            g.edata["theta"] = self.theta(g.edata["theta"])
            g.dstdata["phi"] = self.phi(g.dstdata["x"])
            if not self.batch_norm:
                g.update_all(fn.e_add_v("theta", "phi", "e"), fn.max("e", "x"))
            else:
                g.apply_edges(fn.e_add_v("theta", "phi", "e"))
                # Although the official implementation includes a per-edge
                # batch norm within EdgeConv, I choose to replace it with a
                # global batch norm for a number of reasons:
                #
                # (1) When the point clouds within each batch do not have the
                #     same number of points, batch norm would not work.
                #
                # (2) Even if the point clouds always have the same number of
                #     points, the points may as well be shuffled even with the
                #     same (type of) object (and the official implementation
                #     *does* shuffle the points of the same example for each
                #     epoch).
                #
                #     For example, the first point of a point cloud of an
                #     airplane does not always necessarily reside at its nose.
                #
                #     In this case, the learned statistics of each position
                #     by batch norm is not as meaningful as those learned from
                #     images.
                g.edata["e"] = self.bn(g.edata["e"])
                g.update_all(fn.copy_e("e", "e"), fn.max("e", "x"))
            return g.dstdata["x"]
