"""Torch Module for EdgeConv Layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
from torch import nn

from .... import function as fn
from ....utils import expand_as_pair


class EdgeConv(nn.Module):
    r"""EdgeConv layer.

    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:

    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})

    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.

    Notes
    -----
    Zero in degree nodes could lead to invalid output. A common practice
    to avoid this is to add a self-loop for each node in the graph if it's homogeneous,
    which can be achieved by:

    >>> g = ... # some homogeneous graph
    >>> dgl.add_self_loop(g)

    For Unidirectional bipartite graph, we need to filter out the destination nodes with zero in-degree when use in downstream.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import EdgeConv

    Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> conv = EdgeConv(10, 2)
    >>> res = conv(g, feat)
    >>> res
    tensor([[-3.2300e-01,  9.0517e-01],
            [-3.2300e-01,  9.0517e-01],
            [-3.2300e-01,  9.0517e-01],
            [-3.2300e-01,  9.0517e-01],
            [-3.2300e-01,  9.0517e-01],
            [-3.4028e+38, -3.4028e+38]], grad_fn=<CopyReduceBackward>)

    Case 2: Unidirectional bipartite graph
    >>> u = [0, 0, 1]
    >>> v = [2, 3, 2]
    >>> g = dgl.bipartite((u, v))
    >>> u_fea = th.rand(2, 5)
    >>> v_fea = th.rand(4, 5)
    >>> conv = EdgeConv(5, 2, 3)
    >>> res = conv(g, (u_fea, v_fea))
    >>> res
    tensor([[-3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38],
            [ 8.5870e-01, -4.9619e-01],
            [-1.4017e+00,  1.3946e+00]], grad_fn=<CopyReduceBackward>)
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 batch_norm=False):
        super(EdgeConv, self).__init__()
        self.batch_norm = batch_norm

        self.theta = nn.Linear(in_feat, out_feat)
        self.phi = nn.Linear(in_feat, out_feat)

        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feat)

    def message(self, edges):
        """The message computation function.
        """
        theta_x = self.theta(edges.dst['x'] - edges.src['x'])
        phi_x = self.phi(edges.src['x'])
        return {'e': theta_x + phi_x}

    def forward(self, g, h):
        """Forward computation

        Parameters
        ----------
        g : DGLGraph
            The graph.
        h : Tensor or pair of tensors
            :math:`(N, D)` where :math:`N` is the number of nodes and
            :math:`D` is the number of feature dimensions.

            If a pair of tensors is given, the graph must be a uni-bipartite graph
            with only one edge type, and the two tensors must have the same
            dimensionality on all except the first axis.
        Returns
        -------
        torch.Tensor
            New node features.
        """
        with g.local_scope():
            h_src, h_dst = expand_as_pair(h, g)
            g.srcdata['x'] = h_src
            g.dstdata['x'] = h_dst
            if not self.batch_norm:
                g.update_all(self.message, fn.max('e', 'x'))
            else:
                g.apply_edges(self.message)
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
                g.edata['e'] = self.bn(g.edata['e'])
                g.update_all(fn.copy_e('e', 'e'), fn.max('e', 'x'))
            return g.dstdata['x']
