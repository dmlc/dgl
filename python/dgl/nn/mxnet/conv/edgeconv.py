"""MXNet Module for EdgeConv Layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import mxnet as mx
from mxnet.gluon import nn

from .... import function as fn
from ....utils import expand_as_pair


class EdgeConv(nn.Block):
    r"""EdgeConv layer.

    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:

    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})

    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 batch_norm=False):
        super(EdgeConv, self).__init__()
        self.batch_norm = batch_norm

        with self.name_scope():
            self.theta = nn.Dense(out_feat, in_units=in_feat,
                                  weight_initializer=mx.init.Xavier())
            self.phi = nn.Dense(out_feat, in_units=in_feat,
                                weight_initializer=mx.init.Xavier())

            if batch_norm:
                self.bn = nn.BatchNorm(in_channels=out_feat)

    def message(self, edges):
        r"""The message computation function
        """
        theta_x = self.theta(edges.dst['x'] - edges.src['x'])
        phi_x = self.phi(edges.src['x'])
        return {'e': theta_x + phi_x}

    def forward(self, g, h):
        r"""Forward computation

        Parameters
        ----------
        g : DGLGraph
            The graph.
        h : mxnet.NDArray
            :math:`(N, D)` where :math:`N` is the number of nodes and
            :math:`D` is the number of feature dimensions.

            If a pair of tensors is given, the graph must be a uni-bipartite graph
            with only one edge type, and the two tensors must have the same
            dimensionality on all except the first axis.
        Returns
        -------
        mxnet.NDArray
            New node features.
        """
        with g.local_scope():
            h_src, h_dst = expand_as_pair(h)
            g.srcdata['x'] = h_src
            g.dstdata['x'] = h_dst
            if not self.batch_norm:
                g.update_all(self.message, fn.max('e', 'x'))
            else:
                g.apply_edges(self.message)
                g.edata['e'] = self.bn(g.edata['e'])
                g.update_all(fn.copy_e('e', 'm'), fn.max('m', 'x'))
            return g.dstdata['x']
