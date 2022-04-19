"""Torch Module for Graph Isomorphism Network layer variant with edge features"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
import torch.nn.functional as F
from torch import nn

from .... import function as fn
from ....utils import expand_as_pair

class GINEConv(nn.Module):
    r"""Graph Isomorphism Network layer variant from
    `Strategies for Pre-training Graph Neural Networks <https://arxiv.org/abs/1905.12265>`__

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \sum_{j\in\mathcal{N}(i)}\mathrm{ReLU}(h_j^{l} + e_{j,i}^{l})\right)

    where :math:`e_{j,i}^{l}` is the edge feature.

    Parameters
    ----------
    apply_func : callable module or None
        The :math:`f_\Theta` in the formula. If not None, it will be applied to
        the updated node features. The default value is None.
    init_eps : float, optional
        Initial :math:`\epsilon` value, default: ``0``.
    learn_eps : bool, optional
        If True, :math:`\epsilon` will be a learnable parameter. Default: ``False``.
    """
    def __init__(self,
                 apply_func=None,
                 init_eps=0,
                 learn_eps=False):
        super(GINEConv, self).__init__()
        self.apply_func = apply_func
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = nn.Parameter(th.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', th.FloatTensor([init_eps]))

    def forward(self, graph, feat, edge_feat):
        r"""Forward computation.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it is the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.
            If ``apply_func`` is not None, :math:`D_{in}` should
            fit the input feature size requirement of ``apply_func``.
        edge_feat : torch.Tensor
            Edge feature. It is a tensor of shape :math:`(E, D_{in})` where :math:`E`
            is the number of edges.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is the output feature size of ``apply_func``.
            If ``apply_func`` is None, :math:`D_{out}` should be the same
            as :math:`D_{in}`.
        """
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src
            graph.apply_edges(fn.copy_u('h', 'h'))
            graph.edata['h'] = F.relu(graph.edata['h'] + edge_feat)
            graph.update_all(fn.copy_e('h', 'm'), fn.sum('m', 'neigh'))
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            return rst
