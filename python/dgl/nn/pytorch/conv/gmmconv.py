"""Torch Module for GMM Conv"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from torch.nn import init

from .... import function as fn
from ..utils import Identity
from ....utils import expand_as_pair


class GMMConv(nn.Module):
    r"""The Gaussian Mixture Model Convolution layer from `Geometric Deep
    Learning on Graphs and Manifolds using Mixture Model CNNs
    <http://openaccess.thecvf.com/content_cvpr_2017/papers/Monti_Geometric_Deep_Learning_CVPR_2017_paper.pdf>`__.

    .. math::
        h_i^{l+1} & = \mathrm{aggregate}\left(\left\{\frac{1}{K}
         \sum_{k}^{K} w_k(u_{ij}), \forall j\in \mathcal{N}(i)\right\}\right)

        w_k(u) & = \exp\left(-\frac{1}{2}(u-\mu_k)^T \Sigma_k^{-1} (u - \mu_k)\right)

    Parameters
    ----------
    in_feats : int
        Number of input features.
    out_feats : int
        Number of output features.
    dim : int
        Dimensionality of pseudo-coordinte.
    n_kernels : int
        Number of kernels :math:`K`.
    aggregator_type : str
        Aggregator type (``sum``, ``mean``, ``max``).
    residual : bool
        If True, use residual connection inside this layer. Default: ``False``.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 dim,
                 n_kernels,
                 aggregator_type='sum',
                 residual=False,
                 bias=True):
        super(GMMConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._dim = dim
        self._n_kernels = n_kernels
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        elif aggregator_type == 'max':
            self._reducer = fn.max
        else:
            raise KeyError("Aggregator type {} not recognized.".format(aggregator_type))

        self.mu = nn.Parameter(th.Tensor(n_kernels, dim))
        self.inv_sigma = nn.Parameter(th.Tensor(n_kernels, dim))
        self.fc = nn.Linear(self._in_src_feats, n_kernels * out_feats, bias=False)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = init.calculate_gain('relu')
        init.xavier_normal_(self.fc.weight, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            init.xavier_normal_(self.res_fc.weight, gain=gain)
        init.normal_(self.mu.data, 0, 0.1)
        init.constant_(self.inv_sigma.data, 1)
        if self.bias is not None:
            init.zeros_(self.bias.data)

    def forward(self, graph, feat, pseudo):
        """Compute Gaussian Mixture Model Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            If a single tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of tensors are given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        pseudo : torch.Tensor
            The pseudo coordinate tensor of shape :math:`(E, D_{u})` where
            :math:`E` is the number of edges of the graph and :math:`D_{u}`
            is the dimensionality of pseudo coordinate.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is the output feature size.
        """
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat)
            graph.srcdata['h'] = self.fc(feat_src).view(-1, self._n_kernels, self._out_feats)
            E = graph.number_of_edges()
            # compute gaussian weight
            gaussian = -0.5 * ((pseudo.view(E, 1, self._dim) -
                                self.mu.view(1, self._n_kernels, self._dim)) ** 2)
            gaussian = gaussian * (self.inv_sigma.view(1, self._n_kernels, self._dim) ** 2)
            gaussian = th.exp(gaussian.sum(dim=-1, keepdim=True)) # (E, K, 1)
            graph.edata['w'] = gaussian
            graph.update_all(fn.u_mul_e('h', 'w', 'm'), self._reducer('m', 'h'))
            rst = graph.dstdata['h'].sum(1)
            # residual connection
            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)
            # bias
            if self.bias is not None:
                rst = rst + self.bias
            return rst
