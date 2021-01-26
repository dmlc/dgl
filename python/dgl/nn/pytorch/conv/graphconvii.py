"""Torch modules for GCNII"""
# pylint: disable= no-member, arguments-differ, invalid-name
import math
import torch
import torch.nn as nn

from .... import function as fn
from ....base import DGLError

class GraphConvII(nn.Module):
    r"""

    Description
    -----------
    Graph Convolution Networks with Initial residual connection
    and Identity mapping (GCNII) was introduced in paper `Simple and Deep
    Graph Convolutional Networks <https://arxiv.org/abs/2007.02133>`__.
    The mathematical definition is as follows:

    .. math::
        H^{l+1} &= \sigma\left( \left((1-{\alpha})\tilde{P}H^{l}+{\alpha}H^{0}\right)
                   \left((1-{\beta})I_n+{\beta}W^{l}\right) \right)
        \tilde{P} &= \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}
        \beta &= \log(\lambda/l+1)

    where :math:`\alpha` and :math:`\lambda` are hyperparameters denoting the
    ratio of initial residual connection and ratio of identity mapping with decay,
    respectively.  :math:`\tilde{P}` is the Laplacian matrix of the graph with
    self-loops.

    The graph must be a homogeneous graph.

    The output feature size will always be the same as the input feature size due to
    the residual connection.

    Notes
    -----
    Zero in-degree nodes will lead to invalid output value. This is because no message
    will be passed to those nodes, the aggregation function will be appied on empty input.
    A common practice to avoid this is to add a self-loop for each node in the graph if
    it is homogeneous, which can be achieved by:

    >>> g = ... # a DGLGraph
    >>> g = dgl.add_self_loop(g)

    Parameters
    ----------
    in_size : int
        Input feature size. The output size will be the same as input size.
    alpha : float
        Ratio of initial residual connection :math:`\alpha`.
    lamda : float
        Ratio of identity mapping with decay :math:`\lambda`.  Must be
    norm : bool, optional
        Whether to apply the normalizer.
        Default: ``True`` as in the paper.
    weight : bool, optional
        Whether to use the weight matrix :math:`W`.
        If False then only identity mapping.
        Default: ``True`` as in the paper.
    bias : bool, optional
        If True, adds a learnable bias to the output before activation.
        Default: ``False`` as in the paper.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the output.
        Note that the paper uses ``ReLU`` as the activation function.
        Default: None.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor.
    bias : torch.Tensor
        The learnable bias tensor.
    """
    def __init__(self,
                 in_size,
                 alpha,
                 lamda,
                 norm=True,
                 weight=True,
                 bias=False,
                 activation=None):
        super(GraphConvII, self).__init__()
        self._in_size = in_size
        self._alpha = alpha
        self._lamda = lamda
        self._norm = norm
        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_size, in_size))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(in_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            stdv = 1. / math.sqrt(self._in_size)
            self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, graph, feat, initial_feat, l):
        r"""

        Descrption
        ----------
        Compute GCNII layer.

        Notes
        -----
        * Input shape: :math:`(N, *, \text{in_size})` where * means any number
          of additional dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{in_size})`, which is the same as the input.
        * Weight shape: :math:`(\text{in_size}, \text{out_size})`.

        Parameters
        ----------
        graph : DGLGraph
            The graph.  Must be either a homogeneous graph or a block
            extracted from that graph with :func:`~dgl.transform.to_block`.
        feat : torch.Tensor
            The input feature of shape :math:`(N, *)` :math:`N` is the
            number of nodes, and :math:`*` could be of any shape.
        initial_feat : torch.Tensor
            The initial feature of shape :math:`(N, *)` :math:`N` is the
            number of nodes, and :math:`*` could be of any shape.
        l: int
            The ordinal number of this layer. Start from 1.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        if l < 1:
            raise DGLError('Invalid value of `l`. Must be no less than 1.'
                           ' But got "{}".'.format(l))
        with graph.local_scope():
            device = feat.device

            if self._norm:
                degs = graph.out_degrees().to(device).float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat = feat * norm

            graph.srcdata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']

            if self._norm:
                degs = graph.in_degrees().to(device).float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            rst = (1-self._alpha)*rst + self._alpha*initial_feat

            beta_l = math.log(self._lamda / l + 1)
            if self.weight is not None:
                rst = beta_l * torch.matmul(rst, self.weight) + (1-beta_l) * rst

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst
