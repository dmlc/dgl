"""Torch modules for graph convolutions."""
# pylint: disable= no-member, arguments-differ
import torch as th
from torch import nn
from torch.nn import init

from ... import function as fn

__all__ = ['GraphConv', 'TGConv']

class GraphConv(nn.Module):
    r"""Apply graph convolution over an input signal.

    Graph convolution is introduced in `GCN <https://arxiv.org/abs/1609.02907>`__
    and can be described as below:

    .. math::
      h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ij}}h_j^{(l)}W^{(l)})

    where :math:`\mathcal{N}(i)` is the neighbor set of node :math:`i`. :math:`c_{ij}` is equal
    to the product of the square root of node degrees:
    :math:`\sqrt{|\mathcal{N}(i)|}\sqrt{|\mathcal{N}(j)|}`. :math:`\sigma` is an activation
    function.

    The model parameters are initialized as in the
    `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__ where
    the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
    and the bias is initialized to be zero.

    Notes
    -----
    Zero in degree nodes could lead to invalid normalizer. A common practice
    to avoid this is to add a self-loop for each node in the graph, which
    can be achieved by:

    >>> g = ... # some DGLGraph
    >>> g.add_edges(g.nodes(), g.nodes())


    Parameters
    ----------
    in_feats : int
        Number of input features.
    out_feats : int
        Number of output features.
    norm : bool, optional
        If True, the normalizer :math:`c_{ij}` is applied. Default: ``True``.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor.
    bias : torch.Tensor
        The learnable bias tensor.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm=True,
                 bias=True,
                 activation=None):
        super(GraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm

        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, feat, graph):
        r"""Compute graph convolution.

        Notes
        -----
            * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
              dimensions, :math:`N` is the number of nodes.
            * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
              the same shape as the input.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature
        graph : DGLGraph
            The graph.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        graph = graph.local_var()
        if self._norm:
            norm = th.pow(graph.in_degrees().float(), -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp).to(feat.device)
            feat = feat * norm

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            feat = th.matmul(feat, self.weight)
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.ndata['h']
        else:
            # aggregate first then mult W
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.ndata['h']
            rst = th.matmul(rst, self.weight)

        if self._norm:
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)

class TGConv(nn.Module):
    r"""Apply Topology Adaptive Graph Convolutional Network

    .. math::
        \mathbf{X}^{\prime} = \sum_{k=0}^K \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2}\mathbf{X} \mathbf{\Theta}_{k},

    where :math:`\mathbf{A}` denotes the adjacency matrix and
    :math:`D_{ii} = \sum_{j=0} A_{ij}` its diagonal degree matrix.
    
    Parameters
    ----------
    in_feats : int
        Number of input features.
    out_feats : int
        Number of output features.
    K: int, optional
        Number of hops :math: `K`. (default: 3)
    bias: bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    ----------
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 K=2,
                 bias=True,
                 activation=None):
        super(TGConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._K = K

        self.lin = nn.Linear(in_feats * (self._K + 1), out_feats, bias=bias)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, feat, graph):
        r"""Compute graph convolution

        Parameters
        ----------
        feat : torch.Tensor
            The input feature
        graph : DGLGraph
            The graph.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        graph = graph.local_var()

        norm = th.pow(graph.in_degrees().float(), -0.5)
        shp = norm.shape + (1,) * (feat.dim() - 1)
        norm = th.reshape(norm, shp).to(feat.device)

        #D-1/2 A D -1/2 X
        xs = [feat]
        for k in range(self._K):

            rst = xs[-1] * norm
            graph.ndata['h'] = rst

            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.ndata['h']
            rst = rst * norm
            xs.append(rst)

        rst = self.lin(th.cat(xs, dim=-1))

        if self._activation is not None:
            rst = self._activation(rst)
        
        return rst
