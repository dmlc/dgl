"""Torch modules for graph convolutions."""
import torch as th
import torch.nn as nn
import torch.nn.init as init

from ... import function as fn

__all__ = ['GraphConv']

class GraphConv(nn.Module):
    """Apply graph convolution over an input signal.

    Graph convolution is introduced in `<https://arxiv.org/abs/1609.02907>`__
    and can be described as below:

    .. math::
      h_i^{(l+1)} = b^{(l)} + \sum_{j\in\mathcal{N}(i)}\\frac{1}{c_{ij}}W^{(l)}h_j^{(l)}

    where :math:`\mathcal{N}(i)` is the neighbor set of node :math:`i`. :math:`c_{ij}` is equal
    to the produce of the square root of node degrees:
    :math:`\sqrt{|\mathcal{N}(i)|}\sqrt{|\mathcal{N}(j)|}`.

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
        If True, the normalizer :math:`c_{ij}` is applied. Default: True.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: True.
    feat_name : str, optional
        The temporary feature name used to compute message passing. Default: ``"_gconv_feat"``.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor. Initialized by g
    bias : torch.Tensor
        The learnable bias tensor. Initialized by zeros.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm=True,
                 bias=True,
                 feat_name="_gconv_feat"):
        super(GraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._feat_name = feat_name
        self._msg_name = "_gconv_msg"

        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, feat, graph):
        """Compute graph convolution.

        Notes
        -----
            * Input shape: :math:`(N, *, \\text{in_feats})` where * means any number of additional
              dimensions, :math:`N` is the number of nodes.
            * Output shape: :math:`(N, *, \\text{in_feats})` where all but the last dimension are
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
        if self._norm:
            norm = 1 / th.sqrt(graph.in_degrees().float())
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp)
            feat = feat * norm

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            feat = th.matmul(feat, self.weight)
            graph.ndata[self._feat_name] = feat
            graph.update_all(fn.copy_src(src=self._feat_name, out=self._msg_name),
                             fn.sum(msg=self._msg_name, out=self._feat_name))
            rst = graph.ndata.pop(self._feat_name)
        else:
            # aggregate first then mult W
            graph.ndata[self._feat_name] = feat
            graph.update_all(fn.copy_src(src=self._feat_name, out=self._msg_name),
                             fn.sum(msg=self._msg_name, out=self._feat_name))
            rst = graph.ndata.pop(self._feat_name)
            rst = th.matmul(rst, self.weight)

        if self._norm:
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        return rst
