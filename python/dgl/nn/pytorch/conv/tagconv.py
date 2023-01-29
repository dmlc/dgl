"""Torch Module for Topology Adaptive Graph Convolutional layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn

from .... import function as fn
from .graphconv import EdgeWeightNorm


class TAGConv(nn.Module):
    r"""Topology Adaptive Graph Convolutional layer from `Topology
    Adaptive Graph Convolutional Networks <https://arxiv.org/pdf/1710.10370.pdf>`__

    .. math::
        H^{K} = {\sum}_{k=0}^K (D^{-1/2} A D^{-1/2})^{k} X {\Theta}_{k},

    where :math:`A` denotes the adjacency matrix,
    :math:`D_{ii} = \sum_{j=0} A_{ij}` its diagonal degree matrix,
    :math:`{\Theta}_{k}` denotes the linear weights to sum the results of different hops together.

    Parameters
    ----------
    in_feats : int
        Input feature size. i.e, the number of dimensions of :math:`X`.
    out_feats : int
        Output feature size.  i.e, the number of dimensions of :math:`H^{K}`.
    k: int, optional
        Number of hops :math:`K`. Default: ``2``.
    bias: bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    ----------
    lin : torch.Module
        The learnable linear module.

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import TAGConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> conv = TAGConv(10, 2, k=2)
    >>> res = conv(g, feat)
    >>> res
    tensor([[ 0.5490, -1.6373],
            [ 0.5490, -1.6373],
            [ 0.5490, -1.6373],
            [ 0.5513, -1.8208],
            [ 0.5215, -1.6044],
            [ 0.3304, -1.9927]], grad_fn=<AddmmBackward>)
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        k=2,
        bias=True,
        activation=None,
    ):
        super(TAGConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._k = k
        self._activation = activation
        self.lin = nn.Linear(in_feats * (self._k + 1), out_feats, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized using Glorot uniform initialization.
        """
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.lin.weight, gain=gain)

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute topology adaptive graph convolution.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.
        edge_weight: torch.Tensor, optional
            edge_weight to use in the message passing process. This is equivalent to
            using weighted adjacency matrix in the equation above, and
            :math:`\tilde{D}^{-1/2}\tilde{A} \tilde{D}^{-1/2}`
            is based on :class:`dgl.nn.pytorch.conv.graphconv.EdgeWeightNorm`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        with graph.local_scope():
            assert graph.is_homogeneous, "Graph is not homogeneous"
            if edge_weight is None:
                norm = th.pow(graph.in_degrees().to(feat).clamp(min=1), -0.5)
                shp = norm.shape + (1,) * (feat.dim() - 1)
                norm = th.reshape(norm, shp).to(feat.device)

            msg_func = fn.copy_u("h", "m")
            if edge_weight is not None:
                graph.edata["_edge_weight"] = EdgeWeightNorm("both")(
                    graph, edge_weight
                )
                msg_func = fn.u_mul_e("h", "_edge_weight", "m")
            # D-1/2 A D -1/2 X
            fstack = [feat]
            for _ in range(self._k):
                if edge_weight is None:
                    rst = fstack[-1] * norm
                else:
                    rst = fstack[-1]
                graph.ndata["h"] = rst

                graph.update_all(msg_func, fn.sum(msg="m", out="h"))
                rst = graph.ndata["h"]
                if edge_weight is None:
                    rst = rst * norm
                fstack.append(rst)

            rst = self.lin(th.cat(fstack, dim=-1))

            if self._activation is not None:
                rst = self._activation(rst)

            return rst
