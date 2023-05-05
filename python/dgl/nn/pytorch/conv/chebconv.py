"""Torch Module for Chebyshev Spectral Graph Convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
import torch.nn.functional as F
from torch import nn

from .... import broadcast_nodes, function as fn
from ....base import dgl_warning


class ChebConv(nn.Module):
    r"""Chebyshev Spectral Graph Convolution layer from `Convolutional
    Neural Networks on Graphs with Fast Localized Spectral Filtering
    <https://arxiv.org/pdf/1606.09375.pdf>`__

    .. math::
        h_i^{l+1} &= \sum_{k=0}^{K-1} W^{k, l}z_i^{k, l}

        Z^{0, l} &= H^{l}

        Z^{1, l} &= \tilde{L} \cdot H^{l}

        Z^{k, l} &= 2 \cdot \tilde{L} \cdot Z^{k-1, l} - Z^{k-2, l}

        \tilde{L} &= 2\left(I - \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}\right)/\lambda_{max} - I

    where :math:`\tilde{A}` is :math:`A` + :math:`I`, :math:`W` is learnable weight.


    Parameters
    ----------
    in_feats: int
        Dimension of input features; i.e, the number of dimensions of :math:`h_i^{(l)}`.
    out_feats: int
        Dimension of output features :math:`h_i^{(l+1)}`.
    k : int
        Chebyshev filter size :math:`K`.
    activation : function, optional
        Activation function. Default ``ReLu``.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import ChebConv
    >>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> conv = ChebConv(10, 2, 2)
    >>> res = conv(g, feat)
    >>> res
    tensor([[ 0.6163, -0.1809],
            [ 0.6163, -0.1809],
            [ 0.6163, -0.1809],
            [ 0.9698, -1.5053],
            [ 0.3664,  0.7556],
            [-0.2370,  3.0164]], grad_fn=<AddBackward0>)
    """

    def __init__(self, in_feats, out_feats, k, activation=F.relu, bias=True):
        super(ChebConv, self).__init__()
        self._k = k
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(k * in_feats, out_feats, bias)

    def forward(self, graph, feat, lambda_max=None):
        r"""Compute ChebNet layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.
        lambda_max : list or tensor or None, optional.
            A list(tensor) with length :math:`B`, stores the largest eigenvalue
            of the normalized laplacian of each individual graph in ``graph``,
            where :math:`B` is the batch size of the input graph. Default: None.

            If None, this method would set the default value to 2.
            One can use :func:`dgl.laplacian_lambda_max` to compute this value.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """

        def unnLaplacian(feat, D_invsqrt, graph):
            """Operation Feat * D^-1/2 A D^-1/2"""
            graph.ndata["h"] = feat * D_invsqrt
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            return graph.ndata.pop("h") * D_invsqrt

        with graph.local_scope():
            D_invsqrt = th.pow(
                graph.in_degrees().to(feat).clamp(min=1), -0.5
            ).unsqueeze(-1)

            if lambda_max is None:
                dgl_warning(
                    "lambda_max is not provided, using default value of 2.  "
                    "Please use dgl.laplacian_lambda_max to compute the eigenvalues."
                )
                lambda_max = [2] * graph.batch_size

            if isinstance(lambda_max, list):
                lambda_max = th.Tensor(lambda_max).to(feat)
            if lambda_max.dim() == 1:
                lambda_max = lambda_max.unsqueeze(-1)  # (B,) to (B, 1)

            # broadcast from (B, 1) to (N, 1)
            lambda_max = broadcast_nodes(graph, lambda_max)
            re_norm = 2.0 / lambda_max

            # X_0 is the raw feature, Xt is the list of X_0, X_1, ... X_t
            X_0 = feat
            Xt = [X_0]

            # X_1(f)
            if self._k > 1:
                h = unnLaplacian(X_0, D_invsqrt, graph)
                X_1 = -re_norm * h + X_0 * (re_norm - 1)
                # Append X_1 to Xt
                Xt.append(X_1)

            # Xi(x), i = 2...k
            for _ in range(2, self._k):
                h = unnLaplacian(X_1, D_invsqrt, graph)
                X_i = -2 * re_norm * h + X_1 * 2 * (re_norm - 1) - X_0
                # Add X_1 to Xt
                Xt.append(X_i)
                X_1, X_0 = X_i, X_1

            # Create the concatenation
            Xt = th.cat(Xt, dim=1)

            # linear projection
            h = self.linear(Xt)

            # activation
            if self.activation:
                h = self.activation(h)

        return h
