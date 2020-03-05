"""Torch Module for Chebyshev Spectral Graph Convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from torch.nn import init

from .... import laplacian_lambda_max, broadcast_nodes, function as fn


class ChebConv(nn.Module):
    r"""Chebyshev Spectral Graph Convolution layer from paper `Convolutional
    Neural Networks on Graphs with Fast Localized Spectral Filtering
    <https://arxiv.org/pdf/1606.09375.pdf>`__.

    .. math::
        h_i^{l+1} &= \sum_{k=0}^{K-1} W^{k, l}z_i^{k, l}

        Z^{0, l} &= H^{l}

        Z^{1, l} &= \hat{L} \cdot H^{l}

        Z^{k, l} &= 2 \cdot \hat{L} \cdot Z^{k-1, l} - Z^{k-2, l}

        \hat{L} &= 2\left(I - \hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2}\right)/\lambda_{max} - I

    Parameters
    ----------
    in_feats: int
        Number of input features.
    out_feats: int
        Number of output features.
    k : int
        Chebyshev filter size.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 k,
                 bias=True):
        super(ChebConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.fc = nn.ModuleList([
            nn.Linear(in_feats, out_feats, bias=False) for _ in range(k)
        ])
        self._k = k
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.bias is not None:
            init.zeros_(self.bias)
        for module in self.fc.modules():
            if isinstance(module, nn.Linear):
                init.xavier_normal_(module.weight, init.calculate_gain('relu'))
                if module.bias is not None:
                    init.zeros_(module.bias)

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
            If None, this method would compute the list by calling
            ``dgl.laplacian_lambda_max``.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        with graph.local_scope():
            norm = th.pow(
                graph.in_degrees().float().clamp(min=1), -0.5).unsqueeze(-1).to(feat.device)
            if lambda_max is None:
                lambda_max = laplacian_lambda_max(graph)
            if isinstance(lambda_max, list):
                lambda_max = th.Tensor(lambda_max).to(feat.device)
            if lambda_max.dim() == 1:
                lambda_max = lambda_max.unsqueeze(-1)  # (B,) to (B, 1)
            # broadcast from (B, 1) to (N, 1)
            lambda_max = broadcast_nodes(graph, lambda_max)
            # T0(X)
            Tx_0 = feat
            rst = self.fc[0](Tx_0)
            # T1(X)
            if self._k > 1:
                graph.ndata['h'] = Tx_0 * norm
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                h = graph.ndata.pop('h') * norm
                # Λ = 2 * (I - D ^ -1/2 A D ^ -1/2) / lambda_max - I
                #   = - 2(D ^ -1/2 A D ^ -1/2) / lambda_max + (2 / lambda_max - 1) I
                Tx_1 = -2. * h / lambda_max + Tx_0 * (2. / lambda_max - 1)
                rst = rst + self.fc[1](Tx_1)
            # Ti(x), i = 2...k
            for i in range(2, self._k):
                graph.ndata['h'] = Tx_1 * norm
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                h = graph.ndata.pop('h') * norm
                # Tx_k = 2 * Λ * Tx_(k-1) - Tx_(k-2)
                #      = - 4(D ^ -1/2 A D ^ -1/2) / lambda_max Tx_(k-1) +
                #        (4 / lambda_max - 2) Tx_(k-1) -
                #        Tx_(k-2)
                Tx_2 = -4. * h / lambda_max + Tx_1 * (4. / lambda_max - 2) - Tx_0
                rst = rst + self.fc[i](Tx_2)
                Tx_1, Tx_0 = Tx_2, Tx_1
            # add bias
            if self.bias is not None:
                rst = rst + self.bias
            return rst
