"""EGT Layer"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EGTLayer(nn.Module):
    r"""EGTLayer for Edge-augmented Graph Transformer (EGT), as introduced in
    `Global Self-Attention as a Replacement for Graph Convolution
    Reference `<https://arxiv.org/pdf/2108.03348.pdf>`_

    Parameters
    ----------
    ndim : int
        Node embedding dimension.
    edim : int
        Edge embedding dimension.
    num_heads : int
        Number of attention heads, by which :attr: `ndim` is divisible.
    num_vns : int
        Number of virtual nodes.
    dropout : float, optional
        Dropout probability. Default: 0.0.
    attn_dropout : float, optional
        Attention dropout probability. Default: 0.0.
    activation : callable activation layer, optional
        Activation function. Default: nn.ELU().
    ffn_multiplier : float, optional
        Multiplier of the inner dimension in Feed Forward Network.
        Default: 2.0.
    edge_update : bool, optional
        Whether to update the edge embedding. Default: True.

    Examples
    --------
    >>> import torch as th
    >>> from dgl.nn import EGTLayer

    >>> batch_size = 16
    >>> num_nodes = 100
    >>> ndim, edim = 128, 32
    >>> nfeat = th.rand(batch_size, num_nodes, ndim)
    >>> efeat = th.rand(batch_size, num_nodes, num_nodes, edim)
    >>> net = EGTLayer(
            ndim=ndim,
            edim=edim,
            num_heads=8,
            num_vns=4,
        )
    >>> out = net(nfeat, efeat)
    """

    def __init__(
        self,
        ndim,
        edim,
        num_heads,
        num_vns,
        dropout=0,
        attn_dropout=0,
        activation=nn.ELU(),
        ffn_multiplier=2.0,
        edge_update=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_vns = num_vns
        self.edge_update = edge_update

        assert not (ndim % num_heads)
        self.dot_dim = ndim // num_heads
        self.mha_ln_h = nn.LayerNorm(ndim)
        self.mha_ln_e = nn.LayerNorm(edim)
        self.E = nn.Linear(edim, num_heads)
        self.QKV = nn.Linear(ndim, ndim * 3)
        self.G = nn.Linear(edim, num_heads)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.O_h = nn.Linear(ndim, ndim)
        self.mha_dropout_h = nn.Dropout(dropout)

        node_inner_dim = round(ndim * ffn_multiplier)
        self.node_ffn = nn.Sequential(
            nn.LayerNorm(ndim),
            nn.Linear(ndim, node_inner_dim),
            activation,
            nn.Linear(node_inner_dim, ndim),
            nn.Dropout(dropout),
        )

        if self.edge_update:
            self.O_e = nn.Linear(num_heads, edim)
            self.mha_dropout_e = nn.Dropout(dropout)
            edge_inner_dim = round(edim * ffn_multiplier)
            self.edge_ffn = nn.Sequential(
                nn.LayerNorm(edim),
                nn.Linear(edim, edge_inner_dim),
                activation,
                nn.Linear(edge_inner_dim, edim),
                nn.Dropout(dropout),
            )

    def forward(self, h, e, mask=None):
        """Forward computation. Note: :attr:`h` and :attr:`e` should be padded
        with embedding of virtual nodes if :attr:`num_vns` > 0, while
        :attr:`mask` should be padded with `0` values for virtual nodes.

        Parameters
        ----------
        h : torch.Tensor
            A 3D input tensor. Shape: (batch_size, N, :attr:`ndim`), where N
            is the sum of maximum number of nodes and number of virtual nodes.
        e : torch.Tensor
            Edge embedding used for attention computation and self update.
            Shape: (batch_size, N, N, :attr:`edim`).
        mask : torch.Tensor, optional
            The attention mask used for avoiding computation on invalid
            positions, where valid positions are indicated by `0` and
            invalid positions are indicated by `-inf`.
            Shape: (batch_size, N, N). Default: None.

        Returns
        -------
        h : torch.Tensor
            The output node embedding. Shape: (batch_size, N, :attr:`ndim`).
        e : torch.Tensor
            The output edge embedding. Shape: (batch_size, N, N, :attr:`edim`).
        """

        h_r1 = h
        e_r1 = e

        h_ln = self.mha_ln_h(h)
        e_ln = self.mha_ln_e(e)
        QKV = self.QKV(h_ln)
        E = self.E(e_ln)
        G = self.G(e_ln)
        shp = QKV.shape
        Q, K, V = QKV.view(shp[0], shp[1], -1, self.num_heads).split(
            self.dot_dim, dim=2
        )
        A_hat = torch.einsum("bldh,bmdh->blmh", Q, K)
        H_hat = A_hat.clamp(-5, 5) + E

        if mask is None:
            gates = torch.sigmoid(G)
            A_tild = F.softmax(H_hat, dim=2) * gates
        else:
            gates = torch.sigmoid(G + mask.unsqueeze(-1))
            A_tild = F.softmax(H_hat + mask.unsqueeze(-1), dim=2) * gates

        A_tild = self.attn_dropout(A_tild)
        V_attn = torch.einsum("blmh,bmkh->blkh", A_tild, V)

        # Scale the aggregated values by degree.
        degrees = torch.sum(gates, dim=2, keepdim=True)
        degree_scalers = torch.log(1 + degrees)
        degree_scalers[:, : self.num_vns] = 1.0
        V_attn = V_attn * degree_scalers

        V_attn = V_attn.reshape(shp[0], shp[1], self.num_heads * self.dot_dim)
        h = self.O_h(V_attn)

        h = self.mha_dropout_h(h)
        h.add_(h_r1)
        h_r2 = h
        h = self.node_ffn(h)
        h.add_(h_r2)

        if self.edge_update:
            e = self.O_e(H_hat)
            e = self.mha_dropout_e(e)
            e.add_(e_r1)
            e_r2 = e
            e = self.edge_ffn(e)
            e.add_(e_r2)

        return h, e
