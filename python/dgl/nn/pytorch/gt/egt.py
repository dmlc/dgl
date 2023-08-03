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
    feat_size : int
        Node feature size.
    edge_feat_size : int
        Edge feature size.
    num_heads : int
        Number of attention heads, by which :attr: `feat_size` is divisible.
    num_virtual_nodes : int
        Number of virtual nodes.
    dropout : float, optional
        Dropout probability. Default: 0.0.
    attn_dropout : float, optional
        Attention dropout probability. Default: 0.0.
    activation : callable activation layer, optional
        Activation function. Default: nn.ELU().
    edge_update : bool, optional
        Whether to update the edge embedding. Default: True.

    Examples
    --------
    >>> import torch as th
    >>> from dgl.nn import EGTLayer

    >>> batch_size = 16
    >>> num_nodes = 100
    >>> feat_size, edge_feat_size = 128, 32
    >>> nfeat = th.rand(batch_size, num_nodes, feat_size)
    >>> efeat = th.rand(batch_size, num_nodes, num_nodes, edge_feat_size)
    >>> net = EGTLayer(
            feat_size=feat_size,
            edge_feat_size=edge_feat_size,
            num_heads=8,
            num_virtual_nodes=4,
        )
    >>> out = net(nfeat, efeat)
    """

    def __init__(
        self,
        feat_size,
        edge_feat_size,
        num_heads,
        num_virtual_nodes,
        dropout=0,
        attn_dropout=0,
        activation=nn.ELU(),
        edge_update=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_virtual_nodes = num_virtual_nodes
        self.edge_update = edge_update

        assert (
            feat_size % num_heads == 0
        ), "feat_size must be divisible by num_heads"
        self.dot_dim = feat_size // num_heads
        self.mha_ln_h = nn.LayerNorm(feat_size)
        self.mha_ln_e = nn.LayerNorm(edge_feat_size)
        self.edge_input = nn.Linear(edge_feat_size, num_heads)
        self.qkv_proj = nn.Linear(feat_size, feat_size * 3)
        self.gate = nn.Linear(edge_feat_size, num_heads)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.node_output = nn.Linear(feat_size, feat_size)
        self.mha_dropout_h = nn.Dropout(dropout)

        self.node_ffn = nn.Sequential(
            nn.LayerNorm(feat_size),
            nn.Linear(feat_size, feat_size),
            activation,
            nn.Linear(feat_size, feat_size),
            nn.Dropout(dropout),
        )

        if self.edge_update:
            self.edge_output = nn.Linear(num_heads, edge_feat_size)
            self.mha_dropout_e = nn.Dropout(dropout)
            self.edge_ffn = nn.Sequential(
                nn.LayerNorm(edge_feat_size),
                nn.Linear(edge_feat_size, edge_feat_size),
                activation,
                nn.Linear(edge_feat_size, edge_feat_size),
                nn.Dropout(dropout),
            )

    def forward(self, nfeat, efeat, mask=None):
        """Forward computation. Note: :attr:`nfeat` and :attr:`efeat` should be
        padded with embedding of virtual nodes if :attr:`num_virtual_nodes` > 0,
        while :attr:`mask` should be padded with `0` values for virtual nodes.
        The padding should be put at the beginning.

        Parameters
        ----------
        nfeat : torch.Tensor
            A 3D input tensor. Shape: (batch_size, N, :attr:`feat_size`), where N
            is the sum of the maximum number of nodes and the number of virtual nodes.
        efeat : torch.Tensor
            Edge embedding used for attention computation and self update.
            Shape: (batch_size, N, N, :attr:`edge_feat_size`).
        mask : torch.Tensor, optional
            The attention mask used for avoiding computation on invalid
            positions, where valid positions are indicated by `0` and
            invalid positions are indicated by `-inf`.
            Shape: (batch_size, N, N). Default: None.

        Returns
        -------
        nfeat : torch.Tensor
            The output node embedding. Shape: (batch_size, N, :attr:`feat_size`).
        efeat : torch.Tensor, optional
            The output edge embedding. Shape: (batch_size, N, N, :attr:`edge_feat_size`).
            It is returned only if :attr:`edge_update` is True.
        """
        nfeat_r1 = nfeat
        efeat_r1 = efeat

        nfeat_ln = self.mha_ln_h(nfeat)
        efeat_ln = self.mha_ln_e(efeat)
        qkv = self.qkv_proj(nfeat_ln)
        e_bias = self.edge_input(efeat_ln)
        gates = self.gate(efeat_ln)
        bsz, N, _ = qkv.shape
        q_h, k_h, v_h = qkv.view(bsz, N, -1, self.num_heads).split(
            self.dot_dim, dim=2
        )
        attn_hat = torch.einsum("bldh,bmdh->blmh", q_h, k_h)
        attn_hat = attn_hat.clamp(-5, 5) + e_bias

        if mask is None:
            gates = torch.sigmoid(gates)
            attn_tild = F.softmax(attn_hat, dim=2) * gates
        else:
            gates = torch.sigmoid(gates + mask.unsqueeze(-1))
            attn_tild = F.softmax(attn_hat + mask.unsqueeze(-1), dim=2) * gates

        attn_tild = self.attn_dropout(attn_tild)
        v_attn = torch.einsum("blmh,bmkh->blkh", attn_tild, v_h)

        # Scale the aggregated values by degree.
        degrees = torch.sum(gates, dim=2, keepdim=True)
        degree_scalers = torch.log(1 + degrees)
        degree_scalers[:, : self.num_virtual_nodes] = 1.0
        v_attn = v_attn * degree_scalers

        v_attn = v_attn.reshape(bsz, N, self.num_heads * self.dot_dim)
        nfeat = self.node_output(v_attn)

        nfeat = self.mha_dropout_h(nfeat)
        nfeat.add_(nfeat_r1)
        nfeat_r2 = nfeat
        nfeat = self.node_ffn(nfeat)
        nfeat.add_(nfeat_r2)

        if self.edge_update:
            efeat = self.edge_output(attn_hat)
            efeat = self.mha_dropout_e(efeat)
            efeat.add_(efeat_r1)
            efeat_r2 = efeat
            efeat = self.edge_ffn(efeat)
            efeat.add_(efeat_r2)

            return nfeat, efeat

        return nfeat
