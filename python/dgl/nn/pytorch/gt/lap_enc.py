"""Laplacian Positional Encoder"""

import torch as th
import torch.nn as nn


class LapPosEncoder(nn.Module):
    r"""Laplacian Positional Encoder (LPE), as introduced in
    `GraphGPS: General Powerful Scalable Graph Transformers
    <https://arxiv.org/abs/2205.12454>`__

    This module is a learned laplacian positional encoding module using Transformer or DeepSet.

    Parameters
    ----------
    model_type : str
        Encoder model type for LPE, can only be "Transformer" or "DeepSet".
    num_layer : int
        Number of layers in Transformer/DeepSet Encoder.
    k : int
        Number of smallest non-trivial eigenvectors.
    lpe_dim : int
        Output size of final laplacian encoding.
    n_head : int, optional
        Number of heads in Transformer Encoder.
        Default : 1.
    batch_norm : bool, optional
        If True, apply batch normalization on raw LaplacianPE.
        Default : False.
    num_post_layer : int, optional
        If num_post_layer > 0, apply an MLP of ``num_post_layer`` layers after pooling.
        Default : 0.

    Example
    -------
    >>> import dgl
    >>> from dgl import LaplacianPE
    >>> from dgl.nn import LapPosEncoder

    >>> transform = LaplacianPE(k=5, feat_name='eigvec', eigval_name='eigval', padding=True)
    >>> g = dgl.graph(([0,1,2,3,4,2,3,1,4,0], [2,3,1,4,0,0,1,2,3,4]))
    >>> g = transform(g)
    >>> EigVals, EigVecs = g.ndata['eigval'], g.ndata['eigvec']
    >>> TransformerLPE = LapPosEncoder(model_type="Transformer", num_layer=3, k=5,
                                       lpe_dim=16, n_head=4)
    >>> PosEnc = TransformerLPE(EigVals, EigVecs)
    >>> DeepSetLPE = LapPosEncoder(model_type="DeepSet", num_layer=3, k=5,
                                   lpe_dim=16, num_post_layer=2)
    >>> PosEnc = DeepSetLPE(EigVals, EigVecs)
    """

    def __init__(
        self,
        model_type,
        num_layer,
        k,
        lpe_dim,
        n_head=1,
        batch_norm=False,
        num_post_layer=0,
    ):
        super(LapPosEncoder, self).__init__()
        self.model_type = model_type
        self.linear = nn.Linear(2, lpe_dim)

        if self.model_type == "Transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=lpe_dim, nhead=n_head, batch_first=True
            )
            self.pe_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layer
            )
        elif self.model_type == "DeepSet":
            layers = []
            if num_layer == 1:
                layers.append(nn.ReLU())
            else:
                self.linear = nn.Linear(2, 2 * lpe_dim)
                layers.append(nn.ReLU())
                for _ in range(num_layer - 2):
                    layers.append(nn.Linear(2 * lpe_dim, 2 * lpe_dim))
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(2 * lpe_dim, lpe_dim))
                layers.append(nn.ReLU())
            self.pe_encoder = nn.Sequential(*layers)
        else:
            raise ValueError(
                f"model_type '{model_type}' is not allowed, must be 'Transformer'"
                "or 'DeepSet'."
            )

        if batch_norm:
            self.raw_norm = nn.BatchNorm1d(k)
        else:
            self.raw_norm = None

        if num_post_layer > 0:
            layers = []
            if num_post_layer == 1:
                layers.append(nn.Linear(lpe_dim, lpe_dim))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(lpe_dim, 2 * lpe_dim))
                layers.append(nn.ReLU())
                for _ in range(num_post_layer - 2):
                    layers.append(nn.Linear(2 * lpe_dim, 2 * lpe_dim))
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(2 * lpe_dim, lpe_dim))
                layers.append(nn.ReLU())
            self.post_mlp = nn.Sequential(*layers)
        else:
            self.post_mlp = None

    def forward(self, EigVals, EigVecs):
        r"""
        Parameters
        ----------
        EigVals : Tensor
            Laplacian Eigenvalues of shape :math:`(N, k)`, k different eigenvalues repeat N times,
            can be obtained by using `LaplacianPE`.
        EigVecs : Tensor
            Laplacian Eigenvectors of shape :math:`(N, k)`, can be obtained by using `LaplacianPE`.

        Returns
        -------
        Tensor
            Return the laplacian positional encodings of shape :math:`(N, d)`,
            where :math:`N` is the number of nodes in the input graph,
            :math:`d` is :attr:`lpe_dim`.
        """
        PosEnc = th.cat(
            (EigVecs.unsqueeze(2), EigVals.unsqueeze(2)), dim=2
        ).float()
        empty_mask = th.isnan(PosEnc)

        PosEnc[empty_mask] = 0
        if self.raw_norm:
            PosEnc = self.raw_norm(PosEnc)
        PosEnc = self.linear(PosEnc)

        if self.model_type == "Transformer":
            PosEnc = self.pe_encoder(
                src=PosEnc, src_key_padding_mask=empty_mask[:, :, 1]
            )
        else:
            PosEnc = self.pe_encoder(PosEnc)

        # Remove masked sequences
        PosEnc[empty_mask[:, :, 1]] = 0

        # Sum pooling
        PosEnc = th.sum(PosEnc, 1, keepdim=False)

        # MLP post pooling
        if self.post_mlp:
            PosEnc = self.post_mlp(PosEnc)

        return PosEnc
