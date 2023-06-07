"""Degree Encoder"""

import torch as th
import torch.nn as nn


class DegreeEncoder(nn.Module):
    r"""Degree Encoder, as introduced in
    `Do Transformers Really Perform Bad for Graph Representation?
    <https://proceedings.neurips.cc/paper/2021/file/f1c1592588411002af340cbaedd6fc33-Paper.pdf>`__

    This module is a learnable degree embedding module.

    Parameters
    ----------
    max_degree : int
        Upper bound of degrees to be encoded.
        Each degree will be clamped into the range [0, ``max_degree``].
    embedding_dim : int
        Output dimension of embedding vectors.
    direction : str, optional
        Degrees of which direction to be encoded,
        selected from ``in``, ``out`` and ``both``.
        ``both`` encodes degrees from both directions
        and output the addition of them.
        Default : ``both``.

    Example
    -------
    >>> import dgl
    >>> from dgl.nn import DegreeEncoder
    >>> import torch as th
    >>> from torch.nn.utils.rnn import pad_sequence

    >>> g1 = dgl.graph(([0,0,0,1,1,2,3,3], [1,2,3,0,3,0,0,1]))
    >>> g2 = dgl.graph(([0,1], [1,0]))
    >>> in_degree = pad_sequence([g1.in_degrees(), g2.in_degrees()], batch_first=True)
    >>> out_degree = pad_sequence([g1.out_degrees(), g2.out_degrees()], batch_first=True)
    >>> print(in_degree.shape)
    torch.Size([2, 4])
    >>> degree_encoder = DegreeEncoder(5, 16)
    >>> degree_embedding = degree_encoder(th.stack((in_degree, out_degree)))
    >>> print(degree_embedding.shape)
    torch.Size([2, 4, 16])
    """

    def __init__(self, max_degree, embedding_dim, direction="both"):
        super(DegreeEncoder, self).__init__()
        self.direction = direction
        if direction == "both":
            self.encoder1 = nn.Embedding(
                max_degree + 1, embedding_dim, padding_idx=0
            )
            self.encoder2 = nn.Embedding(
                max_degree + 1, embedding_dim, padding_idx=0
            )
        else:
            self.encoder = nn.Embedding(
                max_degree + 1, embedding_dim, padding_idx=0
            )
        self.max_degree = max_degree

    def forward(self, degrees):
        """
        Parameters
        ----------
        degrees : Tensor
            If :attr:`direction` is ``both``, it should be stacked in degrees and out degrees
            of the batched graph with zero padding, a tensor of shape :math:`(2, B, N)`.
            Otherwise, it should be zero-padded in degrees or out degrees of the batched
            graph, a tensor of shape :math:`(B, N)`, where :math:`B` is the batch size
            of the batched graph, and :math:`N` is the maximum number of nodes.

        Returns
        -------
        Tensor
            Return degree embedding vectors of shape :math:`(B, N, d)`,
            where :math:`d` is :attr:`embedding_dim`.
        """
        degrees = th.clamp(degrees, min=0, max=self.max_degree)

        if self.direction == "in":
            assert len(degrees.shape) == 2
            degree_embedding = self.encoder(degrees)
        elif self.direction == "out":
            assert len(degrees.shape) == 2
            degree_embedding = self.encoder(degrees)
        elif self.direction == "both":
            assert len(degrees.shape) == 3 and degrees.shape[0] == 2
            degree_embedding = self.encoder1(degrees[0]) + self.encoder2(
                degrees[1]
            )
        else:
            raise ValueError(
                f'Supported direction options: "in", "out" and "both", '
                f"but got {self.direction}"
            )
        return degree_embedding
