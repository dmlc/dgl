"""Degree Encoder"""

import torch as th
import torch.nn as nn

from ....base import DGLError


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

    >>> g = dgl.graph(([0,0,0,1,1,2,3,3], [1,2,3,0,3,0,0,1]))
    >>> degree_encoder = DegreeEncoder(5, 16)
    >>> degree_embedding = degree_encoder(g)
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

    def forward(self, g):
        """
        Parameters
        ----------
        g : DGLGraph
            A DGLGraph to be encoded. Graphs with more than one type of edges
            are not allowed.

        Returns
        -------
        Tensor
            Return degree embedding vectors of shape :math:`(N, d)`,
            where :math:`N` is the number of nodes in the input graph and
            :math:`d` is :attr:`embedding_dim`.
        """
        if len(g.etypes) > 1:
            raise DGLError(
                "The input graph should have no more than one type of edges."
            )

        in_degree = th.clamp(g.in_degrees(), min=0, max=self.max_degree)
        out_degree = th.clamp(g.out_degrees(), min=0, max=self.max_degree)

        if self.direction == "in":
            degree_embedding = self.encoder(in_degree)
        elif self.direction == "out":
            degree_embedding = self.encoder(out_degree)
        elif self.direction == "both":
            degree_embedding = self.encoder1(in_degree) + self.encoder2(
                out_degree
            )
        else:
            raise ValueError(
                f'Supported direction options: "in", "out" and "both", '
                f"but got {self.direction}"
            )
        return degree_embedding
