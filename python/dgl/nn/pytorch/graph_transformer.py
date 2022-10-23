import torch
import torch.nn as nn
import dgl


class DegreeEncoder(nn.Module):
    r"""Degree Encoder, as introduced in
    `Do Transformers Really Perform Bad for Graph Representation?
    <https://proceedings.neurips.cc/paper/2021/file/f1c1592588411002af340cbaedd6fc33-Paper.pdf>`__
    This module is a learnable degree embedding module.

    Parameters
    ----------
    max_degree : int
        Upper bound of degrees to be encoded. Each degree will be clamped to (0, max_degree).
    embedding_dim : int
        Dimension of output embedding vector of each node.
    direction : str, optional
        Degrees of which direction to be encoded, only in "in", "out" and "both".
        Default : "both".

    Example
    ------
    >>> import dgl
    >>> from dgl.nn.pytorch import DegreeEncoder

    >>> g = dgl.graph(([0,0,0,1,1,2,3,3], [1,2,3,0,3,0,0,1]))
    >>> degree_encoder = DegreeEncoder(5, 16)
    >>> degree_embedding = degree_encoder(g)
    """
    def __init__(self, max_degree, embedding_dim, direction='both'):
        super(DegreeEncoder, self).__init__()
        self.direction = direction
        self.degree_encoder = nn.Embedding(max_degree + 1, embedding_dim, padding_idx=0)
        self.max_degree = max_degree

    def forward(self, g):
        """
        Parameters
        ----------
        g : DGLGraph
            A DGLGraph containing sampled nodes to be encoded.

        Returns
        -------
        Tensor
            Return degree embedding vectors of shape :math:`(N, embedding_dim)`, where :math:`N` is the
            number of nodes in the input graph.
        """
        if len(g.ntypes) > 1 or len(g.etypes) > 1:
            hg = dgl.to_homogeneous(g)
            in_degree = hg.in_degrees()
            out_degree = hg.out_degrees()
        else:
            in_degree = g.in_degrees()
            out_degree = g.out_degrees()

        in_degree.clamp(0, self.max_degree)
        out_degree.clamp(0, self.max_degree)

        if self.direction == 'in':
            degree_embedding = self.degree_encoder(in_degree)
        elif self.direction == 'out':
            degree_embedding = self.degree_encoder(out_degree)
        elif self.direction == 'both':
            degree_embedding = (self.degree_encoder(in_degree)
                                + self.degree_encoder(out_degree))
        else:
            raise ValueError(
                f'Supported direction options: "in", "out" and "both", but got {self.direction}')

        return degree_embedding