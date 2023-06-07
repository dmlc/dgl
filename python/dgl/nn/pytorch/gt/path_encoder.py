"""Path Encoder"""
import torch as th
import torch.nn as nn


class PathEncoder(nn.Module):
    r"""Path Encoder, as introduced in Edge Encoding of
    `Do Transformers Really Perform Bad for Graph Representation?
    <https://proceedings.neurips.cc/paper/2021/file/f1c1592588411002af340cbaedd6fc33-Paper.pdf>`__

    This module is a learnable path embedding module and encodes the shortest
    path between each pair of nodes as attention bias.

    Parameters
    ----------
    max_len : int
        Maximum number of edges in each path to be encoded.
        Exceeding part of each path will be truncated, i.e.
        truncating edges with serial number no less than :attr:`max_len`.
    feat_dim : int
        Dimension of edge features in the input graph.
    num_heads : int, optional
        Number of attention heads if multi-head attention mechanism is applied.
        Default : 1.

    Examples
    --------
    >>> import torch as th
    >>> import dgl
    >>> from dgl.nn import PathEncoder
    >>> from dgl import shortest_dist

    >>> g = dgl.graph(([0,0,0,1,1,2,3,3], [1,2,3,0,3,0,0,1]))
    >>> edata = th.rand(8, 16)
    >>> # Since shortest_dist returns -1 for unreachable node pairs,
    >>> # edata[-1] should be filled with zero padding.
    >>> edata = th.cat(
            (edata, th.zeros(1, 16)), dim=0
        )
    >>> dist, path = shortest_dist(g, root=None, return_paths=True)
    >>> path_data = edata[path[:, :, :2]]
    >>> path_encoder = PathEncoder(2, 16, num_heads=8)
    >>> out = path_encoder(dist.unsqueeze(0), path_data.unsqueeze(0))
    >>> print(out.shape)
    torch.Size([1, 4, 4, 8])
    """

    def __init__(self, max_len, feat_dim, num_heads=1):
        super().__init__()
        self.max_len = max_len
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.embedding_table = nn.Embedding(max_len * num_heads, feat_dim)

    def forward(self, dist, path_data):
        """
        Parameters
        ----------
        dist : Tensor
            Shortest path distance matrix of the batched graph with zero padding,
            of shape :math:`(B, N, N)`, where :math:`B` is the batch size of
            the batched graph, and :math:`N` is the maximum number of nodes.
        path_data : Tensor
            Edge feature along the shortest path with zero padding, of shape
            :math:`(B, N, N, L, d)`, where :math:`L` is the maximum length of
            the shortest paths, and :math:`d` is :attr:`feat_dim`.

        Returns
        -------
        torch.Tensor
            Return attention bias as path encoding, of shape
            :math:`(B, N, N, H)`, where :math:`B` is the batch size of
            the input graph, :math:`N` is the maximum number of nodes, and
            :math:`H` is :attr:`num_heads`.
        """
        shortest_distance = th.clamp(dist, min=1, max=self.max_len)
        edge_embedding = self.embedding_table.weight.reshape(
            self.max_len, self.num_heads, -1
        )
        path_encoding = th.div(
            th.einsum("bxyld,lhd->bxyh", path_data, edge_embedding).permute(
                3, 0, 1, 2
            ),
            shortest_distance,
        ).permute(1, 2, 3, 0)
        return path_encoding
