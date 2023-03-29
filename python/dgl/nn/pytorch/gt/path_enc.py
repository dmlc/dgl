"""Path Encoder"""
import torch as th
import torch.nn as nn

from ....batch import unbatch
from ....transforms import shortest_dist


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

    >>> u = th.tensor([0, 0, 0, 1, 1, 2, 3, 3])
    >>> v = th.tensor([1, 2, 3, 0, 3, 0, 0, 1])
    >>> g = dgl.graph((u, v))
    >>> edata = th.rand(8, 16)
    >>> path_encoder = PathEncoder(2, 16, num_heads=8)
    >>> out = path_encoder(g, edata)
    """

    def __init__(self, max_len, feat_dim, num_heads=1):
        super().__init__()
        self.max_len = max_len
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.embedding_table = nn.Embedding(max_len * num_heads, feat_dim)

    def forward(self, g, edge_feat):
        """
        Parameters
        ----------
        g : DGLGraph
            A DGLGraph to be encoded, which must be a homogeneous one.
        edge_feat : torch.Tensor
            The input edge feature of shape :math:`(E, d)`,
            where :math:`E` is the number of edges in the input graph and
            :math:`d` is :attr:`feat_dim`.

        Returns
        -------
        torch.Tensor
            Return attention bias as path encoding, of shape
            :math:`(B, N, N, H)`, where :math:`B` is the batch size of
            the input graph, :math:`N` is the maximum number of nodes, and
            :math:`H` is :attr:`num_heads`.
        """
        device = g.device
        g_list = unbatch(g)
        sum_num_edges = 0
        max_num_nodes = th.max(g.batch_num_nodes())
        path_encoding = th.zeros(
            len(g_list), max_num_nodes, max_num_nodes, self.num_heads
        ).to(device)

        for i, ubg in enumerate(g_list):
            num_nodes = ubg.num_nodes()
            num_edges = ubg.num_edges()
            edata = edge_feat[sum_num_edges : (sum_num_edges + num_edges)]
            sum_num_edges = sum_num_edges + num_edges
            edata = th.cat(
                (edata, th.zeros(1, self.feat_dim).to(edata.device)), dim=0
            )
            dist, path = shortest_dist(ubg, root=None, return_paths=True)
            path_len = max(1, min(self.max_len, path.size(dim=2)))

            # shape: [n, n, l], n = num_nodes, l = path_len
            shortest_path = path[:, :, 0:path_len]
            # shape: [n, n]
            shortest_distance = th.clamp(dist, min=1, max=path_len)
            # shape: [n, n, l, d], d = feat_dim
            path_data = edata[shortest_path]
            # shape: [l, h, d]
            edge_embedding = self.embedding_table.weight[
                0 : path_len * self.num_heads
            ].reshape(path_len, self.num_heads, -1)
            # [n, n, l, d] einsum [l, h, d] -> [n, n, h]
            path_encoding[i, :num_nodes, :num_nodes] = th.div(
                th.einsum("xyld,lhd->xyh", path_data, edge_embedding).permute(
                    2, 0, 1
                ),
                shortest_distance,
            ).permute(1, 2, 0)
        return path_encoding
