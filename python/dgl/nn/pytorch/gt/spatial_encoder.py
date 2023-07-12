"""Spatial Encoder"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F


def gaussian(x, mean, std):
    """compute gaussian basis kernel function"""
    const_pi = 3.14159
    a = (2 * const_pi) ** 0.5
    return th.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class SpatialEncoder(nn.Module):
    r"""Spatial Encoder, as introduced in
    `Do Transformers Really Perform Bad for Graph Representation?
    <https://proceedings.neurips.cc/paper/2021/file/f1c1592588411002af340cbaedd6fc33-Paper.pdf>`__

    This module is a learnable spatial embedding module, which encodes
    the shortest distance between each node pair for attention bias.

    Parameters
    ----------
    max_dist : int
        Upper bound of the shortest path distance
        between each node pair to be encoded.
        All distance will be clamped into the range `[0, max_dist]`.
    num_heads : int, optional
        Number of attention heads if multi-head attention mechanism is applied.
        Default : 1.

    Examples
    --------
    >>> import torch as th
    >>> import dgl
    >>> from dgl.nn import SpatialEncoder
    >>> from dgl import shortest_dist

    >>> g1 = dgl.graph(([0,0,0,1,1,2,3,3], [1,2,3,0,3,0,0,1]))
    >>> g2 = dgl.graph(([0,1], [1,0]))
    >>> n1, n2 = g1.num_nodes(), g2.num_nodes()
    >>> # use -1 padding since shortest_dist returns -1 for unreachable node pairs
    >>> dist = -th.ones((2, 4, 4), dtype=th.long)
    >>> dist[0, :n1, :n1] = shortest_dist(g1, root=None, return_paths=False)
    >>> dist[1, :n2, :n2] = shortest_dist(g2, root=None, return_paths=False)
    >>> spatial_encoder = SpatialEncoder(max_dist=2, num_heads=8)
    >>> out = spatial_encoder(dist)
    >>> print(out.shape)
    torch.Size([2, 4, 4, 8])
    """

    def __init__(self, max_dist, num_heads=1):
        super().__init__()
        self.max_dist = max_dist
        self.num_heads = num_heads
        # deactivate node pair between which the distance is -1
        self.embedding_table = nn.Embedding(
            max_dist + 2, num_heads, padding_idx=0
        )

    def forward(self, dist):
        """
        Parameters
        ----------
        dist : Tensor
            Shortest path distance of the batched graph with -1 padding, a tensor
            of shape :math:`(B, N, N)`, where :math:`B` is the batch size of
            the batched graph, and :math:`N` is the maximum number of nodes.

        Returns
        -------
        torch.Tensor
            Return attention bias as spatial encoding of shape
            :math:`(B, N, N, H)`, where :math:`H` is :attr:`num_heads`.
        """
        spatial_encoding = self.embedding_table(
            th.clamp(
                dist,
                min=-1,
                max=self.max_dist,
            )
            + 1
        )
        return spatial_encoding


class SpatialEncoder3d(nn.Module):
    r"""3D Spatial Encoder, as introduced in
    `One Transformer Can Understand Both 2D & 3D Molecular Data
    <https://arxiv.org/pdf/2210.01765.pdf>`__

    This module encodes pair-wise relation between node pair :math:`(i,j)` in
    the 3D geometric space, according to the Gaussian Basis Kernel function:

    :math:`\psi _{(i,j)} ^k = \frac{1}{\sqrt{2\pi} \lvert \sigma^k \rvert}
    \exp{\left ( -\frac{1}{2} \left( \frac{\gamma_{(i,j)} \lvert \lvert r_i -
    r_j \rvert \rvert + \beta_{(i,j)} - \mu^k}{\lvert \sigma^k \rvert} \right)
    ^2 \right)}，k=1,...,K,`

    where :math:`K` is the number of Gaussian Basis kernels. :math:`r_i` is the
    Cartesian coordinate of node :math:`i`.
    :math:`\gamma_{(i,j)}, \beta_{(i,j)}` are learnable scaling factors and
    biases determined by node types. :math:`\mu^k, \sigma^k` are learnable
    centers and standard deviations of the Gaussian Basis kernels.

    Parameters
    ----------
    num_kernels : int
        Number of Gaussian Basis Kernels to be applied. Each Gaussian Basis
        Kernel contains a learnable kernel center and a learnable standard
        deviation.
    num_heads : int, optional
        Number of attention heads if multi-head attention mechanism is applied.
        Default : 1.
    max_node_type : int, optional
        Maximum number of node types. Each node type has a corresponding
        learnable scaling factor and a bias. Default : 100.

    Examples
    --------
    >>> import torch as th
    >>> import dgl
    >>> from dgl.nn import SpatialEncoder3d

    >>> coordinate = th.rand(1, 4, 3)
    >>> node_type = th.tensor([[1, 0, 2, 1]])
    >>> spatial_encoder = SpatialEncoder3d(num_kernels=4,
    ...                                    num_heads=8,
    ...                                    max_node_type=3)
    >>> out = spatial_encoder(coordinate, node_type=node_type)
    >>> print(out.shape)
    torch.Size([1, 4, 4, 8])
    """

    def __init__(self, num_kernels, num_heads=1, max_node_type=100):
        super().__init__()
        self.num_kernels = num_kernels
        self.num_heads = num_heads
        self.max_node_type = max_node_type
        self.means = nn.Parameter(th.empty(num_kernels))
        self.stds = nn.Parameter(th.empty(num_kernels))
        self.linear_layer_1 = nn.Linear(num_kernels, num_kernels)
        self.linear_layer_2 = nn.Linear(num_kernels, num_heads)
        # There are 2 * max_node_type + 3 pairs of gamma and beta parameters:
        # 1. Parameters at position 0 are for default gamma/beta when no node
        #    type is given
        # 2. Parameters at position 1 to max_node_type+1 are for src node types.
        #    (position 1 is for padded unexisting nodes)
        # 3. Parameters at position max_node_type+2 to 2*max_node_type+2 are
        #    for tgt node types. (position max_node_type+2 is for padded)
        #    unexisting nodes)
        self.gamma = nn.Embedding(2 * max_node_type + 3, 1, padding_idx=0)
        self.beta = nn.Embedding(2 * max_node_type + 3, 1, padding_idx=0)

        nn.init.uniform_(self.means, 0, 3)
        nn.init.uniform_(self.stds, 0, 3)
        nn.init.constant_(self.gamma.weight, 1)
        nn.init.constant_(self.beta.weight, 0)

    def forward(self, coord, node_type=None):
        """
        Parameters
        ----------
        coord : torch.Tensor
            3D coordinates of nodes in shape :math:`(B, N, 3)`, where :math:`B`
            is the batch size, :math:`N`: is the maximum number of nodes.
        node_type : torch.Tensor, optional
            Node type ids of nodes. Default : None.

            * If specified, :attr:`node_type` should be a tensor in shape
              :math:`(B, N,)`. The scaling factors in gaussian kernels of each
              pair of nodes are determined by their node types.
            * Otherwise, :attr:`node_type` will be set to zeros of the same
              shape by default.

        Returns
        -------
        torch.Tensor
            Return attention bias as 3D spatial encoding of shape
            :math:`(B, N, N, H)`, where :math:`H` is :attr:`num_heads`.
        """
        bsz, N = coord.shape[:2]
        euc_dist = th.cdist(coord, coord, p=2.0)  # shape: [B, n, n]
        if node_type is None:
            node_type = th.zeros([bsz, N, N, 2], device=coord.device).long()
        else:
            src_node_type = node_type.unsqueeze(-1).repeat(1, 1, N)
            tgt_node_type = node_type.unsqueeze(1).repeat(1, N, 1)
            node_type = th.stack(
                [src_node_type + 2, tgt_node_type + self.max_node_type + 3],
                dim=-1,
            )  # shape: [B, n, n, 2]

        # scaled euclidean distance
        gamma = self.gamma(node_type).sum(dim=-2)  # shape: [B, n, n, 1]
        beta = self.beta(node_type).sum(dim=-2)  # shape: [B, n, n, 1]
        euc_dist = gamma * euc_dist.unsqueeze(-1) + beta  # shape: [B, n, n, 1]
        # gaussian basis kernel
        euc_dist = euc_dist.expand(-1, -1, -1, self.num_kernels)
        gaussian_kernel = gaussian(
            euc_dist, self.means, self.stds.abs() + 1e-2
        )  # shape: [B, n, n, K]
        # linear projection
        encoding = self.linear_layer_1(gaussian_kernel)
        encoding = F.gelu(encoding)
        encoding = self.linear_layer_2(encoding)  # shape: [B, n, n, H]

        return encoding
