"""Spatial Encoder"""

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ....batch import unbatch


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

    This module encodes pair-wise relation between atom pair :math:`(i,j)` in
    the 3D geometric space, according to the Gaussian Basis Kernel function:

    :math:`\psi _{(i,j)} ^k = -\frac{1}{\sqrt{2\pi} \lvert \sigma^k \rvert}
    \exp{\left ( -\frac{1}{2} \left( \frac{\gamma_{(i,j)} \lvert \lvert r_i -
    r_j \rvert \rvert + \beta_{(i,j)} - \mu^k}{\lvert \sigma^k \rvert} \right)
    ^2 \right)}，k=1,...,K,`

    where :math:`K` is the number of Gaussian Basis kernels.
    :math:`r_i` is the Cartesian coordinate of atom :math:`i`.
    :math:`\gamma_{(i,j)}, \beta_{(i,j)}` are learnable scaling factors of
    the Gaussian Basis kernels.

    Parameters
    ----------
    num_kernels : int
        Number of Gaussian Basis Kernels to be applied.
        Each Gaussian Basis Kernel contains a learnable kernel center
        and a learnable scaling factor.
    num_heads : int, optional
        Number of attention heads if multi-head attention mechanism is applied.
        Default : 1.
    max_node_type : int, optional
        Maximum number of node types. Default : 1.

    Examples
    --------
    >>> import torch as th
    >>> import dgl
    >>> from dgl.nn import SpatialEncoder3d

    >>> u = th.tensor([0, 0, 0, 1, 1, 2, 3, 3])
    >>> v = th.tensor([1, 2, 3, 0, 3, 0, 0, 1])
    >>> g = dgl.graph((u, v))
    >>> coordinate = th.rand(4, 3)
    >>> node_type = th.tensor([1, 0, 2, 1])
    >>> spatial_encoder = SpatialEncoder3d(num_kernels=4,
    ...                                    num_heads=8,
    ...                                    max_node_type=3)
    >>> out = spatial_encoder(g, coordinate, node_type=node_type)
    >>> print(out.shape)
    torch.Size([1, 4, 4, 8])
    """

    def __init__(self, num_kernels, num_heads=1, max_node_type=1):
        super().__init__()
        self.num_kernels = num_kernels
        self.num_heads = num_heads
        self.max_node_type = max_node_type
        self.gaussian_means = nn.Embedding(1, num_kernels)
        self.gaussian_stds = nn.Embedding(1, num_kernels)
        self.linear_layer_1 = nn.Linear(num_kernels, num_kernels)
        self.linear_layer_2 = nn.Linear(num_kernels, num_heads)
        if max_node_type == 1:
            self.mul = nn.Embedding(1, 1)
            self.bias = nn.Embedding(1, 1)
        else:
            self.mul = nn.Embedding(max_node_type + 1, 2)
            self.bias = nn.Embedding(max_node_type + 1, 2)
        nn.init.uniform_(self.gaussian_means.weight, 0, 3)
        nn.init.uniform_(self.gaussian_stds.weight, 0, 3)
        nn.init.constant_(self.mul.weight, 0)
        nn.init.constant_(self.bias.weight, 1)

    def forward(self, g, coord, node_type=None):
        """
        Parameters
        ----------
        g : DGLGraph
            A DGLGraph to be encoded, which must be a homogeneous one.
        coord : torch.Tensor
            3D coordinates of nodes in :attr:`g`,
            of shape :math:`(N, 3)`,
            where :math:`N`: is the number of nodes in :attr:`g`.
        node_type : torch.Tensor, optional
            Node types of :attr:`g`. Default : None.

            * If :attr:`max_node_type` is not 1, :attr:`node_type` needs to
              be a tensor in shape :math:`(N,)`. The scaling factors of
              each pair of nodes are determined by their node types.
            * Otherwise, :attr:`node_type` should be None.

        Returns
        -------
        torch.Tensor
            Return attention bias as 3D spatial encoding of shape
            :math:`(B, n, n, H)`, where :math:`B` is the batch size, :math:`n`
            is the maximum number of nodes in unbatched graphs from :attr:`g`,
            and :math:`H` is :attr:`num_heads`.
        """

        device = g.device
        g_list = unbatch(g)
        max_num_nodes = th.max(g.batch_num_nodes())
        spatial_encoding = th.zeros(
            len(g_list), max_num_nodes, max_num_nodes, self.num_heads
        ).to(device)
        sum_num_nodes = 0
        if (self.max_node_type == 1) != (node_type is None):
            raise ValueError(
                "input node_type should be None if and only if "
                "max_node_type is 1."
            )
        for i, ubg in enumerate(g_list):
            num_nodes = ubg.num_nodes()
            sub_coord = coord[sum_num_nodes : sum_num_nodes + num_nodes]
            # shape: [n, n], n = num_nodes
            euc_dist = th.cdist(sub_coord, sub_coord, p=2)
            if node_type is None:
                # shape: [1]
                mul = self.mul.weight[0, 0]
                bias = self.bias.weight[0, 0]
            else:
                sub_node_type = node_type[
                    sum_num_nodes : sum_num_nodes + num_nodes
                ]
                mul_embedding = self.mul(sub_node_type)
                bias_embedding = self.bias(sub_node_type)
                # shape: [n, n]
                mul = mul_embedding[:, 0].unsqueeze(-1).repeat(
                    1, num_nodes
                ) + mul_embedding[:, 1].unsqueeze(0).repeat(num_nodes, 1)
                bias = bias_embedding[:, 0].unsqueeze(-1).repeat(
                    1, num_nodes
                ) + bias_embedding[:, 1].unsqueeze(0).repeat(num_nodes, 1)
            # shape: [n, n, k], k = num_kernels
            scaled_dist = (
                (mul * euc_dist + bias)
                .repeat(self.num_kernels, 1, 1)
                .permute((1, 2, 0))
            )
            # shape: [k]
            gaussian_mean = self.gaussian_means.weight.float().view(-1)
            gaussian_var = (
                self.gaussian_stds.weight.float().view(-1).abs() + 1e-2
            )
            # shape: [n, n, k]
            gaussian_kernel = (
                (
                    -0.5
                    * (
                        th.div(
                            scaled_dist - gaussian_mean, gaussian_var
                        ).square()
                    )
                )
                .exp()
                .div(-math.sqrt(2 * math.pi) * gaussian_var)
            )

            encoding = self.linear_layer_1(gaussian_kernel)
            encoding = F.gelu(encoding)
            # [n, n, k] -> [n, n, a], a = num_heads
            encoding = self.linear_layer_2(encoding)
            spatial_encoding[i, :num_nodes, :num_nodes] = encoding
            sum_num_nodes += num_nodes
        return spatial_encoding
