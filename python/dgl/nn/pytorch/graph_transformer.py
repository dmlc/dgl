"""Torch modules for graph transformers."""
import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ...batch import unbatch
from ...convert import to_homogeneous
from ...transforms import shortest_dist

__all__ = [
    "DegreeEncoder",
    "BiasedMultiheadAttention",
    "PathEncoder",
    "GraphormerLayer",
    "SpatialEncoder",
    "SpatialEncoder3d",
]


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
            self.degree_encoder_1 = nn.Embedding(
                max_degree + 1, embedding_dim, padding_idx=0
            )
            self.degree_encoder_2 = nn.Embedding(
                max_degree + 1, embedding_dim, padding_idx=0
            )
        else:
            self.degree_encoder = nn.Embedding(
                max_degree + 1, embedding_dim, padding_idx=0
            )
        self.max_degree = max_degree

    def forward(self, g):
        """
        Parameters
        ----------
        g : DGLGraph
            A DGLGraph to be encoded. If it is a heterogeneous one,
            it will be transformed into a homogeneous one first.

        Returns
        -------
        Tensor
            Return degree embedding vectors of shape :math:`(N, embedding_dim)`,
            where :math:`N` is th number of nodes in the input graph.
        """
        if len(g.ntypes) > 1 or len(g.etypes) > 1:
            g = to_homogeneous(g)
        in_degree = th.clamp(g.in_degrees(), min=0, max=self.max_degree)
        out_degree = th.clamp(g.out_degrees(), min=0, max=self.max_degree)

        if self.direction == "in":
            degree_embedding = self.degree_encoder(in_degree)
        elif self.direction == "out":
            degree_embedding = self.degree_encoder(out_degree)
        elif self.direction == "both":
            degree_embedding = self.degree_encoder_1(
                in_degree
            ) + self.degree_encoder_2(out_degree)
        else:
            raise ValueError(
                f'Supported direction options: "in", "out" and "both", '
                f"but got {self.direction}"
            )
        return degree_embedding


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
            The input edge feature of shape :math:`(E, feat_dim)`,
            where :math:`E` is the number of edges in the input graph.

        Returns
        -------
        torch.Tensor
            Return attention bias as path encoding,
            of shape :math:`(batch_size, N, N, num_heads)`,
            where :math:`N` is the maximum number of nodes
            and batch_size is the batch size of the input graph.
        """

        g_list = unbatch(g)
        sum_num_edges = 0
        max_num_nodes = th.max(g.batch_num_nodes())
        path_encoding = []

        for ubg in g_list:
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
            # [n, n, h] -> [N, N, h], N = max_num_nodes, padded with -inf
            sub_encoding = th.full(
                (max_num_nodes, max_num_nodes, self.num_heads), float("-inf")
            )
            sub_encoding[0:num_nodes, 0:num_nodes] = th.div(
                th.einsum("xyld,lhd->xyh", path_data, edge_embedding).permute(
                    2, 0, 1
                ),
                shortest_distance,
            ).permute(1, 2, 0)
            path_encoding.append(sub_encoding)
        return th.stack(path_encoding, dim=0)


class BiasedMultiheadAttention(nn.Module):
    r"""Dense Multi-Head Attention Module with Graph Attention Bias.

    Compute attention between nodes with attention bias obtained from graph
    structures, as introduced in `Do Transformers Really Perform Bad for
    Graph Representation? <https://arxiv.org/pdf/2106.05234>`__

    .. math::

        \text{Attn}=\text{softmax}(\dfrac{QK^T}{\sqrt{d}} \circ b)

    :math:`Q` and :math:`K` are feature representation of nodes. :math:`d`
    is the corresponding :attr:`feat_size`. :math:`b` is attention bias, which
    can be additive or multiplicative according to the operator :math:`\circ`.

    Parameters
    ----------
    feat_size : int
        Feature size.
    num_heads : int
        Number of attention heads, by which attr:`feat_size` is divisible.
    bias : bool, optional
        If True, it uses bias for linear projection. Default: True.
    attn_bias_type : str, optional
        The type of attention bias used for modifying attention. Selected from
        'add' or 'mul'. Default: 'add'.

        * 'add' is for additive attention bias.
        * 'mul' is for multiplicative attention bias.
    attn_drop : float, optional
        Dropout probability on attention weights. Defalt: 0.1.

    Examples
    --------
    >>> import torch as th
    >>> from dgl.nn import BiasedMultiheadAttention

    >>> ndata = th.rand(16, 100, 512)
    >>> bias = th.rand(16, 100, 100, 8)
    >>> net = BiasedMultiheadAttention(feat_size=512, num_heads=8)
    >>> out = net(ndata, bias)
    """

    def __init__(
        self,
        feat_size,
        num_heads,
        bias=True,
        attn_bias_type="add",
        attn_drop=0.1,
    ):
        super().__init__()
        self.feat_size = feat_size
        self.num_heads = num_heads
        self.head_dim = feat_size // num_heads
        assert (
            self.head_dim * num_heads == feat_size
        ), "feat_size must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self.attn_bias_type = attn_bias_type

        self.q_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.k_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.v_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.out_proj = nn.Linear(feat_size, feat_size, bias=bias)

        self.dropout = nn.Dropout(p=attn_drop)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters of projection matrices, the same settings as that in Graphormer."""
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2**-0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2**-0.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2**-0.5)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, ndata, attn_bias=None, attn_mask=None):
        """Forward computation.

        Parameters
        ----------
        ndata : torch.Tensor
            A 3D input tensor. Shape: (batch_size, N, :attr:`feat_size`), where
            N is the maximum number of nodes.
        attn_bias : torch.Tensor, optional
            The attention bias used for attention modification. Shape:
            (batch_size, N, N, :attr:`num_heads`).
        attn_mask : torch.Tensor, optional
            The attention mask used for avoiding computation on invalid positions, where
            invalid positions are indicated by non-zero values. Shape: (batch_size, N, N).

        Returns
        -------
        y : torch.Tensor
            The output tensor. Shape: (batch_size, N, :attr:`feat_size`)
        """
        q_h = self.q_proj(ndata).transpose(0, 1)
        k_h = self.k_proj(ndata).transpose(0, 1)
        v_h = self.v_proj(ndata).transpose(0, 1)
        bsz, N, _ = ndata.shape
        q_h = (
            q_h.reshape(N, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            / self.scaling
        )
        k_h = k_h.reshape(N, bsz * self.num_heads, self.head_dim).permute(
            1, 2, 0
        )
        v_h = v_h.reshape(N, bsz * self.num_heads, self.head_dim).transpose(
            0, 1
        )

        attn_weights = (
            th.bmm(q_h, k_h)
            .transpose(0, 2)
            .reshape(N, N, bsz, self.num_heads)
            .transpose(0, 2)
        )

        if attn_bias is not None:
            if self.attn_bias_type == "add":
                attn_weights += attn_bias
            else:
                attn_weights *= attn_bias
        if attn_mask is not None:
            attn_weights[attn_mask.to(th.bool)] = float("-inf")
        attn_weights = F.softmax(
            attn_weights.transpose(0, 2)
            .reshape(N, N, bsz * self.num_heads)
            .transpose(0, 2),
            dim=2,
        )

        attn_weights = self.dropout(attn_weights)

        attn = th.bmm(attn_weights, v_h).transpose(0, 1)

        attn = self.out_proj(
            attn.reshape(N, bsz, self.feat_size).transpose(0, 1)
        )

        return attn


class GraphormerLayer(nn.Module):
    r"""Graphormer Layer with Dense Multi-Head Attention, as introduced
    in `Do Transformers Really Perform Bad for Graph Representation?
    <https://arxiv.org/pdf/2106.05234>`__

    Parameters
    ----------
    feat_size : int
        Feature size.
    hidden_size : int
        Hidden size of feedforward layers.
    num_heads : int
        Number of attention heads, by which :attr:`feat_size` is divisible.
    attn_bias_type : str, optional
        The type of attention bias used for modifying attention. Selected from
        'add' or 'mul'. Default: 'add'.

        * 'add' is for additive attention bias.
        * 'mul' is for multiplicative attention bias.
    norm_first : bool, optional
        If True, it performs layer normalization before attention and
        feedforward operations. Otherwise, it applies layer normalization
        afterwards. Default: False.
    dropout : float, optional
        Dropout probability. Default: 0.1.
    activation : callable activation layer, optional
        Activation function. Default: nn.ReLU().

    Examples
    --------
    >>> import torch as th
    >>> from dgl.nn import GraphormerLayer

    >>> batch_size = 16
    >>> num_nodes = 100
    >>> feat_size = 512
    >>> num_heads = 8
    >>> nfeat = th.rand(batch_size, num_nodes, feat_size)
    >>> bias = th.rand(batch_size, num_nodes, num_nodes, num_heads)
    >>> net = GraphormerLayer(
            feat_size=feat_size,
            hidden_size=2048,
            num_heads=num_heads
        )
    >>> out = net(nfeat, bias)
    """

    def __init__(
        self,
        feat_size,
        hidden_size,
        num_heads,
        attn_bias_type="add",
        norm_first=False,
        dropout=0.1,
        activation=nn.ReLU(),
    ):
        super().__init__()

        self.norm_first = norm_first

        self.attn = BiasedMultiheadAttention(
            feat_size=feat_size,
            num_heads=num_heads,
            attn_bias_type=attn_bias_type,
            attn_drop=dropout,
        )
        self.ffn = nn.Sequential(
            nn.Linear(feat_size, hidden_size),
            activation,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, feat_size),
            nn.Dropout(p=dropout),
        )

        self.dropout = nn.Dropout(p=dropout)
        self.attn_layer_norm = nn.LayerNorm(feat_size)
        self.ffn_layer_norm = nn.LayerNorm(feat_size)

    def forward(self, nfeat, attn_bias=None, attn_mask=None):
        """Forward computation.

        Parameters
        ----------
        nfeat : torch.Tensor
            A 3D input tensor. Shape: (batch_size, N, :attr:`feat_size`), where
            N is the maximum number of nodes.
        attn_bias : torch.Tensor, optional
            The attention bias used for attention modification. Shape:
            (batch_size, N, N, :attr:`num_heads`).
        attn_mask : torch.Tensor, optional
            The attention mask used for avoiding computation on invalid
            positions. Shape: (batch_size, N, N).

        Returns
        -------
        y : torch.Tensor
            The output tensor. Shape: (batch_size, N, :attr:`feat_size`)
        """
        residual = nfeat
        if self.norm_first:
            nfeat = self.attn_layer_norm(nfeat)
        nfeat = self.attn(nfeat, attn_bias, attn_mask)
        nfeat = self.dropout(nfeat)
        nfeat = residual + nfeat
        if not self.norm_first:
            nfeat = self.attn_layer_norm(nfeat)
        residual = nfeat
        if self.norm_first:
            nfeat = self.ffn_layer_norm(nfeat)
        nfeat = self.ffn(nfeat)
        nfeat = residual + nfeat
        if not self.norm_first:
            nfeat = self.ffn_layer_norm(nfeat)
        return nfeat


class SpatialEncoder(nn.Module):
    r"""Spatial Encoder, as introduced in
    `Do Transformers Really Perform Bad for Graph Representation?
    <https://proceedings.neurips.cc/paper/2021/file/f1c1592588411002af340cbaedd6fc33-Paper.pdf>`__
    This module is a learnable spatial embedding module which encodes
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

    >>> u = th.tensor([0, 0, 0, 1, 1, 2, 3, 3])
    >>> v = th.tensor([1, 2, 3, 0, 3, 0, 0, 1])
    >>> g = dgl.graph((u, v))
    >>> spatial_encoder = SpatialEncoder(max_dist=2, num_heads=8)
    >>> out = spatial_encoder(g)
    >>> print(out.shape)
    torch.Size([1, 4, 4, 8])
    """

    def __init__(self, max_dist, num_heads=1):
        super().__init__()
        self.max_dist = max_dist
        self.num_heads = num_heads
        # deactivate node pair between which the distance is -1
        self.embedding_table = nn.Embedding(
            max_dist + 2, num_heads, padding_idx=0
        )

    def forward(self, g):
        """
        Parameters
        ----------
        g : DGLGraph
            A DGLGraph to be encoded, which must be a homogeneous one.

        Returns
        -------
        torch.Tensor
            Return attention bias as spatial encoding of shape
            :math:`(B, N, N, H)`, where :math:`N` is the maximum number of
            nodes, :math:`B` is the batch size of the input graph, and
            :math:`H` is :attr:`num_heads`.
        """
        device = g.device
        g_list = unbatch(g)
        max_num_nodes = th.max(g.batch_num_nodes())
        spatial_encoding = []

        for ubg in g_list:
            num_nodes = ubg.num_nodes()
            dist = (
                th.clamp(
                    shortest_dist(ubg, root=None, return_paths=False),
                    min=-1,
                    max=self.max_dist,
                )
                + 1
            )
            # shape: [n, n, h], n = num_nodes, h = num_heads
            dist_embedding = self.embedding_table(dist)
            # [n, n, h] -> [N, N, h], N = max_num_nodes, padded with -inf
            padded_encoding = th.full(
                (max_num_nodes, max_num_nodes, self.num_heads), float("-inf")
            ).to(device)
            padded_encoding[0:num_nodes, 0:num_nodes] = dist_embedding
            spatial_encoding.append(padded_encoding)
        return th.stack(spatial_encoding, dim=0)


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
        spatial_encoding = []
        sum_num_nodes = 0
        if (self.max_node_type == 1) != (node_type is None):
            raise ValueError(
                "input node_type should be None if and only if "
                "max_node_type is 1."
            )
        for ubg in g_list:
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
            # [n, n, a] -> [N, N, a], N = max_num_nodes, padded with -inf
            padded_encoding = th.full(
                (max_num_nodes, max_num_nodes, self.num_heads), float("-inf")
            ).to(device)
            padded_encoding[0:num_nodes, 0:num_nodes] = encoding
            spatial_encoding.append(padded_encoding)
            sum_num_nodes += num_nodes
        return th.stack(spatial_encoding, dim=0)
