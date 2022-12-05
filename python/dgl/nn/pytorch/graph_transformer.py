"""Torch modules for graph transformers."""
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ...convert import to_homogeneous
from ... import function as fn
from ..functional import edge_softmax
from ...batch import unbatch
from ...transforms import shortest_dist


__all__ = [
    "DegreeEncoder",
    "BiasedMHA",
    "SparseBiasedMHA",
    "PathEncoder"
]

class DegreeEncoder(nn.Module):
    r"""Degree Encoder, as introduced in
    `Do Transformers Really Perform Bad for Graph Representation?
    <https://arxiv.org/abs/2106.05234>`__
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
    >>> print(degree_embedding.size())
    torch.Size([8, 16])
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
            degree_embedding = (self.degree_encoder_1(in_degree)
                                + self.degree_encoder_2(out_degree))
        else:
            raise ValueError(
                f'Supported direction options: "in", "out" and "both", '
                f'but got {self.direction}'
            )

        return degree_embedding


class BiasedMHA(nn.Module):
    r"""Dense Multi-Head Attention Module with Graph Attention Bias.

    Compute attention between nodes with attention bias obtained from graph
    structures, as introduced in `Do Transformers Really Perform Bad for
    Graph Representation? <https://arxiv.org/pdf/2106.05234>`__

    .. math::

        \text{Attn}=\text{softmax}(\dfrac{QK^T}{\sqrt{d}} \circ b)

    :math:`Q` and :math:`K` are feature representations of nodes. :math:`d`
    is the corresponding :attr:`feat_size`. :math:`b` is attention bias, which
    can be additive or multiplicative according to the operator :math:`\circ`.

    Parameters
    ----------
    feat_size : int
        Feature size.
    num_heads : int
        Number of attention heads, by which :attr:`feat_size` is divisible.
    bias : bool, optional
        If True, it uses bias for linear projection. Default: True.
    attn_bias_type : str, optional
        The type of attention bias used for modifying attention. Selected from
        'add', 'mul', or None. Default: 'add'.

        * 'add' is for additive attention bias.
        * 'mul' is for multiplicative attention bias.
        * None is for no attention bias. :attr:`attn_bias` in forward
        computation will be ignored.
    dropout : float, optional
        Dropout probability on attention weights. Default: 0.1.

    Examples
    --------
    >>> import torch as th
    >>> from dgl.nn import BiasedMHA

    >>> ndata = th.rand(16, 100, 512)
    >>> bias = th.rand(16, 100, 100, 8)
    >>> net = BiasedMHA(feat_size=512, num_heads=8)
    >>> out = net(ndata, bias)
    >>> print(out.size())
    torch.Size([16, 100, 512])
    """

    def __init__(
        self, feat_size, num_heads, bias=True, attn_bias_type="add",
        dropout=0.1
    ):
        super().__init__()
        self.feat_size = feat_size
        self.num_heads = num_heads
        self.head_dim = feat_size // num_heads
        assert (
            self.head_dim * num_heads == feat_size
        ), "feat_size must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.attn_bias_type = attn_bias_type

        self.q_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.k_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.v_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.out_proj = nn.Linear(feat_size, feat_size, bias=bias)

        self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters of projection matrices, the same settings as that
        in Graphormer.
        """
        gain = 2 ** -0.5
        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=gain)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, nfeat, attn_bias=None, attn_mask=None):
        """Forward computation.

        Parameters
        ----------
        nfeat : torch.Tensor
            A 3D input tensor. Shape: (batch_size, N, :attr:`feat_size`), where
            N is the maximum number of nodes.
        attn_bias : torch.Tensor, optional
            The attention bias used for attention modification. Shape:
            (batch_size, N, N, :attr:`num_heads`). Default: None
        attn_mask : torch.Tensor, optional
            The attention mask used for avoiding computation on invalid
            positions, where invalid positions are indicated by non-zero values.
            Shape: (batch_size, N, N). Default: None.

        Returns
        -------
        torch.Tensor
            The output tensor. Shape: (batch_size, N, :attr:`feat_size`)
        """
        q_h = self.q_proj(nfeat).transpose(0, 1)
        k_h = self.k_proj(nfeat).transpose(0, 1)
        v_h = self.v_proj(nfeat).transpose(0, 1)
        bsz, N, _ = nfeat.shape
        q_h = q_h.reshape(
            N, bsz * self.num_heads, self.head_dim
        ).transpose(0, 1) / self.scaling
        k_h = k_h.reshape(
            N, bsz * self.num_heads, self.head_dim
        ).permute(1, 2, 0)
        v_h = v_h.reshape(
            N, bsz * self.num_heads, self.head_dim
        ).transpose(0, 1)

        attn_weights = (
            th.bmm(q_h, k_h)
            .transpose(0, 2)
            .reshape(N, N, bsz, self.num_heads)
            .transpose(0, 2)
        )

        if self.attn_bias_type == "add":
            attn_weights += attn_bias
        elif self.attn_bias_type == 'mul':
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


class SparseBiasedMHA(nn.Module):
    r"""Sparse Multi-Head Attention Module with Graph Attention Bias.

    Implement sparse attention computation restricted to neighborhoods of
    nodes, as introduced in `A Generalization of Transformer Networks to
    Graphs <https://arxiv.org/pdf/2012.09699>`__

    .. math::

        \text{Attn}_{ij}=\text{softmax}_{j\in\mathcal{N}(i)}(\dfrac{Q_iK_j^T}
        {\sqrt{d}} \circ b)

    :math:`Q_i` and :math:`K_j` are feature representations of nodes. :math:`d`
    is the corresponding :attr:`feat_size`. :math:`b` is attention bias, which
    can be additive or multiplicative according to the operator :math:`\circ`.
    :math:`\mathcal{N}(i)` is the set of nodes that have an outgoing edge to
    the node :math:`i`.

    Parameters
    ----------
    feat_size : int
        Feature size.
    num_heads : int
        Number of attention heads, by which :attr:`feat_size` is divisible.
    bias : bool, optional
        If True, it uses bias for linear projection. Default: False.
    attn_bias_type : str, optional
        The type of attention bias used for modifying attention. Selected from
        'add', 'mul', or None. Default: None.

        * 'add' is for additive attention bias.
        * 'mul' is for multiplicative attention bias.
        * None is for no attention bias. :attr:`attn_bias` in forward
        computation will be ignored.
    dropout : float, optional
        Dropout probability on attention weights. Default: 0.

    Examples
    --------
    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import SparseBiasedMHA

    >>> num_nodes = 100
    >>> num_edges = 80
    >>> feat_size = 512
    >>> num_heads = 8

    >>> g = dgl.rand_graph(num_nodes, num_edges)
    >>> nfeat = th.rand(num_nodes, feat_size)
    >>> bias = th.rand(num_edges, num_heads)
    >>> net = SparseBiasedMHA(feat_size=feat_size, num_heads=num_heads)
    >>> out = net(g, nfeat, bias)
    >>> print(out.size())
    torch.Size([100, 512])
    """

    def __init__(
        self, feat_size, num_heads, bias=False, attn_bias_type=None,
        dropout=0.
    ):
        super().__init__()
        self.feat_size = feat_size
        self.num_heads = num_heads
        self.head_dim = feat_size // num_heads
        assert (
            self.head_dim * num_heads == feat_size
        ), "feat_size must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.attn_bias_type = attn_bias_type

        self.q_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.k_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.v_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.out_proj = nn.Linear(feat_size, feat_size, bias=bias)

        self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters of projection matrices, the same settings as that
        in Graphormer.
        """
        gain = 2 ** -0.5
        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=gain)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)

    def forward(self, g, nfeat, attn_bias=None):
        """Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The input graph of N nodes and E edges.
        nfeat : torch.Tensor
            A 2D input tensor of node features. Shape: (N, :attr:`feat_size`).
        attn_bias : torch.Tensor, optional
            The attention bias used for attention modification.
            Shape: (E, :attr:`num_heads`). Default: None.

        Returns
        -------
        torch.Tensor
            The output tensor. Shape: (N, :attr:`feat_size`)
        """
        N = g.num_nodes()
        with g.local_scope():
            # linear projection
            g.ndata['q_h'] = self.q_proj(nfeat).reshape(
                N, self.num_heads, self.head_dim
            ) / self.scaling
            g.ndata['k_h'] = self.k_proj(nfeat).reshape(
                N, self.num_heads, self.head_dim
            )
            g.ndata['v_h'] = self.v_proj(nfeat).reshape(
                N, self.num_heads, self.head_dim
            )

            # compute attention weights
            g.apply_edges(fn.u_dot_v('k_h', 'q_h', 'attn'))
            if self.attn_bias_type == 'add':
                g.edata['attn'] += attn_bias.unsqueeze(-1)
            elif self.attn_bias_type == 'mul':
                g.edata['attn'] *= attn_bias.unsqueeze(-1)
            g.edata['attn'] = self.dropout(
                edge_softmax(g, g.edata['attn'], norm_by="dst")
            )

            # compute output
            g.update_all(fn.u_mul_e('v_h', 'attn', 'v_a'), fn.sum('v_a', 'out'))
            attn_val = self.out_proj(g.ndata['out'].reshape(N, self.feat_size))

            return attn_val


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

    >>> u = th.tensor([0, 0, 0, 1, 1, 2, 3, 3])
    >>> v = th.tensor([1, 2, 3, 0, 3, 0, 0, 1])
    >>> g = dgl.graph((u, v))
    >>> edata = th.rand(8, 16)
    >>> path_encoder = dgl.PathEncoder(2, 16, 8)
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
            edata = edge_feat[sum_num_edges: (sum_num_edges + num_edges)]
            sum_num_edges = sum_num_edges + num_edges
            edata = th.cat(
                (edata, th.zeros(1, self.feat_dim).to(edata.device)),
                dim=0
            )
            _, path = shortest_dist(ubg, root=None, return_paths=True)
            path_len = min(self.max_len, path.size(dim=2))

            # shape: [n, n, l], n = num_nodes, l = path_len
            shortest_path = path[:, :, 0: path_len]
            # shape: [n, n]
            shortest_distance = th.clamp(
                shortest_dist(ubg, root=None, return_paths=False),
                min=1,
                max=path_len
            )
            # shape: [n, n, l, d], d = feat_dim
            path_data = edata[shortest_path]
            # shape: [l, h], h = num_heads
            embedding_idx = th.reshape(
                th.arange(self.num_heads * path_len),
                (path_len, self.num_heads)
            ).to(next(self.embedding_table.parameters()).device)
            # shape: [d, l, h]
            edge_embedding = th.permute(
                self.embedding_table(embedding_idx), (2, 0, 1)
            )

            # [n, n, l, d] einsum [d, l, h] -> [n, n, h]
            # [n, n, h] -> [N, N, h], N = max_num_nodes, padded with -inf
            sub_encoding = th.full(
                (max_num_nodes, max_num_nodes, self.num_heads),
                float('-inf')
            )
            sub_encoding[0: num_nodes, 0: num_nodes] = th.div(
                th.einsum(
                    'xyld,dlh->xyh', path_data, edge_embedding
                ).permute(2, 0, 1),
                shortest_distance
            ).permute(1, 2, 0)
            path_encoding.append(sub_encoding)

        return th.stack(path_encoding, dim=0)
