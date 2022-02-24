"""Torch modules for graph attention networks with fully valuable edges (EGAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from torch.nn import init

from .... import function as fn
from ...functional import edge_softmax
from ....base import DGLError

# pylint: enable=W0235
class EGATConv(nn.Module):
    r"""Graph attention layer that handles edge features from `Rossmann-Toolbox
    <https://pubmed.ncbi.nlm.nih.gov/34571541/>`__ (see supplementary data)

    The difference lies in how unnormalized attention scores :math:`e_{ij}` are obtained:

    .. math::
        e_{ij} &= \vec{F} (f_{ij}^{\prime})

        f_{ij}^{\prime} &= \mathrm{LeakyReLU}\left(A [ h_{i} \| f_{ij} \| h_{j}]\right)

    where :math:`f_{ij}^{\prime}` are edge features, :math:`\mathrm{A}` is weight matrix and

    :math: `\vec{F}` is weight vector. After that, resulting node features
    :math:`h_{i}^{\prime}` are updated in the same way as in regular GAT.

    Parameters
    ----------
    in_node_feats : int
        Input node feature size :math:`h_{i}`.
    in_edge_feats : int
        Input edge feature size :math:`f_{ij}`.
    out_node_feats : int
        Output node feature size.
    out_edge_feats : int
        Output edge feature size :math:`f_{ij}^{\prime}`.
    num_heads : int
        Number of attention heads.
    bias : bool, optional
        If True, add bias term to :math: `f_{ij}^{\prime}`. Defaults: ``True``.

    Examples
    ----------
    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import EGATConv

    >>> num_nodes, num_edges = 8, 30
    >>> # generate a graph
    >>> graph = dgl.rand_graph(num_nodes,num_edges)

    >>> node_feats = th.rand((num_nodes, 20))
    >>> edge_feats = th.rand((num_edges, 12))
    >>> egat = EGATConv(in_node_feats=20,
    ...                 in_edge_feats=12,
    ...                 out_node_feats=15,
    ...                 out_edge_feats=10,
    ...                 num_heads=3)
    >>> #forward pass
    >>> new_node_feats, new_edge_feats = egat(graph, node_feats, edge_feats)
    >>> new_node_feats.shape, new_edge_feats.shape
    torch.Size([8, 3, 15]) torch.Size([30, 3, 10])
    """

    def __init__(self,
                 in_node_feats,
                 in_edge_feats,
                 out_node_feats,
                 out_edge_feats,
                 num_heads,
                 bias=True):

        super().__init__()
        self._num_heads = num_heads
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self.fc_node = nn.Linear(in_node_feats, out_node_feats*num_heads, bias=True)
        self.fc_ni = nn.Linear(in_node_feats, out_edge_feats*num_heads, bias=False)
        self.fc_fij = nn.Linear(in_edge_feats, out_edge_feats*num_heads, bias=False)
        self.fc_nj = nn.Linear(in_node_feats, out_edge_feats*num_heads, bias=False)
        self.attn = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_edge_feats)))
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_edge_feats,)))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        gain = init.calculate_gain('relu')
        init.xavier_normal_(self.fc_node.weight, gain=gain)
        init.xavier_normal_(self.fc_ni.weight, gain=gain)
        init.xavier_normal_(self.fc_fij.weight, gain=gain)
        init.xavier_normal_(self.fc_nj.weight, gain=gain)
        init.xavier_normal_(self.attn, gain=gain)
        init.constant_(self.bias, 0)

    def forward(self, graph, nfeats, efeats, get_attention=False):
        r"""
        Compute new node and edge features.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        nfeats : torch.Tensor
            The input node feature of shape :math:`(N, D_{in})`
            where:
                :math:`D_{in}` is size of input node feature,
                :math:`N` is the number of nodes.
        efeats: torch.Tensor
             The input edge feature of shape :math:`(E, F_{in})`
             where:
                 :math:`F_{in}` is size of input node feature,
                 :math:`E` is the number of edges.
        get_attention : bool, optional
                Whether to return the attention values. Default to False.

        Returns
        -------
        pair of torch.Tensor
            node output features followed by edge output features
            The node output feature of shape :math:`(N, H, D_{out})`
            The edge output feature of shape :math:`(F, H, F_{out})`
            where:
                :math:`H` is the number of heads,
                :math:`D_{out}` is size of output node feature,
                :math:`F_{out}` is size of output edge feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, H, 1)`.
            This is returned only when :attr: `get_attention` is ``True``.
        """

        with graph.local_scope():
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'This is harmful for some applications, '
                               'causing silent performance regression. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue.')

            # TODO allow node src and dst feats
            # calc edge attention
            # same trick way as in dgl.nn.pytorch.GATConv, but also includes edge feats
            # https://github.com/dmlc/dgl/blob/master/python/dgl/nn/pytorch/conv/gatconv.py
            f_ni = self.fc_ni(nfeats)
            f_nj = self.fc_nj(nfeats)
            f_fij = self.fc_fij(efeats)
            graph.srcdata.update({'f_ni': f_ni})
            graph.dstdata.update({'f_nj': f_nj})
            # add ni, nj factors
            graph.apply_edges(fn.u_add_v('f_ni', 'f_nj', 'f_tmp'))
            # add fij to node factor
            f_out = graph.edata.pop('f_tmp') + f_fij
            if self.bias is not None:
                f_out = f_out + self.bias
            f_out = nn.functional.leaky_relu(f_out)
            f_out = f_out.view(-1, self._num_heads, self._out_edge_feats)
            # compute attention factor
            e = (f_out * self.attn).sum(dim=-1).unsqueeze(-1)
            graph.edata['a'] = edge_softmax(graph, e)
            graph.ndata['h_out'] = self.fc_node(nfeats).view(-1, self._num_heads,
                                                             self._out_node_feats)
            # calc weighted sum
            graph.update_all(fn.u_mul_e('h_out', 'a', 'm'),
                             fn.sum('m', 'h_out'))

            h_out = graph.ndata['h_out'].view(-1, self._num_heads, self._out_node_feats)
            if get_attention:
                return h_out, f_out, graph.edata.pop('a')
            else:
                return h_out, f_out
