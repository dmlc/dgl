"""Torch modules for xianyu graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn

from .... import function as fn
from ..softmax import edge_softmax

class XianYuGatConv(nn.Module):
    r"""
    This is an DGL implementation of the core attention mechanism in
     `XianYu Graph <https://arxiv.org/abs/1908.10679>`.

    ..math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{e, j} W_j^{(l)}h_j^{(l)}

    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} & = \mathrm{softmax_i} (e_{ij}^{l})

        e_{ij}^{l} & = \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)
    """

    def __init__(self,
                 src_dim,
                 edge_dim,
                 dst_dim,
                 out_dim
                 ):
        super(XianYuGatConv, self).__init__()
        self._in_src_dim = src_dim
        self._in_edge_dim = edge_dim
        self._in_dst_dim = dst_dim
        self._out_dim = out_dim

        self.fc_src_hid = nn.Linear((self._in_src_dim + self._in_edge_dim), self._out_dim, bias=False)
        self.fc_dst_hid = nn.Linear(self._in_dst_dim, self._out_dim, bias=False)

    def forward(self, graph, src_feats, edge_feats, dst_feats):
        r"""
        Input a graph, either homograph or a bipartities. Output attention-based values to the destination

        :param graph:
        :param src_feats:
        :param edge_feats:
        :param dst_feats:
        :return:
        """
        graph = graph.local_var()

        # assign features to graph
        graph.srcdata['src_feat'] = src_feats
        graph.edata['edge_feat'] = edge_feats
        graph.dstdata['dst_feat'] = self.fc_dst_hid(dst_feats)

        # Step 0. prepare source and edge dimensionality
        graph.apply_edges(lambda edges: {'h_ue': th.cat([edges.src['src_feat'],
                                                         edges.data['edge_feat']], dim=-1)})
        graph.edata['h_ue_attn'] = self.fc_src_hid(graph.edata['h_ue'])

        # Step 1. dot product
        graph.apply_edges(fn.e_dot_v('h_ue_attn', 'dst_feat', 'a'))

        # Step 2. edge softmax to compute attention scores
        graph.edata['sa'] = edge_softmax(graph, graph.edata['a'])

        # Step 3. Broadcast softmax value to each edge, and then attention is done
        graph.edata['attn'] = graph.edata['h_ue_attn'] * graph.edata['sa'].unsqueeze(dim=0).T

        # Step 4. Aggregate attention to dst,user nodes, so formula 7 is done
        graph.update_all(fn.copy_e('attn', 'm'), fn.sum('m', 'agg_u'))

        # output results to the destination nodes
        rst = graph.dstdata['agg_u']

        return rst
