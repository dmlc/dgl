"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
from torch import nn

from .... import function as fn
from ..softmax import edge_softmax
from ....utils import expand_as_pair


class DotGatConv(nn.Module):
    r"""Apply dot product version of self attention in GCN.

        .. math::
            h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i, j} h_j^{(l)}

        where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and node :math:`j`:

        .. math::
            \alpha_{i, j} = \mathrm{softmax_i}(e_{ij}^{l})

            e_{ij}^{l} = ({W_i^{(l)} h_i^{(l)}})^T \cdot {W_j^{(l)} h_j^{(l)}}

        where :math:`W_i` and :math:`W_j` transform node :math:`i`'s and node :math:`j`'s
        features into the same dimension, so that when compute note features' similarity,
        we can use dot-product.
    """

    def __init__(self,
                 in_feats,
                 out_feats
                 ):
        super(DotGatConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, self._out_feats, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, self._out_feats, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, self._out_feats, bias=False)

    def forward(self, graph, feat):
        r"""Apply dot product version of self attention in GCN.

        Parameters
        ----------
        graph: DGLGraph or bi_partities graph
            The graph
        feat: torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}` is size
            of output feature.
        """

        graph = graph.local_var()

        # check if feat is a tuple
        if isinstance(feat, tuple):
            h_src = feat[0]
            h_dst = feat[1]
            feat_src = self.fc_src(h_src)
            feat_dst = self.fc_dst(h_dst)
        else:
            h_src = feat
            feat_src = feat_dst = self.fc(h_src)

        # Assign features to nodes
        graph.srcdata.update({'ft': feat_src})
        graph.dstdata.update({'ft': feat_dst})

        # Step 1. dot product
        graph.apply_edges(fn.u_dot_v('ft', 'ft', 'a'))

        # Step 2. edge softmax to compute attention scores
        graph.edata['sa'] = edge_softmax(graph, graph.edata['a'])

        # Step 3. Broadcast softmax value to each edge, and then attention is done
        graph.apply_edges(lambda edges: {'attn': edges.src['ft'] * \
                                                 edges.data['sa'].unsqueeze(dim=0).T})

        # Step 4. Aggregate attention to dst,user nodes, so formula 7 is done
        graph.update_all(fn.copy_e('attn', 'm'), fn.sum('m', 'agg_u'))

        # output results to the destination nodes
        rst = graph.dstdata['agg_u']

        return rst
