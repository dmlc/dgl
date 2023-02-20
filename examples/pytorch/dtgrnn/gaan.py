import dgl
import dgl.function as fn
import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
from dgl.base import DGLError
from dgl.nn.functional import edge_softmax


class WeightedGATConv(dglnn.GATConv):
    """
    This model inherit from dgl GATConv for traffic prediction task,
    it add edge weight when aggregating the node feature.
    """

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(
                        -1, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc(h_dst).view(
                        -1, self._num_heads, self._out_feats
                    )
                else:
                    feat_src = self.fc_src(h_src).view(
                        -1, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc_dst(h_dst).view(
                        -1, self._num_heads, self._out_feats
                    )
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats
                )
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))
            # compute softmax
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            # compute weighted attention
            graph.edata["a"] = (
                graph.edata["a"].permute(1, 2, 0) * graph.edata["weight"]
            ).permute(2, 0, 1)
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(
                    h_dst.shape[0], -1, self._out_feats
                )
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst


class GatedGAT(nn.Module):
    """Gated Graph Attention module, it is a general purpose
    graph attention module proposed in paper GaAN. The paper use
    it for traffic prediction task
    Parameter
    ==========
    in_feats : int
        number of input feature

    out_feats : int
        number of output feature

    map_feats : int
        intermediate feature size for gate computation

    num_heads : int
        number of head for multihead attention
    """

    def __init__(self, in_feats, out_feats, map_feats, num_heads):
        super(GatedGAT, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.map_feats = map_feats
        self.num_heads = num_heads
        self.gatlayer = WeightedGATConv(
            self.in_feats, self.out_feats, self.num_heads
        )
        self.gate_fn = nn.Linear(
            2 * self.in_feats + self.map_feats, self.num_heads
        )
        self.gate_m = nn.Linear(self.in_feats, self.map_feats)
        self.merger_layer = nn.Linear(
            self.in_feats + self.out_feats, self.out_feats
        )

    def forward(self, g, x):
        with g.local_scope():
            g.ndata["x"] = x
            g.ndata["z"] = self.gate_m(x)
            g.update_all(fn.copy_u("x", "x"), fn.mean("x", "mean_z"))
            g.update_all(fn.copy_u("z", "z"), fn.max("z", "max_z"))
            nft = torch.cat(
                [g.ndata["x"], g.ndata["max_z"], g.ndata["mean_z"]], dim=1
            )
            gate = self.gate_fn(nft).sigmoid()
            attn_out = self.gatlayer(g, x)
            node_num = g.num_nodes()
            gated_out = (
                (gate.view(-1) * attn_out.view(-1, self.out_feats).T).T
            ).view(node_num, self.num_heads, self.out_feats)
            gated_out = gated_out.mean(1)
            merge = self.merger_layer(torch.cat([x, gated_out], dim=1))
            return merge
