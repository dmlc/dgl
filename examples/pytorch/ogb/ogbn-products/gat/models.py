# update time: 2020.11.02 17:33
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair


class GATConv(nn.Module):
    def __init__(
        self,
        node_feats,
        edge_feats,
        out_feats,
        n_heads=1,
        attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        residual=True,
        activation=None,
        use_attn_dst=True,
        allow_zero_in_degree=True,
        use_symmetric_norm=False,
    ):
        super(GATConv, self).__init__()
        self._n_heads = n_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(node_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._use_symmetric_norm = use_symmetric_norm

        # feat fc
        self.src_fc = nn.Linear(
            self._in_src_feats, out_feats * n_heads, bias=False
        )
        if residual:
            self.dst_fc = nn.Linear(self._in_src_feats, out_feats * n_heads)
            self.bias = None
        else:
            self.dst_fc = None
            self.bias = nn.Parameter(out_feats * n_heads)

        # attn fc
        self.attn_src_fc = nn.Linear(self._in_src_feats, n_heads, bias=False)
        if use_attn_dst:
            self.attn_dst_fc = nn.Linear(
                self._in_src_feats, n_heads, bias=False
            )
        else:
            self.attn_dst_fc = None
        if edge_feats > 0:
            self.attn_edge_fc = nn.Linear(edge_feats, n_heads, bias=False)
        else:
            self.attn_edge_fc = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope, inplace=True)
        self.activation = activation

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.src_fc.weight, gain=gain)
        if self.dst_fc is not None:
            nn.init.xavier_normal_(self.dst_fc.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_src_fc.weight, gain=gain)
        if self.attn_dst_fc is not None:
            nn.init.xavier_normal_(self.attn_dst_fc.weight, gain=gain)
        if self.attn_edge_fc is not None:
            nn.init.xavier_normal_(self.attn_edge_fc.weight, gain=gain)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat_src, feat_edge=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            if graph.is_block:
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
            else:
                feat_dst = feat_src

            if self._use_symmetric_norm:
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            feat_src_fc = self.src_fc(feat_src).view(
                -1, self._n_heads, self._out_feats
            )
            feat_dst_fc = self.dst_fc(feat_dst).view(
                -1, self._n_heads, self._out_feats
            )
            attn_src = self.attn_src_fc(feat_src).view(-1, self._n_heads, 1)

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
            graph.srcdata.update(
                {"feat_src_fc": feat_src_fc, "attn_src": attn_src}
            )

            if self.attn_dst_fc is not None:
                attn_dst = self.attn_dst_fc(feat_dst).view(-1, self._n_heads, 1)
                graph.dstdata.update({"attn_dst": attn_dst})
                graph.apply_edges(
                    fn.u_add_v("attn_src", "attn_dst", "attn_node")
                )
            else:
                graph.apply_edges(fn.copy_u("attn_src", "attn_node"))

            e = graph.edata["attn_node"]
            if feat_edge is not None:
                attn_edge = self.attn_edge_fc(feat_edge).view(
                    -1, self._n_heads, 1
                )
                graph.edata.update({"attn_edge": attn_edge})
                e += graph.edata["attn_edge"]
            e = self.leaky_relu(e)

            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.num_edges(), device=e.device)
                bound = int(graph.num_edges() * self.edge_drop)
                eids = perm[bound:]
                graph.edata["a"] = torch.zeros_like(e)
                graph.edata["a"][eids] = self.attn_drop(
                    edge_softmax(graph, e[eids], eids=eids)
                )
            else:
                graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            graph.update_all(
                fn.u_mul_e("feat_src_fc", "a", "m"), fn.sum("m", "feat_src_fc")
            )
            rst = graph.dstdata["feat_src_fc"]

            if self._use_symmetric_norm:
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, 0.5)
                shp = norm.shape + (1,) * (feat_dst.dim())
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            # residual
            if self.dst_fc is not None:
                rst += feat_dst_fc
            else:
                rst += self.bias

            # activation
            if self.activation is not None:
                rst = self.activation(rst, inplace=True)

            return rst


class GAT(nn.Module):
    def __init__(
        self,
        node_feats,
        edge_feats,
        n_classes,
        n_layers,
        n_heads,
        n_hidden,
        edge_emb,
        activation,
        dropout,
        input_drop,
        attn_drop,
        edge_drop,
        use_attn_dst=True,
        allow_zero_in_degree=False,
        residual=False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.node_encoder = nn.Linear(node_feats, n_hidden)
        if edge_emb > 0:
            self.edge_encoder = nn.ModuleList()
        else:
            self.edge_encoder = None

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else node_feats
            out_hidden = n_hidden

            if self.edge_encoder is not None:
                self.edge_encoder.append(nn.Linear(edge_feats, edge_emb))
            self.convs.append(
                GATConv(
                    in_hidden,
                    edge_emb,
                    out_hidden,
                    n_heads=n_heads,
                    attn_drop=attn_drop,
                    edge_drop=edge_drop,
                    use_attn_dst=use_attn_dst,
                    allow_zero_in_degree=allow_zero_in_degree,
                )
            )
            self.norms.append(nn.BatchNorm1d(n_heads * out_hidden))

        self.pred_linear = nn.Linear(n_heads * n_hidden, n_classes)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.residual = residual

    def forward(self, g, inference=False):
        if not isinstance(g, list):
            subgraphs = [g] * self.n_layers
        else:
            subgraphs = g

        h = subgraphs[0].srcdata["feat"]
        h = self.input_drop(h)

        h_last = None

        for i in range(self.n_layers):
            if self.edge_encoder is not None:
                efeat = subgraphs[i].edata["feat"]
                efeat_emb = self.edge_encoder[i](efeat)
                efeat_emb = F.relu(efeat_emb, inplace=True)
            else:
                efeat_emb = None

            h = self.convs[i](subgraphs[i], h, efeat_emb).flatten(1, -1)

            if self.residual and h_last is not None:
                h += h_last[: h.shape[0], :]

            h_last = h

            h = self.norms[i](h)
            h = self.activation(h, inplace=True)
            h = self.dropout(h)

            if inference:
                torch.cuda.empty_cache()

        h = self.pred_linear(h)

        return h


class MLP(nn.Module):
    def __init__(
        self,
        in_feats,
        n_classes,
        n_layers,
        n_hidden,
        activation,
        dropout=0.0,
        input_drop=0.0,
        residual=False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes

            self.linears.append(nn.Linear(in_hidden, out_hidden))

            if i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_hidden))

        self.activation = activation
        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, h):
        h = self.input_drop(h)

        h_last = None

        for i in range(self.n_layers):
            h = self.linears[i](h)

            if self.residual and 0 < i < self.n_layers - 1:
                h += h_last

            h_last = h

            if i < self.n_layers - 1:
                h = self.norms[i](h)
                h = self.activation(h, inplace=True)
                h = self.dropout(h)

        return h
