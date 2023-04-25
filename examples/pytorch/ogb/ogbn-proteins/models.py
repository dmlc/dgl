import math
from functools import partial

import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl._ffi.base import DGLError
from dgl.base import ALL
from dgl.nn.pytorch.utils import Identity
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from torch.nn import init
from torch.utils.checkpoint import checkpoint


class MWEConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        activation,
        bias=True,
        num_channels=8,
        aggr_mode="sum",
    ):
        super(MWEConv, self).__init__()
        self.num_channels = num_channels
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = nn.Parameter(
            torch.Tensor(in_feats, out_feats, num_channels)
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats, num_channels))
        else:
            self.bias = None
        self.reset_parameters()
        self.activation = activation

        if aggr_mode == "concat":
            self.aggr_mode = "concat"
            self.final = nn.Linear(out_feats * self.num_channels, out_feats)
        elif aggr_mode == "sum":
            self.aggr_mode = "sum"
            self.final = nn.Linear(out_feats, out_feats)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            stdv = 1.0 / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, node_state_prev):
        node_state = node_state_prev

        # if self.dropout:
        #     node_states = self.dropout(node_state)

        g = g.local_var()

        new_node_states = []

        ## perform weighted convolution for every channel of edge weight
        for c in range(self.num_channels):
            node_state_c = node_state
            if self._out_feats < self._in_feats:
                g.ndata["feat_" + str(c)] = torch.mm(
                    node_state_c, self.weight[:, :, c]
                )
            else:
                g.ndata["feat_" + str(c)] = node_state_c
            g.update_all(
                fn.u_mul_e("feat_" + str(c), "feat_" + str(c), "m"),
                fn.sum("m", "feat_" + str(c) + "_new"),
            )
            node_state_c = g.ndata.pop("feat_" + str(c) + "_new")
            if self._out_feats >= self._in_feats:
                node_state_c = torch.mm(node_state_c, self.weight[:, :, c])
            if self.bias is not None:
                node_state_c = node_state_c + self.bias[:, c]
            node_state_c = self.activation(node_state_c)
            new_node_states.append(node_state_c)
        if self.aggr_mode == "sum":
            node_states = torch.stack(new_node_states, dim=1).sum(1)
        elif self.aggr_mode == "concat":
            node_states = torch.cat(new_node_states, dim=1)

        node_states = self.final(node_states)

        return node_states


class MWE_GCN(nn.Module):
    def __init__(
        self,
        n_input,
        n_hidden,
        n_output,
        n_layers,
        activation,
        dropout,
        aggr_mode="sum",
        device="cpu",
    ):
        super(MWE_GCN, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.layers = nn.ModuleList()

        self.layers.append(
            MWEConv(
                n_input, n_hidden, activation=activation, aggr_mode=aggr_mode
            )
        )
        for i in range(n_layers - 1):
            self.layers.append(
                MWEConv(
                    n_hidden,
                    n_hidden,
                    activation=activation,
                    aggr_mode=aggr_mode,
                )
            )

        self.pred_out = nn.Linear(n_hidden, n_output)
        self.device = device

    def forward(self, g, node_state=None):
        node_state = torch.ones(g.num_nodes(), 1).float().to(self.device)

        for layer in self.layers:
            node_state = F.dropout(
                node_state, p=self.dropout, training=self.training
            )
            node_state = layer(g, node_state)
            node_state = self.activation(node_state)

        out = self.pred_out(node_state)
        return out


class MWE_DGCN(nn.Module):
    def __init__(
        self,
        n_input,
        n_hidden,
        n_output,
        n_layers,
        activation,
        dropout,
        residual=False,
        aggr_mode="sum",
        device="cpu",
    ):
        super(MWE_DGCN, self).__init__()
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout
        self.residual = residual

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        self.layers.append(
            MWEConv(
                n_input, n_hidden, activation=activation, aggr_mode=aggr_mode
            )
        )

        for i in range(n_layers - 1):
            self.layers.append(
                MWEConv(
                    n_hidden,
                    n_hidden,
                    activation=activation,
                    aggr_mode=aggr_mode,
                )
            )

        for i in range(n_layers):
            self.layer_norms.append(
                nn.LayerNorm(n_hidden, elementwise_affine=True)
            )

        self.pred_out = nn.Linear(n_hidden, n_output)
        self.device = device

    def forward(self, g, node_state=None):
        node_state = torch.ones(g.num_nodes(), 1).float().to(self.device)

        node_state = self.layers[0](g, node_state)

        for layer in range(1, self.n_layers):
            node_state_new = self.layer_norms[layer - 1](node_state)
            node_state_new = self.activation(node_state_new)
            node_state_new = F.dropout(
                node_state_new, p=self.dropout, training=self.training
            )

            if self.residual == "true":
                node_state = node_state + self.layers[layer](g, node_state_new)
            else:
                node_state = self.layers[layer](g, node_state_new)

        node_state = self.layer_norms[self.n_layers - 1](node_state)
        node_state = self.activation(node_state)
        node_state = F.dropout(
            node_state, p=self.dropout, training=self.training
        )

        out = self.pred_out(node_state)

        return out


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
                degs = graph.srcdata["deg"]
                # degs = graph.out_degrees().float().clamp(min=1)
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
                degs = graph.dstdata["deg"]
                # degs = graph.in_degrees().float().clamp(min=1)
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

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else n_hidden
            out_hidden = n_hidden
            # bias = i == n_layers - 1

            if edge_emb > 0:
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
                    use_symmetric_norm=False,
                )
            )
            self.norms.append(nn.BatchNorm1d(n_heads * out_hidden))

        self.pred_linear = nn.Linear(n_heads * n_hidden, n_classes)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, g):
        if not isinstance(g, list):
            subgraphs = [g] * self.n_layers
        else:
            subgraphs = g

        h = subgraphs[0].srcdata["feat"]
        h = self.node_encoder(h)
        h = F.relu(h, inplace=True)
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

            if h_last is not None:
                h += h_last[: h.shape[0], :]

            h_last = h

            h = self.norms[i](h)
            h = self.activation(h, inplace=True)
            h = self.dropout(h)

        h = self.pred_linear(h)

        return h
