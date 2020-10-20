import math

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from torch.utils.checkpoint import checkpoint


class MWEConv(nn.Module):
    def __init__(self, in_feats, out_feats, activation, bias=True, num_channels=8, aggr_mode="sum"):
        super(MWEConv, self).__init__()
        self.num_channels = num_channels
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats, num_channels))

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
                g.ndata["feat_" + str(c)] = torch.mm(node_state_c, self.weight[:, :, c])
            else:
                g.ndata["feat_" + str(c)] = node_state_c
            g.update_all(
                fn.src_mul_edge("feat_" + str(c), "feat_" + str(c), "m"), fn.sum("m", "feat_" + str(c) + "_new")
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
    def __init__(self, n_input, n_hidden, n_output, n_layers, activation, dropout, aggr_mode="sum", device="cpu"):
        super(MWE_GCN, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.layers = nn.ModuleList()

        self.layers.append(MWEConv(n_input, n_hidden, activation=activation, aggr_mode=aggr_mode))
        for i in range(n_layers - 1):
            self.layers.append(MWEConv(n_hidden, n_hidden, activation=activation, aggr_mode=aggr_mode))

        self.pred_out = nn.Linear(n_hidden, n_output)
        self.device = device

    def forward(self, g, node_state=None):
        node_state = torch.ones(g.number_of_nodes(), 1).float().to(self.device)

        for layer in self.layers:
            node_state = F.dropout(node_state, p=self.dropout, training=self.training)
            node_state = layer(g, node_state)
            node_state = self.activation(node_state)

        out = self.pred_out(node_state)
        return out


class MWE_DGCN(nn.Module):
    def __init__(
        self, n_input, n_hidden, n_output, n_layers, activation, dropout, residual=False, aggr_mode="sum", device="cpu"
    ):
        super(MWE_DGCN, self).__init__()
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout
        self.residual = residual

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        self.layers.append(MWEConv(n_input, n_hidden, activation=activation, aggr_mode=aggr_mode))

        for i in range(n_layers - 1):
            self.layers.append(MWEConv(n_hidden, n_hidden, activation=activation, aggr_mode=aggr_mode))

        for i in range(n_layers):
            self.layer_norms.append(nn.LayerNorm(n_hidden, elementwise_affine=True))

        self.pred_out = nn.Linear(n_hidden, n_output)
        self.device = device

    def forward(self, g, node_state=None):
        node_state = torch.ones(g.number_of_nodes(), 1).float().to(self.device)

        node_state = self.layers[0](g, node_state)

        for layer in range(1, self.n_layers):
            node_state_new = self.layer_norms[layer - 1](node_state)
            node_state_new = self.activation(node_state_new)
            node_state_new = F.dropout(node_state_new, p=self.dropout, training=self.training)

            if self.residual == "true":
                node_state = node_state + self.layers[layer](g, node_state_new)
            else:
                node_state = self.layers[layer](g, node_state_new)

        node_state = self.layer_norms[self.n_layers - 1](node_state)
        node_state = self.activation(node_state)
        node_state = F.dropout(node_state, p=self.dropout, training=self.training)

        out = self.pred_out(node_state)

        return out


class GATConv(nn.Module):
    def __init__(
        self,
        node_feats,
        edge_feats,
        out_feats,
        n_heads=1,
        attn_dropout=0.0,
        activation=None,
        allow_zero_in_degree=False,
    ):
        super(GATConv, self).__init__()
        self._num_heads = n_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(node_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.src_fc = nn.Linear(self._in_src_feats, out_feats * n_heads, bias=False)
        self.dst_fc = nn.Linear(self._in_src_feats, out_feats * n_heads)
        self.attn_src_fc = nn.Linear(self._in_src_feats, n_heads, bias=False)
        self.attn_edge_fc = nn.Linear(edge_feats, n_heads, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.src_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.dst_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat_src, feat_edge):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            if graph.is_block:
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
            else:
                feat_dst = feat_src

            # NOTE: We modified the method in GAT paper by removing the
            # "leaky_relu" in calculating attention, thus reducing the amount
            # of calculation:
            # softmax(a^T [Wh_i || Wh_j]) = softmax(a^T Wh_i)
            # where j indicates the destination node of an edge.
            feat_src_fc = self.src_fc(feat_src).view(-1, self._num_heads, self._out_feats)
            feat_dst_fc = self.dst_fc(feat_dst).view(-1, self._num_heads, self._out_feats)
            attn_src = self.attn_src_fc(feat_src).view(-1, self._num_heads, 1)
            attn_edge = self.attn_edge_fc(feat_edge).view(-1, self._num_heads, 1)

            graph.srcdata.update({"feat_src_fc": feat_src_fc, "attn_src": attn_src})
            graph.edata.update({"attn_edge": attn_edge})

            graph.apply_edges(fn.copy_u("attn_src", "attn_node"))
            e = graph.edata["attn_node"] + graph.edata["attn_edge"]
            graph.edata["a"] = self.attn_dropout(edge_softmax(graph, e))
            graph.update_all(fn.u_mul_e("feat_src_fc", "a", "m"), fn.sum("m", "feat_src_fc"))

            rst = graph.dstdata["feat_src_fc"] + feat_dst_fc

            if self.activation is not None:
                rst = self.activation(rst)

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
        attn_dropout,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.node_encoder = nn.Linear(node_feats, n_hidden)
        self.edge_encoder = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else n_hidden
            out_hidden = n_hidden

            self.edge_encoder.append(nn.Linear(edge_feats, edge_emb))
            self.convs.append(GATConv(in_hidden, edge_emb, out_hidden, n_heads=n_heads, attn_dropout=attn_dropout))
            self.norms.append(nn.BatchNorm1d(n_heads * out_hidden))

        self.pred_linear = nn.Linear(n_heads * n_hidden, n_classes)

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

        h_last = None

        for i in range(self.n_layers):
            efeat = subgraphs[i].edata["feat"]
            efeat_emb = self.edge_encoder[i](efeat)
            efeat_emb = F.relu(efeat_emb, inplace=True)

            h = self.convs[i](subgraphs[i], h, efeat_emb).flatten(1, -1)

            if h_last is not None:
                h += h_last[: h.shape[0], :]

            h_last = h

            if self.training:
                h = checkpoint(self.norms[i], h)
            else:
                h = self.norms[i](h)
            h = self.activation(h, inplace=True)
            h = self.dropout(h)

        if self.training:
            h = checkpoint(self.pred_linear, h)
        else:
            h = self.pred_linear(h)

        return h
