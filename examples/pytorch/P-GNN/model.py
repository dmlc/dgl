import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class PGNN_layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PGNN_layer, self).__init__()
        self.input_dim = input_dim

        self.linear_hidden_u = nn.Linear(input_dim, output_dim)
        self.linear_hidden_v = nn.Linear(input_dim, output_dim)
        self.linear_out_position = nn.Linear(output_dim, 1)
        self.act = nn.ReLU()

    def forward(self, graph, feature, anchor_eid, dists_max):
        with graph.local_scope():
            u_feat = self.linear_hidden_u(feature)
            v_feat = self.linear_hidden_v(feature)
            graph.srcdata.update({"u_feat": u_feat})
            graph.dstdata.update({"v_feat": v_feat})

            graph.apply_edges(fn.u_mul_e("u_feat", "sp_dist", "u_message"))
            graph.apply_edges(fn.v_add_e("v_feat", "u_message", "message"))

            messages = torch.index_select(
                graph.edata["message"],
                0,
                torch.LongTensor(anchor_eid).to(feature.device),
            )
            messages = messages.reshape(
                dists_max.shape[0], dists_max.shape[1], messages.shape[-1]
            )

            messages = self.act(messages)  # n*m*d

            out_position = self.linear_out_position(messages).squeeze(
                -1
            )  # n*m_out
            out_structure = torch.mean(messages, dim=1)  # n*d

            return out_position, out_structure


class PGNN(nn.Module):
    def __init__(self, input_dim, feature_dim=32, dropout=0.5):
        super(PGNN, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.linear_pre = nn.Linear(input_dim, feature_dim)
        self.conv_first = PGNN_layer(feature_dim, feature_dim)
        self.conv_out = PGNN_layer(feature_dim, feature_dim)

    def forward(self, data):
        x = data["graph"].ndata["feat"]
        graph = data["graph"]
        x = self.linear_pre(x)
        x_position, x = self.conv_first(
            graph, x, data["anchor_eid"], data["dists_max"]
        )

        x = self.dropout(x)
        x_position, x = self.conv_out(
            graph, x, data["anchor_eid"], data["dists_max"]
        )
        x_position = F.normalize(x_position, p=2, dim=-1)
        return x_position
