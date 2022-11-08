import math

import torch
import torch.nn.functional as F
from dgl.nn import GraphConv, SortPooling
from torch.nn import Conv1d, Embedding, Linear, MaxPool1d, ModuleList


class NGNN_GCNConv(torch.nn.Module):
    def __init__(
        self, input_channels, hidden_channels, output_channels, num_layers
    ):
        super(NGNN_GCNConv, self).__init__()
        self.conv = GraphConv(input_channels, hidden_channels)
        self.fc = Linear(hidden_channels, hidden_channels)
        self.fc2 = Linear(hidden_channels, output_channels)
        self.num_layers = num_layers

    def reset_parameters(self):
        self.conv.reset_parameters()
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=gain)
        for bias in [self.fc.bias, self.fc2.bias]:
            stdv = 1.0 / math.sqrt(bias.size(0))
            bias.data.uniform_(-stdv, stdv)

    def forward(self, g, x, edge_weight=None):
        x = self.conv(g, x, edge_weight)
        if self.num_layers == 2:
            x = F.relu(x)
            x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


# An end-to-end deep learning architecture for graph classification, AAAI-18.
class DGCNN(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_layers,
        max_z,
        k,
        feature_dim=0,
        GNN=GraphConv,
        NGNN=NGNN_GCNConv,
        dropout=0.0,
        ngnn_type="all",
        num_ngnn_layers=1,
    ):
        super(DGCNN, self).__init__()

        self.feature_dim = feature_dim
        self.dropout = dropout

        self.k = k
        self.sort_pool = SortPooling(k=self.k)

        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels + self.feature_dim

        self.num_ngnn_layers = num_ngnn_layers
        if ngnn_type in ["input", "all"]:
            self.convs.append(
                NGNN(
                    initial_channels,
                    hidden_channels,
                    hidden_channels,
                    self.num_ngnn_layers,
                )
            )
        else:
            self.convs.append(GNN(initial_channels, hidden_channels))

        if ngnn_type in ["hidden", "all"]:
            for _ in range(0, num_layers - 1):
                self.convs.append(
                    NGNN(
                        hidden_channels,
                        hidden_channels,
                        hidden_channels,
                        self.num_ngnn_layers,
                    )
                )
        else:
            for _ in range(0, num_layers - 1):
                self.convs.append(GNN(hidden_channels, hidden_channels))

        if ngnn_type in ["output", "all"]:
            self.convs.append(
                NGNN(hidden_channels, hidden_channels, 1, self.num_ngnn_layers)
            )
        else:
            self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(
            conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1
        )
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, g, z, x=None, edge_weight=None):
        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        xs = [x]

        for conv in self.convs:
            xs += [
                F.dropout(
                    torch.tanh(conv(g, xs[-1], edge_weight=edge_weight)),
                    p=self.dropout,
                    training=self.training,
                )
            ]
        x = torch.cat(xs[1:], dim=-1)

        # global pooling
        x = self.sort_pool(g, x)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
