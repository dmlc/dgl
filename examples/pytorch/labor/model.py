import dgl
import dgl.nn as dglnn
import sklearn.linear_model as lm
import sklearn.metrics as skm
import torch as th
import torch.functional as F
import torch.nn as nn

from dgl.nn import GATv2Conv


class GATv2(nn.Module):
    def __init__(
        self,
        num_layers,
        in_dim,
        num_hidden,
        num_classes,
        heads,
        activation,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
    ):
        super(GATv2, self).__init__()
        self.num_layers = num_layers
        self.gatv2_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gatv2_layers.append(
            GATv2Conv(
                in_dim,
                num_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
                True,
                bias=False,
                share_weights=True,
            )
        )
        # hidden layers
        for l in range(1, num_layers - 1):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gatv2_layers.append(
                GATv2Conv(
                    num_hidden * heads[l - 1],
                    num_hidden,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                    True,
                    bias=False,
                    share_weights=True,
                )
            )
        # output projection
        self.gatv2_layers.append(
            GATv2Conv(
                num_hidden * heads[-2],
                num_classes,
                heads[-1],
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                None,
                True,
                bias=False,
                share_weights=True,
            )
        )

    def forward(self, mfgs, h):
        for l, mfg in enumerate(mfgs):
            h = self.gatv2_layers[l](mfg, h)
            h = h.flatten(1) if l < self.num_layers - 1 else h.mean(1)
        return h


class SAGE(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(
                block,
                h,
                edge_weight=block.edata["edge_weights"]
                if "edge_weights" in block.edata
                else None,
            )
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


class RGAT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        num_etypes,
        num_layers,
        num_heads,
        dropout,
        pred_ntype,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()

        self.convs.append(
            nn.ModuleList(
                [
                    dglnn.GATConv(
                        in_channels,
                        hidden_channels // num_heads,
                        num_heads,
                        allow_zero_in_degree=True,
                    )
                    for _ in range(num_etypes)
                ]
            )
        )
        self.norms.append(nn.BatchNorm1d(hidden_channels))
        self.skips.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(
                nn.ModuleList(
                    [
                        dglnn.GATConv(
                            hidden_channels,
                            hidden_channels // num_heads,
                            num_heads,
                            allow_zero_in_degree=True,
                        )
                        for _ in range(num_etypes)
                    ]
                )
            )
            self.norms.append(nn.BatchNorm1d(hidden_channels))
            self.skips.append(nn.Linear(hidden_channels, hidden_channels))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )
        self.dropout = nn.Dropout(dropout)

        self.hidden_channels = hidden_channels
        self.pred_ntype = pred_ntype
        self.num_etypes = num_etypes

    def forward(self, mfgs, x):
        for i in range(len(mfgs)):
            mfg = mfgs[i]
            x_dst = x[mfg.dst_in_src]
            for data in [mfg.srcdata, mfg.dstdata]:
                for k in list(data.keys()):
                    if k not in ["features", "labels"]:
                        data.pop(k)
            mfg = dgl.block_to_graph(mfg)
            x_skip = self.skips[i](x_dst)
            for j in range(self.num_etypes):
                subg = mfg.edge_subgraph(
                    mfg.edata["etype"] == j, relabel_nodes=False
                )
                x_skip += self.convs[i][j](subg, (x, x_dst)).view(
                    -1, self.hidden_channels
                )
            x = self.norms[i](x_skip)
            x = th.nn.functional.elu(x)
            x = self.dropout(x)
        return self.mlp(x)
