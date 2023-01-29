import torch
import torch.nn as nn

from dgl.nn.pytorch import (
    AGNNConv,
    APPNPConv,
    ChebConv,
    GATConv,
    GINConv,
    GraphConv,
    SAGEConv,
    SGConv,
    TAGConv,
)


class GCN(nn.Module):
    def __init__(
        self, g, in_feats, n_classes, n_hidden, n_layers, activation, dropout
    ):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, activation=activation)
            )
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h


class GAT(nn.Module):
    def __init__(
        self,
        g,
        in_dim,
        num_classes,
        num_hidden,
        num_layers,
        heads,
        activation,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
    ):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(
            GATConv(
                in_dim,
                num_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
            )
        )
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(
                GATConv(
                    num_hidden * heads[l - 1],
                    num_hidden,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                )
            )
        # output projection
        self.gat_layers.append(
            GATConv(
                num_hidden * heads[-2],
                num_classes,
                heads[-1],
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                None,
            )
        )

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits


class GraphSAGE(nn.Module):
    def __init__(
        self,
        g,
        in_feats,
        n_classes,
        n_hidden,
        n_layers,
        activation,
        dropout,
        aggregator_type,
    ):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g

        # input layer
        self.layers.append(
            SAGEConv(
                in_feats,
                n_hidden,
                aggregator_type,
                feat_drop=dropout,
                activation=activation,
            )
        )
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                SAGEConv(
                    n_hidden,
                    n_hidden,
                    aggregator_type,
                    feat_drop=dropout,
                    activation=activation,
                )
            )
        # output layer
        self.layers.append(
            SAGEConv(
                n_hidden,
                n_classes,
                aggregator_type,
                feat_drop=dropout,
                activation=None,
            )
        )  # activation None

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(self.g, h)
        return h


class APPNP(nn.Module):
    def __init__(
        self,
        g,
        in_feats,
        n_classes,
        n_hidden,
        n_layers,
        activation,
        feat_drop,
        edge_drop,
        alpha,
        k,
    ):
        super(APPNP, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, n_hidden))
        # hidden layers
        for i in range(1, n_layers):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        # output layer
        self.layers.append(nn.Linear(n_hidden, n_classes))
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, features):
        # prediction step
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        h = self.propagate(self.g, h)
        return h


class TAGCN(nn.Module):
    def __init__(
        self, g, in_feats, n_classes, n_hidden, n_layers, activation, dropout
    ):
        super(TAGCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(TAGConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                TAGConv(n_hidden, n_hidden, activation=activation)
            )
        # output layer
        self.layers.append(TAGConv(n_hidden, n_classes))  # activation=None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h


class AGNN(nn.Module):
    def __init__(
        self,
        g,
        in_feats,
        n_classes,
        n_hidden,
        n_layers,
        init_beta,
        learn_beta,
        dropout,
    ):
        super(AGNN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList(
            [AGNNConv(init_beta, learn_beta) for _ in range(n_layers)]
        )
        self.proj = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_feats, n_hidden), nn.ReLU()
        )
        self.cls = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(n_hidden, n_classes)
        )

    def forward(self, features):
        h = self.proj(features)
        for layer in self.layers:
            h = layer(self.g, h)
        return self.cls(h)


class SGC(nn.Module):
    def __init__(self, g, in_feats, n_classes, n_hidden, k, bias):
        super(SGC, self).__init__()
        self.g = g
        self.net = SGConv(in_feats, n_classes, k=k, cached=True, bias=bias)

    def forward(self, features):
        return self.net(self.g, features)


class GIN(nn.Module):
    def __init__(
        self, g, in_feats, n_classes, n_hidden, n_layers, init_eps, learn_eps
    ):
        super(GIN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(
            GINConv(
                nn.Sequential(
                    nn.Dropout(0.6),
                    nn.Linear(in_feats, n_hidden),
                    nn.ReLU(),
                ),
                "mean",
                init_eps,
                learn_eps,
            )
        )
        for i in range(n_layers - 1):
            self.layers.append(
                GINConv(
                    nn.Sequential(
                        nn.Dropout(0.6),
                        nn.Linear(n_hidden, n_hidden),
                        nn.ReLU(),
                    ),
                    "mean",
                    init_eps,
                    learn_eps,
                )
            )
        self.layers.append(
            GINConv(
                nn.Sequential(
                    nn.Dropout(0.6),
                    nn.Linear(n_hidden, n_classes),
                ),
                "mean",
                init_eps,
                learn_eps,
            )
        )

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(self.g, h)
        return h


class ChebNet(nn.Module):
    def __init__(self, g, in_feats, n_classes, n_hidden, n_layers, k, bias):
        super(ChebNet, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(ChebConv(in_feats, n_hidden, k, bias=bias))
        for _ in range(n_layers - 1):
            self.layers.append(ChebConv(n_hidden, n_hidden, k, bias=bias))

        self.layers.append(ChebConv(n_hidden, n_classes, k, bias=bias))

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(self.g, h, [2])
        return h
