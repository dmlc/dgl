"""
How Powerful are Graph Neural Networks
https://arxiv.org/abs/1810.00826
https://openreview.net/forum?id=ryGs6iA5Km
Author's implementation: https://github.com/weihua916/powerful-gnns
"""

import mxnet as mx

from dgl.nn.mxnet.conv import GINConv
from dgl.nn.mxnet.glob import AvgPooling, MaxPooling, SumPooling
from mxnet import gluon, nd
from mxnet.gluon import nn


class ApplyNodeFunc(nn.Block):
    """Update the node feature hv with MLP, BN and ReLU."""

    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        with self.name_scope():
            self.mlp = mlp
            self.bn = nn.BatchNorm(in_channels=self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = nd.relu(h)
        return h


class MLP(nn.Block):
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction

        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.output_dim = output_dim

        with self.name_scope():
            if num_layers < 1:
                raise ValueError("number of layers should be positive!")
            elif num_layers == 1:
                # Linear model
                self.linear = nn.Dense(output_dim, in_units=input_dim)
            else:
                self.linear_or_not = False
                self.linears = nn.Sequential()
                self.batch_norms = nn.Sequential()

                self.linears.add(nn.Dense(hidden_dim, in_units=input_dim))
                for layer in range(num_layers - 2):
                    self.linears.add(nn.Dense(hidden_dim, in_units=hidden_dim))
                self.linears.add(nn.Dense(output_dim, in_units=hidden_dim))

                for layer in range(num_layers - 1):
                    self.batch_norms.add(nn.BatchNorm(in_channels=hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for i in range(self.num_layers - 1):
                h = nd.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GIN(nn.Block):
    """GIN model"""

    def __init__(
        self,
        num_layers,
        num_mlp_layers,
        input_dim,
        hidden_dim,
        output_dim,
        final_dropout,
        learn_eps,
        graph_pooling_type,
        neighbor_pooling_type,
    ):
        """model parameters setting

        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)

        """
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        with self.name_scope():
            # List of MLPs
            self.ginlayers = nn.Sequential()
            self.batch_norms = nn.Sequential()

            for i in range(self.num_layers - 1):
                if i == 0:
                    mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
                else:
                    mlp = MLP(
                        num_mlp_layers, hidden_dim, hidden_dim, hidden_dim
                    )

                self.ginlayers.add(
                    GINConv(
                        ApplyNodeFunc(mlp),
                        neighbor_pooling_type,
                        0,
                        self.learn_eps,
                    )
                )
                self.batch_norms.add(nn.BatchNorm(in_channels=hidden_dim))

            self.linears_prediction = nn.Sequential()

            for i in range(num_layers):
                if i == 0:
                    self.linears_prediction.add(
                        nn.Dense(output_dim, in_units=input_dim)
                    )
                else:
                    self.linears_prediction.add(
                        nn.Dense(output_dim, in_units=hidden_dim)
                    )

            self.drop = nn.Dropout(final_dropout)

            if graph_pooling_type == "sum":
                self.pool = SumPooling()
            elif graph_pooling_type == "mean":
                self.pool = AvgPooling()
            elif graph_pooling_type == "max":
                self.pool = MaxPooling()
            else:
                raise NotImplementedError

    def forward(self, g, h):
        hidden_rep = [h]

        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = nd.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0
        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer = score_over_layer + self.drop(
                self.linears_prediction[i](pooled_h)
            )

        return score_over_layer
