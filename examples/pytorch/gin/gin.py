"""
How Powerful are Graph Neural Networks
https://arxiv.org/abs/1810.00826
https://openreview.net/forum?id=ryGs6iA5Km
Author's implementation: https://github.com/weihua916/powerful-gnns
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn


# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')
reduce_sum = fn.sum(msg='m', out='h')
reduce_max = fn.max(msg='m', out='h')


def reduce_mean(nodes):
    return {'h': torch.mean(nodes.mailbox['m'], dim=1)[0]}


class ApplyNodes(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp, layer):
        super(ApplyNodes, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)
        self.layer = layer

    def forward(self, nodes):
        h = self.mlp(nodes.data['h'])
        h = self.bn(h)
        h = F.relu(h)

        return {'h': h}


class GINLayer(nn.Module):
    """Neighbor pooling and reweight nodes before send graph into MLP"""
    def __init__(self, eps, layer, mlp, neighbor_pooling_type, learn_eps):
        super(GINLayer, self).__init__()
        self.bn = nn.BatchNorm1d(mlp.output_dim)
        self.neighbor_pooling_type = neighbor_pooling_type
        self.eps = eps
        self.learn_eps = learn_eps
        self.layer = layer
        self.apply_mod = ApplyNodes(mlp, layer)

    def forward(self, g, feature):
        g.ndata['h'] = feature

        if self.neighbor_pooling_type == 'sum':
            reduce_func = reduce_sum
        elif self.neighbor_pooling_type == 'mean':
            reduce_func = reduce_mean
        elif self.neighbor_pooling_type == 'max':
            reduce_func = reduce_max
        else:
            raise NotImplementedError()

        h = feature  # h0
        g.update_all(msg, reduce_func)
        pooled = g.ndata['h']

        # reweight the center node when aggregating it with its neighbors
        if self.learn_eps:
            pooled = pooled + (1 + self.eps[self.layer])*h

        g.ndata['h'] = pooled
        g.apply_nodes(func=self.apply_mod)

        return g.ndata.pop('h')


class MLP(nn.Module):
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
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class GIN(nn.Module):
    """GIN model"""
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type, device):
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
        device: str
            which device to use

        """
        super(GIN, self).__init__()
        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(GINLayer(
                self.eps, layer, mlp, neighbor_pooling_type, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))

    def forward(self, g):
        h = g.ndata['attr']
        h = h.to(self.device)

        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for layer in range(self.num_layers - 1):
            h = self.ginlayers[layer](g, h)
            hidden_rep.append(h)

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for layer, h in enumerate(hidden_rep):
            g.ndata['h'] = h
            if self.graph_pooling_type == 'sum':
                pooled_h = dgl.sum_nodes(g, 'h')
            elif self.graph_pooling_type == 'mean':
                pooled_h = dgl.mean_nodes(g, 'h')
            elif self.graph_pooling_type == 'max':
                pooled_h = dgl.max_nodes(g, 'h')
            else:
                raise NotImplementedError()

            score_over_layer += F.dropout(
                self.linears_prediction[layer](pooled_h),
                self.final_dropout,
                training=self.training)

        return score_over_layer
