import copy

import dgl

import torch
from dgl.nn.pytorch.conv import GraphConv, SAGEConv
from torch import nn
from torch.nn import BatchNorm1d, Parameter
from torch.nn.init import ones_, zeros_


class LayerNorm(nn.Module):
    def __init__(self, in_channels, eps=1e-5, affine=True):
        super().__init__()
        self.in_channels = in_channels
        self.eps = eps

        if affine:
            self.weight = Parameter(torch.Tensor(in_channels))
            self.bias = Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        ones_(self.weight)
        zeros_(self.bias)

    def forward(self, x, batch=None):
        device = x.device
        if batch is None:
            x = x - x.mean()
            out = x / (x.std(unbiased=False) + self.eps)
        else:
            batch_size = int(batch.max()) + 1
            batch_idx = [batch == i for i in range(batch_size)]
            norm = (
                torch.tensor([i.sum() for i in batch_idx], dtype=x.dtype)
                .clamp_(min=1)
                .to(device)
            )
            norm = norm.mul_(x.size(-1)).view(-1, 1)
            tmp_list = [x[i] for i in batch_idx]
            mean = (
                torch.concat([i.sum(0).unsqueeze(0) for i in tmp_list], dim=0)
                .sum(dim=-1, keepdim=True)
                .to(device)
            )
            mean = mean / norm
            x = x - mean.index_select(0, batch.long())
            var = (
                torch.concat(
                    [(i * i).sum(0).unsqueeze(0) for i in tmp_list], dim=0
                )
                .sum(dim=-1, keepdim=True)
                .to(device)
            )
            var = var / norm
            out = x / (var + self.eps).sqrt().index_select(0, batch.long())

        if self.weight is not None and self.bias is not None:
            out = out * self.weight + self.bias

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels})"


class MLP_Predictor(nn.Module):
    r"""MLP used for predictor. The MLP has one hidden layer.
    Args:
        input_size (int): Size of input features.
        output_size (int): Size of output features.
        hidden_size (int, optional): Size of hidden layer. (default: :obj:`4096`).
    """

    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.PReLU(1),
            nn.Linear(hidden_size, output_size, bias=True),
        )
        self.reset_parameters()

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()


class GCN(nn.Module):
    def __init__(self, layer_sizes, batch_norm_mm=0.99):
        super(GCN, self).__init__()

        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(GraphConv(in_dim, out_dim))
            self.layers.append(BatchNorm1d(out_dim, momentum=batch_norm_mm))
            self.layers.append(nn.PReLU())

    def forward(self, g):
        x = g.ndata["feat"]
        for layer in self.layers:
            if isinstance(layer, GraphConv):
                x = layer(g, x)
            else:
                x = layer(x)
        return x

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


class GraphSAGE_GCN(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()

        input_size, hidden_size, embedding_size = layer_sizes

        self.convs = nn.ModuleList(
            [
                SAGEConv(input_size, hidden_size, "mean"),
                SAGEConv(hidden_size, hidden_size, "mean"),
                SAGEConv(hidden_size, embedding_size, "mean"),
            ]
        )

        self.skip_lins = nn.ModuleList(
            [
                nn.Linear(input_size, hidden_size, bias=False),
                nn.Linear(input_size, hidden_size, bias=False),
            ]
        )

        self.layer_norms = nn.ModuleList(
            [
                LayerNorm(hidden_size),
                LayerNorm(hidden_size),
                LayerNorm(embedding_size),
            ]
        )

        self.activations = nn.ModuleList(
            [
                nn.PReLU(),
                nn.PReLU(),
                nn.PReLU(),
            ]
        )

    def forward(self, g):
        x = g.ndata["feat"]
        if "batch" in g.ndata.keys():
            batch = g.ndata["batch"]
        else:
            batch = None

        h1 = self.convs[0](g, x)
        h1 = self.layer_norms[0](h1, batch)
        h1 = self.activations[0](h1)

        x_skip_1 = self.skip_lins[0](x)
        h2 = self.convs[1](g, h1 + x_skip_1)
        h2 = self.layer_norms[1](h2, batch)
        h2 = self.activations[1](h2)

        x_skip_2 = self.skip_lins[1](x)
        ret = self.convs[2](g, h1 + h2 + x_skip_2)
        ret = self.layer_norms[2](ret, batch)
        ret = self.activations[2](ret)
        return ret

    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()
        for m in self.skip_lins:
            m.reset_parameters()
        for m in self.activations:
            m.weight.data.fill_(0.25)
        for m in self.layer_norms:
            m.reset_parameters()


class BGRL(nn.Module):
    r"""BGRL architecture for Graph representation learning.
    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.
    .. note::
        `encoder` must have a `reset_parameters` method, as the weights of the target network will be initialized
        differently from the online network.
    """

    def __init__(self, encoder, predictor):
        super(BGRL, self).__init__()
        # online network
        self.online_encoder = encoder
        self.predictor = predictor

        # target network
        self.target_encoder = copy.deepcopy(encoder)

        # reinitialize weights
        self.target_encoder.reset_parameters()

        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(
            self.predictor.parameters()
        )

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.
        Args:
            mm (float): Momentum used in moving average update.
        """
        for param_q, param_k in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1.0 - mm)

    def forward(self, online_x, target_x):
        # forward online network
        online_y = self.online_encoder(online_x)

        # prediction
        online_q = self.predictor(online_y)

        # forward target network
        with torch.no_grad():
            target_y = self.target_encoder(target_x).detach()
        return online_q, target_y


def compute_representations(net, dataset, device):
    r"""Pre-computes the representations for the entire data.
    Returns:
        [torch.Tensor, torch.Tensor]: Representations and labels.
    """
    net.eval()
    reps = []
    labels = []

    if len(dataset) == 1:
        g = dataset[0]
        g = dgl.add_self_loop(g)
        g = g.to(device)
        with torch.no_grad():
            reps.append(net(g))
            labels.append(g.ndata["label"])
    else:
        for g in dataset:
            # forward
            g = g.to(device)
            with torch.no_grad():
                reps.append(net(g))
                labels.append(g.ndata["label"])

    reps = torch.cat(reps, dim=0)
    labels = torch.cat(labels, dim=0)
    return [reps, labels]
