import math

import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class ARMAConv(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_stacks,
        num_layers,
        activation=None,
        dropout=0.0,
        bias=True,
    ):
        super(ARMAConv, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = num_stacks
        self.T = num_layers
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

        # init weight
        self.w_0 = nn.ModuleDict(
            {
                str(k): nn.Linear(in_dim, out_dim, bias=False)
                for k in range(self.K)
            }
        )
        # deeper weight
        self.w = nn.ModuleDict(
            {
                str(k): nn.Linear(out_dim, out_dim, bias=False)
                for k in range(self.K)
            }
        )
        # v
        self.v = nn.ModuleDict(
            {
                str(k): nn.Linear(in_dim, out_dim, bias=False)
                for k in range(self.K)
            }
        )
        # bias
        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(self.K, self.T, 1, self.out_dim)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        for k in range(self.K):
            glorot(self.w_0[str(k)].weight)
            glorot(self.w[str(k)].weight)
            glorot(self.v[str(k)].weight)
        zeros(self.bias)

    def forward(self, g, feats):
        with g.local_scope():
            init_feats = feats
            # assume that the graphs are undirected and graph.in_degrees() is the same as graph.out_degrees()
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).to(feats.device).unsqueeze(1)
            output = []

            for k in range(self.K):
                feats = init_feats
                for t in range(self.T):
                    feats = feats * norm
                    g.ndata["h"] = feats
                    g.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                    feats = g.ndata.pop("h")
                    feats = feats * norm

                    if t == 0:
                        feats = self.w_0[str(k)](feats)
                    else:
                        feats = self.w[str(k)](feats)

                    feats += self.dropout(self.v[str(k)](init_feats))
                    feats += self.v[str(k)](self.dropout(init_feats))

                    if self.bias is not None:
                        feats += self.bias[k][t]

                    if self.activation is not None:
                        feats = self.activation(feats)
                output.append(feats)

            return torch.stack(output).mean(dim=0)


class ARMA4NC(nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        num_stacks,
        num_layers,
        activation=None,
        dropout=0.0,
    ):
        super(ARMA4NC, self).__init__()

        self.conv1 = ARMAConv(
            in_dim=in_dim,
            out_dim=hid_dim,
            num_stacks=num_stacks,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )

        self.conv2 = ARMAConv(
            in_dim=hid_dim,
            out_dim=out_dim,
            num_stacks=num_stacks,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, feats):
        feats = F.relu(self.conv1(g, feats))
        feats = self.dropout(feats)
        feats = self.conv2(g, feats)
        return feats
