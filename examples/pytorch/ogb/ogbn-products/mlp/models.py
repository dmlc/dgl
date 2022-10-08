import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        in_feats,
        n_classes,
        n_layers,
        n_hidden,
        activation,
        dropout=0.0,
        input_drop=0.0,
        residual=False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes

            self.linears.append(nn.Linear(in_hidden, out_hidden))

            if i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_hidden))

        self.activation = activation
        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, h):
        h = self.input_drop(h)

        h_last = None

        for i in range(self.n_layers):
            h = self.linears[i](h)

            if self.residual and 0 < i < self.n_layers - 1:
                h += h_last

            h_last = h

            if i < self.n_layers - 1:
                h = self.norms[i](h)
                h = self.activation(h, inplace=True)
                h = self.dropout(h)

        return h
