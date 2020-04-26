"""WLN"""
# pylint: disable= no-member, arguments-differ, invalid-name
import math
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter

__all__ = ['WLN']

class WLNLinear(nn.Module):
    r"""Linear layer for WLN

    Let stddev be

    .. math::
        \min(\frac{1.0}{\sqrt{in_feats}}, 0.1)

    The weight of the linear layer is initialized from a normal distribution
    with mean 0 and std as specified in stddev.

    Parameters
    ----------
    in_feats : int
        Size for the input.
    out_feats : int
        Size for the output.
    bias : bool
        Whether bias will be added to the output. Default to True.
    """
    def __init__(self, in_feats, out_feats, bias=True):
        super(WLNLinear, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.weight = Parameter(torch.Tensor(out_feats, in_feats))
        if bias:
            self.bias = Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters."""
        stddev = min(1.0 / math.sqrt(self.in_feats), 0.1)
        nn.init.normal_(self.weight, std=stddev)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, feats):
        """Applies the layer.

        Parameters
        ----------
        feats : float32 tensor of shape (N, *, in_feats)
            N for the number of samples, * for any additional dimensions.

        Returns
        -------
        float32 tensor of shape (N, *, out_feats)
            Result of the layer.
        """
        return F.linear(feats, self.weight, self.bias)

    def extra_repr(self):
        """Return a description of the layer."""
        return 'in_feats={}, out_feats={}, bias={}'.format(
            self.in_feats, self.out_feats, self.bias is not None
        )

class WLN(nn.Module):
    """Weisfeiler-Lehman Network (WLN)

    WLN is introduced in `Predicting Organic Reaction Outcomes with
    Weisfeiler-Lehman Network <https://arxiv.org/abs/1709.04555>`__.

    This class performs message passing and updates node representations.

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    node_out_feats : int
        Size for the output node representations. Default to 300.
    n_layers : int
        Number of times for message passing. Note that same parameters
        are shared across n_layers message passing. Default to 3.
    """
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats=300,
                 n_layers=3):
        super(WLN, self).__init__()

        self.n_layers = n_layers
        self.project_node_in_feats = nn.Sequential(
            WLNLinear(node_in_feats, node_out_feats, bias=False),
            nn.ReLU()
        )
        self.project_concatenated_messages = nn.Sequential(
            WLNLinear(edge_in_feats + node_out_feats, node_out_feats),
            nn.ReLU()
        )
        self.get_new_node_feats = nn.Sequential(
            WLNLinear(2 * node_out_feats, node_out_feats),
            nn.ReLU()
        )
        self.project_edge_messages = WLNLinear(edge_in_feats, node_out_feats, bias=False)
        self.project_node_messages = WLNLinear(node_out_feats, node_out_feats, bias=False)
        self.project_self = WLNLinear(node_out_feats, node_out_feats, bias=False)

    def forward(self, g, node_feats, edge_feats):
        """Performs message passing and updates node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges.

        Returns
        -------
        float32 tensor of shape (V, node_out_feats)
            Updated node representations.
        """
        node_feats = self.project_node_in_feats(node_feats)
        for _ in range(self.n_layers):
            g = g.local_var()
            g.ndata['hv'] = node_feats
            g.apply_edges(fn.copy_src('hv', 'he_src'))
            concat_edge_feats = torch.cat([g.edata['he_src'], edge_feats], dim=1)
            g.edata['he'] = self.project_concatenated_messages(concat_edge_feats)
            g.update_all(fn.copy_edge('he', 'm'), fn.sum('m', 'hv_new'))
            node_feats = self.get_new_node_feats(
                torch.cat([node_feats, g.ndata['hv_new']], dim=1))

        g = g.local_var()
        g.ndata['hv'] = self.project_node_messages(node_feats)
        g.edata['he'] = self.project_edge_messages(edge_feats)
        g.update_all(fn.u_mul_e('hv', 'he', 'm'), fn.sum('m', 'h_nbr'))
        h_self = self.project_self(node_feats)  # (V, node_out_feats)
        return g.ndata['h_nbr'] * h_self
