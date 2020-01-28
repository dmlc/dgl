"""Torch modules for interaction blocks in SchNet"""
# pylint: disable= no-member, arguments-differ, invalid-name
import numpy as np
import torch.nn as nn

from .... import function as fn

class ShiftedSoftplus(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{SSP}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x)) - \log(\text{shift})

    Parameters
    ----------
    beta : int
        :math:`\beta` value for the mathematical formulation. Default to 1.
    shift : int
        :math:`\text{shift}` value for the mathematical formulation. Default to 2.
    """
    def __init__(self, beta=1, shift=2, threshold=20):
        super(ShiftedSoftplus, self).__init__()

        self.shift = shift
        self.softplus = nn.Softplus(beta=beta, threshold=threshold)

    def forward(self, inputs):
        """Applies the activation function.

        Parameters
        ----------
        inputs : float32 tensor of shape (N, *)
            * denotes any number of additional dimensions.

        Returns
        -------
        float32 tensor of shape (N, *)
            Result of applying the activation function to the input.
        """
        return self.softplus(inputs) - np.log(float(self.shift))

class CFConv(nn.Module):
    r"""CFConv in SchNet.

    SchNet is introduced in `SchNet: A continuous-filter convolutional neural network for
    modeling quantum interactions <https://arxiv.org/abs/1706.08566>`__.

    It combines node and edge features in message passing and updates node representations.

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    hidden_feats : int
        Size for the hidden representations.
    out_feats : int
        Size for the output representations.
    """
    def __init__(self, node_in_feats, edge_in_feats, hidden_feats, out_feats):
        super(CFConv, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Linear(edge_in_feats, hidden_feats),
            ShiftedSoftplus(),
            nn.Linear(hidden_feats, hidden_feats),
            ShiftedSoftplus()
        )
        self.project_node = nn.Linear(node_in_feats, hidden_feats)
        self.project_out = nn.Sequential(
            nn.Linear(hidden_feats, out_feats),
            ShiftedSoftplus()
        )

    def forward(self, g, node_feats, edge_feats):
        """Performs message passing and updates node representations.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features, V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features, E for the number of edges.

        Returns
        -------
        float32 tensor of shape (V, out_feats)
            Updated node representations.
        """
        g = g.local_var()
        g.ndata['hv'] = self.project_node(node_feats)
        g.edata['he'] = self.project_edge(edge_feats)
        g.update_all(fn.u_mul_e('hv', 'he', 'm'), fn.sum('m', 'h'))
        return self.project_out(g.ndata['h'])
