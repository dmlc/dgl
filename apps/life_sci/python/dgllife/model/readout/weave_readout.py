"""Readout for Weave"""
# pylint: disable= no-member, arguments-differ, invalid-name
import dgl
import torch
import torch.nn as nn

from torch.distributions import Normal

__all__ = ['WeaveGather']

# pylint: disable=W0221, E1101, E1102
class WeaveGather(nn.Module):
    r"""Readout in Weave

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    gaussian_expand : bool
        Whether to expand each dimension of node features by gaussian histogram.
        Default to True.
    gaussian_memberships : list of 2-tuples
        For each tuple, the first and second element separately specifies the mean
        and std for constructing a normal distribution. This argument comes into
        effect only when ``gaussian_expand==True``. By default, we set this to be
        ``[(-1.645, 0.283), (-1.080, 0.170), (-0.739, 0.134), (-0.468, 0.118),
        (-0.228, 0.114), (0., 0.114), (0.228, 0.114), (0.468, 0.118),
        (0.739, 0.134), (1.080, 0.170), (1.645, 0.283)]``.
    activation : callable
        Activation function to apply. Default to tanh.
    """
    def __init__(self,
                 node_in_feats,
                 gaussian_expand=True,
                 gaussian_memberships=None,
                 activation=nn.Tanh()):
        super(WeaveGather, self).__init__()

        self.gaussian_expand = gaussian_expand
        if gaussian_expand:
            if gaussian_memberships is None:
                gaussian_memberships = [
                    (-1.645, 0.283), (-1.080, 0.170), (-0.739, 0.134), (-0.468, 0.118),
                    (-0.228, 0.114), (0., 0.114), (0.228, 0.114), (0.468, 0.118),
                    (0.739, 0.134), (1.080, 0.170), (1.645, 0.283)]
            means, stds = map(list, zip(*gaussian_memberships))
            self.means = nn.ParameterList([
                nn.Parameter(torch.tensor(value), requires_grad=False)
                for value in means
            ])
            self.stds = nn.ParameterList([
                nn.Parameter(torch.tensor(value), requires_grad=False)
                for value in stds
            ])
            self.to_out = nn.Linear(node_in_feats * len(self.means), node_in_feats)
            self.activation = activation

    def gaussian_histogram(self, node_feats):
        r"""Constructs a gaussian histogram to capture the distribution of features

        Parameters
        ----------
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.

        Returns
        -------
        float32 tensor of shape (V, node_in_feats * len(self.means))
            Updated node representations
        """
        gaussian_dists = [Normal(self.means[i], self.stds[i])
                          for i in range(len(self.means))]
        max_log_probs = [gaussian_dists[i].log_prob(self.means[i])
                         for i in range(len(self.means))]
        # Normalize the probabilities by the maximum point-wise probabilities,
        # whose results will be in range [0, 1]. Note that division of probabilities
        # is equivalent to subtraction of log probabilities and the latter one is cheaper.
        log_probs = [gaussian_dists[i].log_prob(node_feats) - max_log_probs[i]
                     for i in range(len(self.means))]
        probs = torch.stack(log_probs, dim=2).exp() # (V, node_in_feats, len(self.means))
        # Add a bias to avoid numerical issues in division
        probs = probs + 1e-7
        # Normalize the probabilities across all Gaussian distributions
        probs = probs / probs.sum(2, keepdim=True)

        return probs.reshape(node_feats.shape[0],
                             node_feats.shape[1] * len(self.means))

    def forward(self, g, node_feats):
        r"""Computes graph representations out of node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.

        Returns
        -------
        g_feats : float32 tensor of shape (G, node_in_feats)
            Output graph representations. G for the number of graphs in the batch.
        """
        if self.gaussian_expand:
            node_feats = self.gaussian_histogram(node_feats)

        with g.local_scope():
            g.ndata['h'] = node_feats
            g_feats = dgl.sum_nodes(g, 'h')

        if self.gaussian_expand:
            g_feats = self.to_out(g_feats)
            if self.activation is not None:
                g_feats = self.activation(g_feats)

        return g_feats
