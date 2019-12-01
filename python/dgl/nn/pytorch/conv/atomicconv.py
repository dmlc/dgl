import numpy as np
import torch as th
import torch.nn as nn

from .... import function as fn

class RadialPooling(nn.Module):
    def __init__(self, interaction_cutoffs, rbf_kernel_means, rbf_kernel_scaling):
        super(RadialPooling, self).__init__()

        self.interaction_cutoffs = nn.Parameter(
            interaction_cutoffs.reshape(-1, 1, 1), requires_grad=True)
        self.rbf_kernel_means = nn.Parameter(
            rbf_kernel_means.reshape(-1, 1, 1), requires_grad=True)
        self.rbf_kernel_scaling = nn.Parameter(
            rbf_kernel_scaling.reshape(-1, 1, 1), requires_grad=True)

    def forward(self, distances):
        scaled_euclidean_distance = - self.rbf_kernel_scaling * \
                                    (distances - self.rbf_kernel_means) ** 2          # (K, E, 1)
        rbf_kernel_results = th.exp(scaled_euclidean_distance)                        # (K, E, 1)

        cos_values = 0.5 * (th.cos(np.pi * distances / self.interaction_cutoffs) + 1) # (K, E, 1)
        cutoff_values = th.where(
            distances <= self.interaction_cutoffs,
            cos_values, th.zeros_like(cos_values))                                    # (K, E, 1)

        # Note that there appears to be an inconsistency between the paper and
        # DeepChem's implementation. In the paper, the scaled_euclidean_distance first
        # gets multiplied by cutoff_values, followed by exponentiation. Here we follow
        # the practice of DeepChem.
        return rbf_kernel_results * cutoff_values

class AtomicConv(nn.Module):
    """
    Parameters
    ----------
    features_to_use: None or float32 tensor of shape (T)
    """
    def __init__(self, radial_params, features_to_use=None):
        super(AtomicConv, self).__init__()

        self.radial_pooling = RadialPooling(interaction_cutoffs=radial_params[:, 0],
                                            rbf_kernel_means=radial_params[:, 1],
                                            rbf_kernel_scaling=radial_params[:, 2])
        self.features_to_use = nn.Parameter(features_to_use, requires_grad=False)
        if features_to_use is None:
            self.num_channels = 1
        else:
            self.num_channels = len(features_to_use)

    def forward(self, graph, feat, distances):
        # (K, E, 1)
        radial_pooled_values = self.radial_pooling(distances)
        graph = graph.local_var()
        if self.features_to_use is not None:
            # (V * d_in, 1)
            flattened_feat = feat.reshape(-1, 1)
            # (V * d_in, len(self.features_to_use))
            flattened_feat = (flattened_feat == self.features_to_use).float()
            # (V, d_in * len(self.features_to_use))
            feat = flattened_feat.reshape(feat.shape[0], -1)
        # (V, d_in * len(self.features_to_use), 1)
        graph.ndata['hv'] = feat.unsqueeze(-1)
        # (E, K)
        graph.edata['he'] = radial_pooled_values.reshape(graph.number_of_edges(), -1)
        graph.update_all(fn.src_mul_edge('hv', 'he', 'm'), fn.sum('m', 'hv_new'))

        # (V, K * d_in * len(self.features_to_use))
        return graph.ndata['hv_new'].reshape(
            graph.number_of_nodes(), -1).contiguous()
