"""Torch Module for Atomic Convolution Layer"""
import numpy as np
import torch as th
import torch.nn as nn

from .... import function as fn

class RadialPooling(nn.Module):
    """Radial Pooling from paper `Atomic Convolutional Networks for
    Predicting Protein-Ligand Binding Affinity <https://arxiv.org/abs/1703.10603>`__.

    Let :math:`r_{ij}` be the distance between

    Parameters
    ----------
    interaction_cutoffs : float32 tensor of shape (K)
        :math:`c_k` in the equations above. Roughly they can be considered as learnable cutoffs
        for deciding if two atoms are neighbors. K for the number of radial filters.
    rbf_kernel_means : float32 tensor of shape (K)
        :math:`r_k` in the equations above. K for the number of radial filters.
    rbf_kernel_scaling : float32 tensor of shape (K)
        :math:`\gamma_k` in the equations above. K for the number of radial filters.
    """
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
    r"""Atomic Convolution Layer from paper `Atomic Convolutional Networks for
    Predicting Protein-Ligand Binding Affinity <https://arxiv.org/abs/1703.10603>`__.

    We denote the type of atom :math:`i` by :math:`z_i` and the distance between atom
    :math:`i` and :math:`j` by :math:`r_{ij}`.

    **Distance Transformation**

    An atomic convolution layer first transforms distances with radial filters and
    then perform a pooling operation.

    For radial filter indexed by :math:`k`, it projects edge distances with

    .. math::
        h_{ij}^{k} = \exp(-\gamma_{k}|r_{ij}-r_{k}|^2)

    If :math:`r_{ij} < c_k`,

    .. math::
        f_{ij}^{k} = 0.5 * \cos(\frac{\pi r_{ij}}{c_k} + 1),

    else,

    .. math::
        f_{ij}^{k} = 0.

    Finally,

    .. math::
        e_{ij}^{k} = h_{ij}^{k} * f_{ij}^{k}

    **Aggregation**

    For each type :math:`t`, each atom collects distance information from all neighbor atoms
    of type :math:`t`:

    .. math::
        p_{i, t}^{k} = \sum_{j\in N(i)} e_{ij}^{k} * 1(z_j == t)

    We concatenate the results for all RBF kernels and atom types.

    Notes
    -----

    * This convolution operation is designed for molecular graphs in Chemistry, but it might
      be possible to extend it to more general graphs.

    * There seems to be an inconsistency about the definition of :math:`e_{ij}^{k}` in the
      paper and the author's implementation. We follow the author's implementation. In the
      paper, :math:`e_{ij}^{k}` was defined as
      :math:`\exp(-\gamma_{k}|r_{ij}-r_{k}|^2 * f_{ij}^{k})`.

    * :math:`\gamma_{k}`, :math:`r_k` and :math:`c_k` are all learnable.

    Parameters
    ----------
    interaction_cutoffs : float32 tensor of shape (K)
        :math:`c_k` in the equations above. Roughly they can be considered as learnable cutoffs
        for deciding if two atoms are neighbors. K for the number of radial filters.
    rbf_kernel_means : float32 tensor of shape (K)
        :math:`r_k` in the equations above. K for the number of radial filters.
    rbf_kernel_scaling : float32 tensor of shape (K)
        :math:`\gamma_k` in the equations above. K for the number of radial filters.
    features_to_use : None or float tensor of shape (T)
        In the original paper, these are atomic numbers to consider, representing the types
        of atoms. T for the number of types of atomic numbers. Default to None.
    """
    def __init__(self, interaction_cutoffs, rbf_kernel_means,
                 rbf_kernel_scaling, features_to_use=None):
        super(AtomicConv, self).__init__()

        self.radial_pooling = RadialPooling(interaction_cutoffs=interaction_cutoffs,
                                            rbf_kernel_means=rbf_kernel_means,
                                            rbf_kernel_scaling=rbf_kernel_scaling)
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
            # (V, 1)
            flattened_feat = feat
            # (V, len(self.features_to_use))
            flattened_feat = (flattened_feat == self.features_to_use).float()
            # (V, len(self.features_to_use))
            feat = flattened_feat
        else:
            feat = feat
        # (V, d_in * len(self.features_to_use), 1)
        graph.ndata['hv'] = feat.unsqueeze(-1)
        # (E, K)
        graph.edata['he'] = radial_pooled_values.transpose(1, 0).squeeze(-1)
        graph.update_all(fn.src_mul_edge('hv', 'he', 'm'), fn.sum('m', 'hv_new'))

        # (V, K * len(self.features_to_use))
        return graph.ndata['hv_new'].view(graph.number_of_nodes(), -1)
