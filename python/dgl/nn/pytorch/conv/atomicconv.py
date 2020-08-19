"""Torch Module for Atomic Convolution Layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import numpy as np
import torch as th
import torch.nn as nn

class RadialPooling(nn.Module):
    r"""

    Description
    -----------
    Radial pooling from paper `Atomic Convolutional Networks for
    Predicting Protein-Ligand Binding Affinity <https://arxiv.org/abs/1703.10603>`__.

    We denote the distance between atom :math:`i` and :math:`j` by :math:`r_{ij}`.

    A radial pooling layer transforms distances with radial filters. For radial filter
    indexed by :math:`k`, it projects edge distances with

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

    Parameters
    ----------
    interaction_cutoffs : float32 tensor of shape (K)
        :math:`c_k` in the equations above. Roughly they can be considered as learnable cutoffs
        and two atoms are considered as connected if the distance between them is smaller than
        the cutoffs. K for the number of radial filters.
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
        """

        Description
        -----------
        Apply the layer to transform edge distances.

        Parameters
        ----------
        distances : Float32 tensor of shape (E, 1)
            Distance between end nodes of edges. E for the number of edges.

        Returns
        -------
        Float32 tensor of shape (K, E, 1)
            Transformed edge distances. K for the number of radial filters.
        """
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

def msg_func(edges):
    """

    Description
    -----------
    Send messages along edges.

    Parameters
    ----------
    edges : EdgeBatch
        A batch of edges.

    Returns
    -------
    dict mapping 'm' to Float32 tensor of shape (E, K * T)
        Messages computed. E for the number of edges, K for the number of
        radial filters and T for the number of features to use
        (types of atomic number in the paper).
    """
    return {'m': th.einsum(
        'ij,ik->ijk', edges.src['hv'], edges.data['he']).view(len(edges), -1)}

def reduce_func(nodes):
    """

    Description
    -----------
    Collect messages and update node representations.

    Parameters
    ----------
    nodes : NodeBatch
        A batch of nodes.

    Returns
    -------
    dict mapping 'hv_new' to Float32 tensor of shape (V, K * T)
        Updated node representations. V for the number of nodes, K for the number of
        radial filters and T for the number of features to use
        (types of atomic number in the paper).
    """
    return {'hv_new': nodes.mailbox['m'].sum(1)}

class AtomicConv(nn.Module):
    r"""

    Description
    -----------
    Atomic Convolution Layer from paper `Atomic Convolutional Networks for
    Predicting Protein-Ligand Binding Affinity <https://arxiv.org/abs/1703.10603>`__.

    Denoting the type of atom :math:`i` by :math:`z_i` and the distance between atom
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

    Then concatenate the results for all RBF kernels and atom types.

    Parameters
    ----------
    interaction_cutoffs : float32 tensor of shape (K)
        :math:`c_k` in the equations above. Roughly they can be considered as learnable cutoffs
        and two atoms are considered as connected if the distance between them is smaller than
        the cutoffs. K for the number of radial filters.
    rbf_kernel_means : float32 tensor of shape (K)
        :math:`r_k` in the equations above. K for the number of radial filters.
    rbf_kernel_scaling : float32 tensor of shape (K)
        :math:`\gamma_k` in the equations above. K for the number of radial filters.
    features_to_use : None or float tensor of shape (T)
        In the original paper, these are atomic numbers to consider, representing the types
        of atoms. T for the number of types of atomic numbers. Default to None.

    Note
    ----

    * This convolution operation is designed for molecular graphs in Chemistry, but it might
      be possible to extend it to more general graphs.

    * There seems to be an inconsistency about the definition of :math:`e_{ij}^{k}` in the
      paper and the author's implementation. We follow the author's implementation. In the
      paper, :math:`e_{ij}^{k}` was defined as
      :math:`\exp(-\gamma_{k}|r_{ij}-r_{k}|^2 * f_{ij}^{k})`.

    * :math:`\gamma_{k}`, :math:`r_k` and :math:`c_k` are all learnable.

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import AtomicConv

    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 1)
    >>> edist = th.ones(6, 1)
    >>> interaction_cutoffs = th.ones(3).float() * 2
    >>> rbf_kernel_means = th.ones(3).float()
    >>> rbf_kernel_scaling = th.ones(3).float()
    >>> conv = AtomicConv(interaction_cutoffs, rbf_kernel_means, rbf_kernel_scaling)
    >>> res = conv(g, feat, edist)
    >>> res
    tensor([[0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000],
                [1.0000, 1.0000, 1.0000],
                [0.5000, 0.5000, 0.5000],
                [0.0000, 0.0000, 0.0000]], grad_fn=<ViewBackward>)
    """
    def __init__(self, interaction_cutoffs, rbf_kernel_means,
                 rbf_kernel_scaling, features_to_use=None):
        super(AtomicConv, self).__init__()

        self.radial_pooling = RadialPooling(interaction_cutoffs=interaction_cutoffs,
                                            rbf_kernel_means=rbf_kernel_means,
                                            rbf_kernel_scaling=rbf_kernel_scaling)
        if features_to_use is None:
            self.num_channels = 1
            self.features_to_use = None
        else:
            self.num_channels = len(features_to_use)
            self.features_to_use = nn.Parameter(features_to_use, requires_grad=False)

    def forward(self, graph, feat, distances):
        """

        Description
        -----------
        Apply the atomic convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            Topology based on which message passing is performed.
        feat : Float32 tensor of shape :math:`(V, 1)`
            Initial node features, which are atomic numbers in the paper.
            :math:`V` for the number of nodes.
        distances : Float32 tensor of shape :math:`(E, 1)`
            Distance between end nodes of edges. E for the number of edges.

        Returns
        -------
        Float32 tensor of shape :math:`(V, K * T)`
            Updated node representations. :math:`V` for the number of nodes, :math:`K` for the
            number of radial filters, and :math:`T` for the number of types of atomic numbers.
        """
        with graph.local_scope():
            radial_pooled_values = self.radial_pooling(distances)                # (K, E, 1)
            if self.features_to_use is not None:
                feat = (feat == self.features_to_use).float()                    # (V, T)
            graph.ndata['hv'] = feat
            graph.edata['he'] = radial_pooled_values.transpose(1, 0).squeeze(-1) # (E, K)
            graph.update_all(msg_func, reduce_func)

            return graph.ndata['hv_new'].view(graph.number_of_nodes(), -1)       # (V, K * T)
