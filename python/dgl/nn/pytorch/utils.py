"""Utilities for pytorch NN package"""
# pylint: disable=no-member, invalid-name

import torch as th
import torch.nn.functional as F
from torch import nn

from ... import DGLGraph, function as fn
from ...base import dgl_warning


def matmul_maybe_select(A, B):
    """Perform Matrix multiplication C = A * B but A could be an integer id vector.

    If A is an integer vector, we treat it as multiplying a one-hot encoded tensor.
    In this case, the expensive dense matrix multiply can be replaced by a much
    cheaper index lookup.

    For example,
    ::

        A = [2, 0, 1],
        B = [[0.1, 0.2],
             [0.3, 0.4],
             [0.5, 0.6]]

    then matmul_maybe_select(A, B) is equivalent to
    ::

        [[0, 0, 1],     [[0.1, 0.2],
         [1, 0, 0],  *   [0.3, 0.4],
         [0, 1, 0]]      [0.5, 0.6]]

    In all other cases, perform a normal matmul.

    Parameters
    ----------
    A : torch.Tensor
        lhs tensor
    B : torch.Tensor
        rhs tensor

    Returns
    -------
    C : torch.Tensor
        result tensor
    """
    if A.dtype == th.int64 and len(A.shape) == 1:
        return B.index_select(0, A)
    else:
        return th.matmul(A, B)


def bmm_maybe_select(A, B, index):
    """Slice submatrices of A by the given index and perform bmm.

    B is a 3D tensor of shape (N, D1, D2), which can be viewed as a stack of
    N matrices of shape (D1, D2). The input index is an integer vector of length M.
    A could be either:
    (1) a dense tensor of shape (M, D1),
    (2) an integer vector of length M.
    The result C is a 2D matrix of shape (M, D2)

    For case (1), C is computed by bmm:
    ::

        C[i, :] = matmul(A[i, :], B[index[i], :, :])

    For case (2), C is computed by index select:
    ::

        C[i, :] = B[index[i], A[i], :]

    Parameters
    ----------
    A : torch.Tensor
        lhs tensor
    B : torch.Tensor
        rhs tensor
    index : torch.Tensor
        index tensor

    Returns
    -------
    C : torch.Tensor
        return tensor
    """
    if A.dtype == th.int64 and len(A.shape) == 1:
        # following is a faster version of B[index, A, :]
        B = B.view(-1, B.shape[2])
        flatidx = index * B.shape[1] + A
        return B.index_select(0, flatidx)
    else:
        BB = B.index_select(0, index)
        return th.bmm(A.unsqueeze(1), BB).squeeze()


# pylint: disable=W0235
class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive.
    (Identity has already been supported by PyTorch 1.2, we will directly
    import torch.nn.Identity in the future)
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input"""
        return x


class Sequential(nn.Sequential):
    r"""A sequential container for stacking graph neural network modules

    DGL supports two modes: sequentially apply GNN modules on 1) the same graph or
    2) a list of given graphs. In the second case, the number of graphs equals the
    number of modules inside this container.

    Parameters
    ----------
    *args :
        Sub-modules of torch.nn.Module that will be added to the container in
        the order by which they are passed in the constructor.

    Examples
    --------
    The following example uses PyTorch backend.

    Mode 1: sequentially apply GNN modules on the same graph

    >>> import torch
    >>> import dgl
    >>> import torch.nn as nn
    >>> import dgl.function as fn
    >>> from dgl.nn.pytorch import Sequential
    >>> class ExampleLayer(nn.Module):
    >>>     def __init__(self):
    >>>         super().__init__()
    >>>     def forward(self, graph, n_feat, e_feat):
    >>>         with graph.local_scope():
    >>>             graph.ndata['h'] = n_feat
    >>>             graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
    >>>             n_feat += graph.ndata['h']
    >>>             graph.apply_edges(fn.u_add_v('h', 'h', 'e'))
    >>>             e_feat += graph.edata['e']
    >>>             return n_feat, e_feat
    >>>
    >>> g = dgl.DGLGraph()
    >>> g.add_nodes(3)
    >>> g.add_edges([0, 1, 2, 0, 1, 2, 0, 1, 2], [0, 0, 0, 1, 1, 1, 2, 2, 2])
    >>> net = Sequential(ExampleLayer(), ExampleLayer(), ExampleLayer())
    >>> n_feat = torch.rand(3, 4)
    >>> e_feat = torch.rand(9, 4)
    >>> net(g, n_feat, e_feat)
    (tensor([[39.8597, 45.4542, 25.1877, 30.8086],
             [40.7095, 45.3985, 25.4590, 30.0134],
             [40.7894, 45.2556, 25.5221, 30.4220]]),
     tensor([[80.3772, 89.7752, 50.7762, 60.5520],
             [80.5671, 89.3736, 50.6558, 60.6418],
             [80.4620, 89.5142, 50.3643, 60.3126],
             [80.4817, 89.8549, 50.9430, 59.9108],
             [80.2284, 89.6954, 50.0448, 60.1139],
             [79.7846, 89.6882, 50.5097, 60.6213],
             [80.2654, 90.2330, 50.2787, 60.6937],
             [80.3468, 90.0341, 50.2062, 60.2659],
             [80.0556, 90.2789, 50.2882, 60.5845]]))

    Mode 2: sequentially apply GNN modules on different graphs

    >>> import torch
    >>> import dgl
    >>> import torch.nn as nn
    >>> import dgl.function as fn
    >>> import networkx as nx
    >>> from dgl.nn.pytorch import Sequential
    >>> class ExampleLayer(nn.Module):
    >>>     def __init__(self):
    >>>         super().__init__()
    >>>     def forward(self, graph, n_feat):
    >>>         with graph.local_scope():
    >>>             graph.ndata['h'] = n_feat
    >>>             graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
    >>>             n_feat += graph.ndata['h']
    >>>             return n_feat.view(graph.num_nodes() // 2, 2, -1).sum(1)
    >>>
    >>> g1 = dgl.DGLGraph(nx.erdos_renyi_graph(32, 0.05))
    >>> g2 = dgl.DGLGraph(nx.erdos_renyi_graph(16, 0.2))
    >>> g3 = dgl.DGLGraph(nx.erdos_renyi_graph(8, 0.8))
    >>> net = Sequential(ExampleLayer(), ExampleLayer(), ExampleLayer())
    >>> n_feat = torch.rand(32, 4)
    >>> net([g1, g2, g3], n_feat)
    tensor([[209.6221, 225.5312, 193.8920, 220.1002],
            [250.0169, 271.9156, 240.2467, 267.7766],
            [220.4007, 239.7365, 213.8648, 234.9637],
            [196.4630, 207.6319, 184.2927, 208.7465]])
    """

    def __init__(self, *args):
        super(Sequential, self).__init__(*args)

    def forward(self, graph, *feats):
        r"""

        Sequentially apply modules to the input.

        Parameters
        ----------
        graph : DGLGraph or list of DGLGraphs
            The graph(s) to apply modules on.

        *feats :
            Input features.
            The output of the :math:`i`-th module should match the input
            of the :math:`(i+1)`-th module in the sequential.
        """
        if isinstance(graph, list):
            for graph_i, module in zip(graph, self):
                if not isinstance(feats, tuple):
                    feats = (feats,)
                feats = module(graph_i, *feats)
        elif isinstance(graph, DGLGraph):
            for module in self:
                if not isinstance(feats, tuple):
                    feats = (feats,)
                feats = module(graph, *feats)
        else:
            raise TypeError(
                "The first argument of forward must be a DGLGraph"
                " or a list of DGLGraph s"
            )
        return feats


class WeightBasis(nn.Module):
    r"""Basis decomposition from `Modeling Relational Data with Graph
    Convolutional Networks <https://arxiv.org/abs/1703.06103>`__

    It can be described as below:

    .. math::

        W_o = \sum_{b=1}^B a_{ob} V_b

    Each weight output :math:`W_o` is essentially a linear combination of basis
    transformations :math:`V_b` with coefficients :math:`a_{ob}`.

    If is useful as a form of regularization on a large parameter matrix. Thus,
    the number of weight outputs is usually larger than the number of bases.

    Parameters
    ----------
    shape : tuple[int]
        Shape of the basis parameter.
    num_bases : int
        Number of bases.
    num_outputs : int
        Number of outputs.
    """

    def __init__(self, shape, num_bases, num_outputs):
        super(WeightBasis, self).__init__()
        self.shape = shape
        self.num_bases = num_bases
        self.num_outputs = num_outputs

        if num_outputs <= num_bases:
            dgl_warning(
                "The number of weight outputs should be larger than the number"
                " of bases."
            )

        self.weight = nn.Parameter(th.Tensor(self.num_bases, *shape))
        nn.init.xavier_uniform_(
            self.weight, gain=nn.init.calculate_gain("relu")
        )
        # linear combination coefficients
        self.w_comp = nn.Parameter(th.Tensor(self.num_outputs, self.num_bases))
        nn.init.xavier_uniform_(
            self.w_comp, gain=nn.init.calculate_gain("relu")
        )

    def forward(self):
        r"""Forward computation

        Returns
        -------
        weight : torch.Tensor
            Composed weight tensor of shape ``(num_outputs,) + shape``
        """
        # generate all weights from bases
        weight = th.matmul(self.w_comp, self.weight.view(self.num_bases, -1))
        return weight.view(self.num_outputs, *self.shape)


class JumpingKnowledge(nn.Module):
    r"""The Jumping Knowledge aggregation module from `Representation Learning on
    Graphs with Jumping Knowledge Networks <https://arxiv.org/abs/1806.03536>`__

    It aggregates the output representations of multiple GNN layers with

    **concatenation**

    .. math::

        h_i^{(1)} \, \Vert \, \ldots \, \Vert \, h_i^{(T)}

    or **max pooling**

    .. math::

        \max \left( h_i^{(1)}, \ldots, h_i^{(T)} \right)

    or **LSTM**

    .. math::

        \sum_{t=1}^T \alpha_i^{(t)} h_i^{(t)}

    with attention scores :math:`\alpha_i^{(t)}` obtained from a BiLSTM

    Parameters
    ----------
    mode : str
        The aggregation to apply. It can be 'cat', 'max', or 'lstm',
        corresponding to the equations above in order.
    in_feats : int, optional
        This argument is only required if :attr:`mode` is ``'lstm'``.
        The output representation size of a single GNN layer. Note that
        all GNN layers need to have the same output representation size.
    num_layers : int, optional
        This argument is only required if :attr:`mode` is ``'lstm'``.
        The number of GNN layers for output aggregation.

    Examples
    --------
    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import JumpingKnowledge

    >>> # Output representations of two GNN layers
    >>> num_nodes = 3
    >>> in_feats = 4
    >>> feat_list = [th.zeros(num_nodes, in_feats), th.ones(num_nodes, in_feats)]

    >>> # Case1
    >>> model = JumpingKnowledge()
    >>> model(feat_list).shape
    torch.Size([3, 8])

    >>> # Case2
    >>> model = JumpingKnowledge(mode='max')
    >>> model(feat_list).shape
    torch.Size([3, 4])

    >>> # Case3
    >>> model = JumpingKnowledge(mode='max', in_feats=in_feats, num_layers=len(feat_list))
    >>> model(feat_list).shape
    torch.Size([3, 4])
    """

    def __init__(self, mode="cat", in_feats=None, num_layers=None):
        super(JumpingKnowledge, self).__init__()
        assert mode in [
            "cat",
            "max",
            "lstm",
        ], "Expect mode to be 'cat', or 'max' or 'lstm', got {}".format(mode)
        self.mode = mode

        if mode == "lstm":
            assert in_feats is not None, "in_feats is required for lstm mode"
            assert (
                num_layers is not None
            ), "num_layers is required for lstm mode"
            hidden_size = (num_layers * in_feats) // 2
            self.lstm = nn.LSTM(
                in_feats, hidden_size, bidirectional=True, batch_first=True
            )
            self.att = nn.Linear(2 * hidden_size, 1)

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters. This comes into effect only for the lstm mode.
        """
        if self.mode == "lstm":
            self.lstm.reset_parameters()
            self.att.reset_parameters()

    def forward(self, feat_list):
        r"""

        Description
        -----------
        Aggregate output representations across multiple GNN layers.

        Parameters
        ----------
        feat_list : list[Tensor]
            feat_list[i] is the output representations of a GNN layer.

        Returns
        -------
        Tensor
            The aggregated representations.
        """
        if self.mode == "cat":
            return th.cat(feat_list, dim=-1)
        elif self.mode == "max":
            return th.stack(feat_list, dim=-1).max(dim=-1)[0]
        else:
            # LSTM
            stacked_feat_list = th.stack(
                feat_list, dim=1
            )  # (N, num_layers, in_feats)
            alpha, _ = self.lstm(stacked_feat_list)
            alpha = self.att(alpha).squeeze(-1)  # (N, num_layers)
            alpha = th.softmax(alpha, dim=-1)
            return (stacked_feat_list * alpha.unsqueeze(-1)).sum(dim=1)


class LabelPropagation(nn.Module):
    r"""Label Propagation from `Learning from Labeled and Unlabeled Data with Label
    Propagation <http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf>`__

    .. math::

        \mathbf{Y}^{(t+1)} = \alpha \tilde{A} \mathbf{Y}^{(t)} + (1 - \alpha) \mathbf{Y}^{(0)}

    where unlabeled data is initially set to zero and inferred from labeled data via
    propagation. :math:`\alpha` is a weight parameter for balancing between updated labels
    and initial labels. :math:`\tilde{A}` denotes the normalized adjacency matrix.

    Parameters
    ----------
    k: int
        The number of propagation steps.
    alpha : float
        The :math:`\alpha` coefficient in range [0, 1].
    norm_type : str, optional
        The type of normalization applied to the adjacency matrix, must be one of the
        following choices:

        * ``row``: row-normalized adjacency as :math:`D^{-1}A`

        * ``sym``: symmetrically normalized adjacency as :math:`D^{-1/2}AD^{-1/2}`

        Default: 'sym'.
    clamp : bool, optional
        A bool flag to indicate whether to clamp the labels to [0, 1] after propagation.
        Default: True.
    normalize: bool, optional
        A bool flag to indicate whether to apply row-normalization after propagation.
        Default: False.
    reset : bool, optional
        A bool flag to indicate whether to reset the known labels after each
        propagation step. Default: False.

    Examples
    --------
    >>> import torch
    >>> import dgl
    >>> from dgl.nn import LabelPropagation

    >>> label_propagation = LabelPropagation(k=5, alpha=0.5, clamp=False, normalize=True)
    >>> g = dgl.rand_graph(5, 10)
    >>> labels = torch.tensor([0, 2, 1, 3, 0]).long()
    >>> mask = torch.tensor([0, 1, 1, 1, 0]).bool()
    >>> new_labels = label_propagation(g, labels, mask)
    """

    def __init__(
        self,
        k,
        alpha,
        norm_type="sym",
        clamp=True,
        normalize=False,
        reset=False,
    ):
        super(LabelPropagation, self).__init__()
        self.k = k
        self.alpha = alpha
        self.norm_type = norm_type
        self.clamp = clamp
        self.normalize = normalize
        self.reset = reset

    def forward(self, g, labels, mask=None):
        r"""Compute the label propagation process.

        Parameters
        ----------
        g : DGLGraph
            The input graph.
        labels : torch.Tensor
            The input node labels. There are three cases supported.

            * A LongTensor of shape :math:`(N, 1)` or :math:`(N,)` for node class labels in
              multiclass classification, where :math:`N` is the number of nodes.
            * A LongTensor of shape :math:`(N, C)` for one-hot encoding of node class labels
              in multiclass classification, where :math:`C` is the number of classes.
            * A LongTensor of shape :math:`(N, L)` for node labels in multilabel binary
              classification, where :math:`L` is the number of labels.
        mask : torch.Tensor
            The bool indicators of shape :math:`(N,)` with True denoting labeled nodes.
            Default: None, indicating all nodes are labeled.

        Returns
        -------
        torch.Tensor
            The propagated node labels of shape :math:`(N, D)` with float type, where :math:`D`
            is the number of classes or labels.
        """
        with g.local_scope():
            # multi-label / multi-class
            if len(labels.size()) > 1 and labels.size(1) > 1:
                labels = labels.to(th.float32)
            # single-label multi-class
            else:
                labels = F.one_hot(labels.view(-1)).to(th.float32)

            y = labels
            if mask is not None:
                y = th.zeros_like(labels)
                y[mask] = labels[mask]

            init = (1 - self.alpha) * y
            in_degs = g.in_degrees().float().clamp(min=1)
            out_degs = g.out_degrees().float().clamp(min=1)
            if self.norm_type == "sym":
                norm_i = th.pow(in_degs, -0.5).to(labels.device).unsqueeze(1)
                norm_j = th.pow(out_degs, -0.5).to(labels.device).unsqueeze(1)
            elif self.norm_type == "row":
                norm_i = th.pow(in_degs, -1.0).to(labels.device).unsqueeze(1)
            else:
                raise ValueError(
                    f"Expect norm_type to be 'sym' or 'row', got {self.norm_type}"
                )

            for _ in range(self.k):
                g.ndata["h"] = y * norm_j if self.norm_type == "sym" else y
                g.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                y = init + self.alpha * g.ndata["h"] * norm_i

                if self.clamp:
                    y = y.clamp_(0.0, 1.0)
                if self.normalize:
                    y = F.normalize(y, p=1)
                if self.reset:
                    y[mask] = labels[mask]

            return y
