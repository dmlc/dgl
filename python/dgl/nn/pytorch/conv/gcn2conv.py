"""Torch Module for Graph Convolutional Network via Initial residual
    and Identity mapping (GCNII) layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import math

import torch as th
from torch import nn

from .... import function as fn
from ....base import DGLError
from .graphconv import EdgeWeightNorm


class GCN2Conv(nn.Module):
    r"""Graph Convolutional Network via Initial residual
    and Identity mapping (GCNII) from `Simple and Deep Graph Convolutional
    Networks <https://arxiv.org/abs/2007.02133>`__

    It is mathematically is defined as follows:

    .. math::

        \mathbf{h}^{(l+1)} =\left( (1 - \alpha)(\mathbf{D}^{-1/2} \mathbf{\hat{A}}
        \mathbf{D}^{-1/2})\mathbf{h}^{(l)} + \alpha {\mathbf{h}^{(0)}} \right)
        \left( (1 - \beta_l) \mathbf{I} + \beta_l \mathbf{W} \right)

    where :math:`\mathbf{\hat{A}}` is the adjacency matrix with self-loops,
    :math:`\mathbf{D}_{ii} = \sum_{j=0} \mathbf{A}_{ij}` is its diagonal degree matrix,
    :math:`\mathbf{h}^{(0)}` is the initial node features,
    :math:`\mathbf{h}^{(l)}` is the feature of layer :math:`l`,
    :math:`\alpha` is the fraction of initial node features, and
    :math:`\beta_l` is the hyperparameter to tune the strength of identity mapping.
    It is defined by :math:`\beta_l = \log(\frac{\lambda}{l}+1)\approx\frac{\lambda}{l}`,
    where :math:`\lambda` is a hyperparameter. :math:`\beta` ensures that the decay of
    the weight matrix adaptively increases as we stack more layers.

    Parameters
    ----------
    in_feats : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    layer : int
        the index of current layer.
    alpha : float
        The fraction of the initial input features. Default: ``0.1``
    lambda_ : float
        The hyperparameter to ensure the decay of the weight matrix
        adaptively increases. Default: ``1``
    project_initial_features : bool
        Whether to share a weight matrix between initial features and
        smoothed features. Default: ``True``
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Default: ``False``.

    Note
    ----
    Zero in-degree nodes will lead to invalid output value. This is because no message
    will be passed to those nodes, the aggregation function will be appied on empty input.
    A common practice to avoid this is to add a self-loop for each node in the graph if
    it is homogeneous, which can be achieved by:

    >>> g = ... # a DGLGraph
    >>> g = dgl.add_self_loop(g)

    Calling ``add_self_loop`` will not work for some graphs, for example, heterogeneous graph
    since the edge type can not be decided for self_loop edges. Set ``allow_zero_in_degree``
    to ``True`` for those cases to unblock the code and handle zero-in-degree nodes manually.
    A common practise to handle this is to filter out the nodes with zero-in-degree when use
    after conv.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import GCN2Conv

    >>> # Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 3)
    >>> g = dgl.add_self_loop(g)
    >>> conv1 = GCN2Conv(3, layer=1, alpha=0.5, \
    ...         project_initial_features=True, allow_zero_in_degree=True)
    >>> conv2 = GCN2Conv(3, layer=2, alpha=0.5, \
    ...         project_initial_features=True, allow_zero_in_degree=True)
    >>> res = feat
    >>> res = conv1(g, res, feat)
    >>> res = conv2(g, res, feat)
    >>> print(res)
    tensor([[1.3803, 3.3191, 2.9572],
            [1.3803, 3.3191, 2.9572],
            [1.3803, 3.3191, 2.9572],
            [1.4770, 3.8326, 3.2451],
            [1.3623, 3.2102, 2.8679],
            [1.3803, 3.3191, 2.9572]], grad_fn=<AddBackward0>)

    """

    def __init__(
        self,
        in_feats,
        layer,
        alpha=0.1,
        lambda_=1,
        project_initial_features=True,
        allow_zero_in_degree=False,
        bias=True,
        activation=None,
    ):
        super().__init__()

        self._in_feats = in_feats
        self._project_initial_features = project_initial_features

        self.alpha = alpha
        self.beta = math.log(lambda_ / layer + 1)

        self._bias = bias
        self._activation = activation
        self._allow_zero_in_degree = allow_zero_in_degree

        self.weight1 = nn.Parameter(th.Tensor(self._in_feats, self._in_feats))

        if self._project_initial_features:
            self.register_parameter("weight2", None)
        else:
            self.weight2 = nn.Parameter(
                th.Tensor(self._in_feats, self._in_feats)
            )

        if self._bias:
            self.bias = nn.Parameter(th.Tensor(self._in_feats))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        """
        nn.init.normal_(self.weight1)
        if not self._project_initial_features:
            nn.init.normal_(self.weight2)
        if self._bias:
            nn.init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, feat_0, edge_weight=None):
        r"""

        Description
        -----------
        Compute graph convolution.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is the size of input feature and :math:`N` is the number of nodes.
        feat_0 : torch.Tensor
            The initial feature of shape :math:`(N, D_{in})`
        edge_weight: torch.Tensor, optional
            edge_weight to use in the message passing process. This is equivalent to
            using weighted adjacency matrix in the equation above, and
            :math:`\tilde{D}^{-1/2}\tilde{A} \tilde{D}^{-1/2}`
            is based on :class:`dgl.nn.pytorch.conv.graphconv.EdgeWeightNorm`.


        Returns
        -------
        torch.Tensor
            The output feature

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

        Note
        ----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: :math:`(\text{in_feats}, \text{out_feats})`.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            # normalize  to get smoothed representation
            if edge_weight is None:
                degs = graph.in_degrees().to(feat).clamp(min=1)
                norm = th.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)
            else:
                edge_weight = EdgeWeightNorm("both")(graph, edge_weight)

            if edge_weight is None:
                feat = feat * norm
            graph.ndata["h"] = feat
            msg_func = fn.copy_u("h", "m")
            if edge_weight is not None:
                graph.edata["_edge_weight"] = edge_weight
                msg_func = fn.u_mul_e("h", "_edge_weight", "m")
            graph.update_all(msg_func, fn.sum("m", "h"))
            feat = graph.ndata.pop("h")
            if edge_weight is None:
                feat = feat * norm
            # scale
            feat = feat * (1 - self.alpha)

            # initial residual connection to the first layer
            feat_0 = feat_0[: feat.size(0)] * self.alpha
            feat_sum = feat + feat_0

            if self._project_initial_features:
                feat_proj_sum = feat_sum @ self.weight1
            else:
                feat_proj_sum = feat @ self.weight1 + feat_0 @ self.weight2

            rst = (1 - self.beta) * feat_sum + self.beta * feat_proj_sum

            if self._bias:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = "in={_in_feats}"
        summary += ", share_weight={_share_weights}, alpha={alpha}, beta={beta}"
        if "self._bias" in self.__dict__:
            summary += ", bias={bias}"
        if "self._activation" in self.__dict__:
            summary += ", activation={_activation}"

        return summary.format(**self.__dict__)
