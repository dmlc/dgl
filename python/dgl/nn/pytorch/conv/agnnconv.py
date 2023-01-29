"""Torch Module for Attention-based Graph Neural Network layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from torch.nn import functional as F

from .... import function as fn
from ....base import DGLError
from ....utils import expand_as_pair
from ...functional import edge_softmax


class AGNNConv(nn.Module):
    r"""Attention-based Graph Neural Network layer from `Attention-based Graph Neural Network for
    Semi-Supervised Learning <https://arxiv.org/abs/1803.03735>`__

    .. math::
        H^{l+1} = P H^{l}

    where :math:`P` is computed as:

    .. math::
        P_{ij} = \mathrm{softmax}_i ( \beta \cdot \cos(h_i^l, h_j^l))

    where :math:`\beta` is a single scalar parameter.

    Parameters
    ----------
    init_beta : float, optional
        The :math:`\beta` in the formula, a single scalar parameter.
    learn_beta : bool, optional
        If True, :math:`\beta` will be learnable parameter.
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

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import AGNNConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = th.ones(6, 10)
    >>> conv = AGNNConv()
    >>> res = conv(g, feat)
    >>> res
    tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]],
        grad_fn=<BinaryReduceBackward>)
    """

    def __init__(
        self, init_beta=1.0, learn_beta=True, allow_zero_in_degree=False
    ):
        super(AGNNConv, self).__init__()
        self._allow_zero_in_degree = allow_zero_in_degree
        if learn_beta:
            self.beta = nn.Parameter(th.Tensor([init_beta]))
        else:
            self.register_buffer("beta", th.Tensor([init_beta]))

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

    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute AGNN layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, *)` :math:`N` is the
            number of nodes, and :math:`*` could be of any shape.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, *)` and :math:`(N_{out}, *)`, the :math:`*` in the later
            tensor must equal the previous one.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *)` where :math:`*`
            should be the same as input shape.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
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

            feat_src, feat_dst = expand_as_pair(feat, graph)

            graph.srcdata["h"] = feat_src
            graph.srcdata["norm_h"] = F.normalize(feat_src, p=2, dim=-1)
            if isinstance(feat, tuple) or graph.is_block:
                graph.dstdata["norm_h"] = F.normalize(feat_dst, p=2, dim=-1)
            # compute cosine distance
            graph.apply_edges(fn.u_dot_v("norm_h", "norm_h", "cos"))
            cos = graph.edata.pop("cos")
            e = self.beta * cos
            graph.edata["p"] = edge_softmax(graph, e)
            graph.update_all(fn.u_mul_e("h", "p", "m"), fn.sum("m", "h"))
            return graph.dstdata.pop("h")
