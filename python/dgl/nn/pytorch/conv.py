"""Torch modules for graph convolutions."""
import torch.nn as nn

from ... import function as fn
from ...base import ALL, is_all

__all__ = ['GraphConv']

class GraphConv(nn.Module):
    """Apply graph convolution over an input signal.

    Graph convolution is introduced in `<https://arxiv.org/abs/1609.02907>`__
    and can be described as below:

    .. math::

        h_i^{(l+1)} = \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ij}}(W^{(l)}h_j^{(l)} + b^{(l)})

    where :math:`\mathcal{N}(i)` is the neighbor set of node :math:`i`. :math:`c_{ij}` is equal
    to the produce of the square root of node degrees:
    :math:`\sqrt{|\mathcal{N}(i)|}\sart{|\mathcal{N}(j)|}`.

    Paramters
    ---------
    in_feats : int
        Number of input features.
    out_feats : int
        Number of output features.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: True.
    field_name : str, optional
        The temporary field name used to compute message passing. Default: "_tmp_fld".

    Attributes
    ----------
    linear : torch.nn.Linear
        The linear layer used in this module.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 bias=True,
                 field_name="_tmp_fld"):
        super(GraphConv, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self._field_name = field_name
        self._msg_name = "_tmp_msg"

    def forward(self, feat, graph, edges=None):
        """Compute graph convolution.

        The graph convolution module can be applied on the full graph or on a given edges.

        - If applied on the full graph, all the nodes will aggregate features
          from their neighbors.
        - If applied on edges, the dst nodes will aggregate features from the src nodes.

        Notes
        -----
            * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
              dimensions, :math:`N` is the number of nodes.
            * Output shape: :math:`(N, *, \text{in_feats})` where all but the last dimension are
              the same shape as the input.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature
        graph : DGLGraph
            The graph.
        edges : edges, optional
            Edges can be a pair of endpoint nodes (u, v), or a
            tensor of edge ids. If not specified, the module will be applied
            on the full graph.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        graph.ndata[self._field_name] = feat
        if edges is None:
            # full graph
            g.update_all(fn.copy_src(src=self._field_name, out=self._msg_name),
                         fn.sum(msg=self._msg_name, out=self._field_name),
                         self._update_func)
        else:
            # on edges
            g.send_and_recv(edges,
                            fn.copy_src(src=self._field_name, out=self._msg_name),
                            fn.sum(msg=self._msg_name, out=self._field_name),
                            self._update_func)
        return graph.ndata.pop(self._field_name)
    
    def _update_func(self, nodes):
        """Internal node update function."""
        feat = self.linear(nodes.data[self._field_name])
        return {self._field_name : feat}
