"""Torch modules for graph global pooling."""
# pylint: disable= no-member, arguments-differ, invalid-name, W0235
import numpy as np
import torch as th
import torch.nn as nn

from ...backend import pytorch as F
from ...base import dgl_warning
from ...readout import (
    broadcast_nodes,
    max_nodes,
    mean_nodes,
    softmax_nodes,
    sum_nodes,
    topk_nodes,
)

__all__ = [
    "SumPooling",
    "AvgPooling",
    "MaxPooling",
    "SortPooling",
    "GlobalAttentionPooling",
    "Set2Set",
    "SetTransformerEncoder",
    "SetTransformerDecoder",
    "WeightAndSum",
]


class SumPooling(nn.Module):
    r"""Apply sum pooling over the nodes in a graph.

    .. math::
        r^{(i)} = \sum_{k=1}^{N_i} x^{(i)}_k

    Notes
    -----
        Input: Could be one graph, or a batch of graphs. If using a batch of graphs,
        make sure nodes in all graphs have the same feature size, and concatenate
        nodes' feature together as the input.

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import SumPooling
    >>>
    >>> g1 = dgl.rand_graph(3, 4)  # g1 is a random graph with 3 nodes and 4 edges
    >>> g1_node_feats = th.rand(3, 5)  # feature size is 5
    >>> g1_node_feats
    tensor([[0.8948, 0.0699, 0.9137, 0.7567, 0.3637],
            [0.8137, 0.8938, 0.8377, 0.4249, 0.6118],
            [0.5197, 0.9030, 0.6825, 0.5725, 0.4755]])
    >>>
    >>> g2 = dgl.rand_graph(4, 6)  # g2 is a random graph with 4 nodes and 6 edges
    >>> g2_node_feats = th.rand(4, 5)  # feature size is 5
    >>> g2_node_feats
    tensor([[0.2053, 0.2426, 0.4111, 0.9028, 0.5658],
            [0.5278, 0.6365, 0.9990, 0.2351, 0.8945],
            [0.3134, 0.0580, 0.4349, 0.7949, 0.3891],
            [0.0142, 0.2709, 0.3330, 0.8521, 0.6925]])
    >>>
    >>> sumpool = SumPooling()  # create a sum pooling layer

    Case 1: Input a single graph

    >>> sumpool(g1, g1_node_feats)
    tensor([[2.2282, 1.8667, 2.4338, 1.7540, 1.4511]])

    Case 2: Input a batch of graphs

    Build a batch of DGL graphs and concatenate all graphs' node features into one tensor.

    >>> batch_g = dgl.batch([g1, g2])
    >>> batch_f = th.cat([g1_node_feats, g2_node_feats])
    >>>
    >>> sumpool(batch_g, batch_f)
    tensor([[2.2282, 1.8667, 2.4338, 1.7540, 1.4511],
            [1.0608, 1.2080, 2.1780, 2.7849, 2.5420]])
    """

    def __init__(self):
        super(SumPooling, self).__init__()

    def forward(self, graph, feat):
        r"""

        Compute sum pooling.

        Parameters
        ----------
        graph : DGLGraph
            a DGLGraph or a batch of DGLGraphs
        feat : torch.Tensor
            The input feature with shape :math:`(N, D)`, where :math:`N` is the number
            of nodes in the graph, and :math:`D` means the size of features.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, D)`, where :math:`B` refers to the
            batch size of input graphs.
        """
        with graph.local_scope():
            graph.ndata["h"] = feat
            readout = sum_nodes(graph, "h")
            return readout


class AvgPooling(nn.Module):
    r"""Apply average pooling over the nodes in a graph.

    .. math::
        r^{(i)} = \frac{1}{N_i}\sum_{k=1}^{N_i} x^{(i)}_k

    Notes
    -----
        Input: Could be one graph, or a batch of graphs. If using a batch of graphs,
        make sure nodes in all graphs have the same feature size, and concatenate
        nodes' feature together as the input.

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import AvgPooling
    >>>
    >>> g1 = dgl.rand_graph(3, 4)  # g1 is a random graph with 3 nodes and 4 edges
    >>> g1_node_feats = th.rand(3, 5)  # feature size is 5
    >>> g1_node_feats
    tensor([[0.8948, 0.0699, 0.9137, 0.7567, 0.3637],
            [0.8137, 0.8938, 0.8377, 0.4249, 0.6118],
            [0.5197, 0.9030, 0.6825, 0.5725, 0.4755]])
    >>>
    >>> g2 = dgl.rand_graph(4, 6)  # g2 is a random graph with 4 nodes and 6 edges
    >>> g2_node_feats = th.rand(4, 5)  # feature size is 5
    >>> g2_node_feats
    tensor([[0.2053, 0.2426, 0.4111, 0.9028, 0.5658],
            [0.5278, 0.6365, 0.9990, 0.2351, 0.8945],
            [0.3134, 0.0580, 0.4349, 0.7949, 0.3891],
            [0.0142, 0.2709, 0.3330, 0.8521, 0.6925]])
    >>>
    >>> avgpool = AvgPooling()  # create an average pooling layer

    Case 1: Input single graph

    >>> avgpool(g1, g1_node_feats)
    tensor([[0.7427, 0.6222, 0.8113, 0.5847, 0.4837]])

    Case 2: Input a batch of graphs

    Build a batch of DGL graphs and concatenate all graphs' note features into one tensor.

    >>> batch_g = dgl.batch([g1, g2])
    >>> batch_f = th.cat([g1_node_feats, g2_node_feats])
    >>>
    >>> avgpool(batch_g, batch_f)
    tensor([[0.7427, 0.6222, 0.8113, 0.5847, 0.4837],
            [0.2652, 0.3020, 0.5445, 0.6962, 0.6355]])
    """

    def __init__(self):
        super(AvgPooling, self).__init__()

    def forward(self, graph, feat):
        r"""

        Compute average pooling.

        Parameters
        ----------
        graph : DGLGraph
            A DGLGraph or a batch of DGLGraphs.
        feat : torch.Tensor
            The input feature with shape :math:`(N, D)`, where :math:`N` is the number
            of nodes in the graph, and :math:`D` means the size of features.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, D)`, where
            :math:`B` refers to the batch size of input graphs.
        """
        with graph.local_scope():
            graph.ndata["h"] = feat
            readout = mean_nodes(graph, "h")
            return readout


class MaxPooling(nn.Module):
    r"""Apply max pooling over the nodes in a graph.

    .. math::
        r^{(i)} = \max_{k=1}^{N_i}\left( x^{(i)}_k \right)

    Notes
    -----
        Input: Could be one graph, or a batch of graphs. If using a batch of graphs,
        make sure nodes in all graphs have the same feature size, and concatenate
        nodes' feature together as the input.

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import MaxPooling
    >>>
    >>> g1 = dgl.rand_graph(3, 4)  # g1 is a random graph with 3 nodes and 4 edges
    >>> g1_node_feats = th.rand(3, 5)  # feature size is 5
    >>> g1_node_feats
    tensor([[0.8948, 0.0699, 0.9137, 0.7567, 0.3637],
            [0.8137, 0.8938, 0.8377, 0.4249, 0.6118],
            [0.5197, 0.9030, 0.6825, 0.5725, 0.4755]])
    >>>
    >>> g2 = dgl.rand_graph(4, 6)  # g2 is a random graph with 4 nodes and 6 edges
    >>> g2_node_feats = th.rand(4, 5)  # feature size is 5
    >>> g2_node_feats
    tensor([[0.2053, 0.2426, 0.4111, 0.9028, 0.5658],
            [0.5278, 0.6365, 0.9990, 0.2351, 0.8945],
            [0.3134, 0.0580, 0.4349, 0.7949, 0.3891],
            [0.0142, 0.2709, 0.3330, 0.8521, 0.6925]])
    >>>
    >>> maxpool = MaxPooling()  # create a max pooling layer

    Case 1: Input a single graph

    >>> maxpool(g1, g1_node_feats)
    tensor([[0.8948, 0.9030, 0.9137, 0.7567, 0.6118]])

    Case 2: Input a batch of graphs

    Build a batch of DGL graphs and concatenate all graphs' node features into one tensor.

    >>> batch_g = dgl.batch([g1, g2])
    >>> batch_f = th.cat([g1_node_feats, g2_node_feats])
    >>>
    >>> maxpool(batch_g, batch_f)
    tensor([[0.8948, 0.9030, 0.9137, 0.7567, 0.6118],
            [0.5278, 0.6365, 0.9990, 0.9028, 0.8945]])
    """

    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, graph, feat):
        r"""Compute max pooling.

        Parameters
        ----------
        graph : DGLGraph
            A DGLGraph or a batch of DGLGraphs.
        feat : torch.Tensor
            The input feature with shape :math:`(N, *)`, where
            :math:`N` is the number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, *)`, where
            :math:`B` refers to the batch size.
        """
        with graph.local_scope():
            graph.ndata["h"] = feat
            readout = max_nodes(graph, "h")
            return readout


class SortPooling(nn.Module):
    r"""Sort Pooling from `An End-to-End Deep Learning Architecture for Graph Classification
    <https://www.cse.wustl.edu/~ychen/public/DGCNN.pdf>`__

    It first sorts the node features in ascending order along the feature dimension,
    and selects the sorted features of top-k nodes (ranked by the largest value of each node).

    Parameters
    ----------
    k : int
        The number of nodes to hold for each graph.

    Notes
    -----
        Input: Could be one graph, or a batch of graphs. If using a batch of graphs,
        make sure nodes in all graphs have the same feature size, and concatenate
        nodes' feature together as the input.

    Examples
    --------

    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import SortPooling
    >>>
    >>> g1 = dgl.rand_graph(3, 4)  # g1 is a random graph with 3 nodes and 4 edges
    >>> g1_node_feats = th.rand(3, 5)  # feature size is 5
    >>> g1_node_feats
    tensor([[0.8948, 0.0699, 0.9137, 0.7567, 0.3637],
            [0.8137, 0.8938, 0.8377, 0.4249, 0.6118],
            [0.5197, 0.9030, 0.6825, 0.5725, 0.4755]])
    >>>
    >>> g2 = dgl.rand_graph(4, 6)  # g2 is a random graph with 4 nodes and 6 edges
    >>> g2_node_feats = th.rand(4, 5)  # feature size is 5
    >>> g2_node_feats
    tensor([[0.2053, 0.2426, 0.4111, 0.9028, 0.5658],
            [0.5278, 0.6365, 0.9990, 0.2351, 0.8945],
            [0.3134, 0.0580, 0.4349, 0.7949, 0.3891],
            [0.0142, 0.2709, 0.3330, 0.8521, 0.6925]])
    >>>
    >>> sortpool = SortPooling(k=2)  # create a sort pooling layer

    Case 1: Input a single graph

    >>> sortpool(g1, g1_node_feats)
    tensor([[0.0699, 0.3637, 0.7567, 0.8948, 0.9137, 0.4755, 0.5197, 0.5725, 0.6825,
             0.9030]])

    Case 2: Input a batch of graphs

    Build a batch of DGL graphs and concatenate all graphs' node features into one tensor.

    >>> batch_g = dgl.batch([g1, g2])
    >>> batch_f = th.cat([g1_node_feats, g2_node_feats])
    >>>
    >>> sortpool(batch_g, batch_f)
    tensor([[0.0699, 0.3637, 0.7567, 0.8948, 0.9137, 0.4755, 0.5197, 0.5725, 0.6825,
             0.9030],
            [0.2351, 0.5278, 0.6365, 0.8945, 0.9990, 0.2053, 0.2426, 0.4111, 0.5658,
             0.9028]])
    """

    def __init__(self, k):
        super(SortPooling, self).__init__()
        self.k = k

    def forward(self, graph, feat):
        r"""

        Compute sort pooling.

        Parameters
        ----------
        graph : DGLGraph
            A DGLGraph or a batch of DGLGraphs.
        feat : torch.Tensor
            The input node feature with shape :math:`(N, D)`, where :math:`N` is the
            number of nodes in the graph, and :math:`D` means the size of features.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, k * D)`, where :math:`B` refers
            to the batch size of input graphs.
        """
        with graph.local_scope():
            # Sort the feature of each node in ascending order.
            feat, _ = feat.sort(dim=-1)
            graph.ndata["h"] = feat
            # Sort nodes according to their last features.
            ret = topk_nodes(graph, "h", self.k, sortby=-1)[0].view(
                -1, self.k * feat.shape[-1]
            )
            return ret


class GlobalAttentionPooling(nn.Module):
    r"""Global Attention Pooling from `Gated Graph Sequence Neural Networks
    <https://arxiv.org/abs/1511.05493>`__

    .. math::
        r^{(i)} = \sum_{k=1}^{N_i}\mathrm{softmax}\left(f_{gate}
        \left(x^{(i)}_k\right)\right) f_{feat}\left(x^{(i)}_k\right)

    Parameters
    ----------
    gate_nn : torch.nn.Module
        A neural network that computes attention scores for each feature.
    feat_nn : torch.nn.Module, optional
        A neural network applied to each feature before combining them with attention
        scores.

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import GlobalAttentionPooling
    >>>
    >>> g1 = dgl.rand_graph(3, 4)  # g1 is a random graph with 3 nodes and 4 edges
    >>> g1_node_feats = th.rand(3, 5)  # feature size is 5
    >>> g1_node_feats
    tensor([[0.8948, 0.0699, 0.9137, 0.7567, 0.3637],
            [0.8137, 0.8938, 0.8377, 0.4249, 0.6118],
            [0.5197, 0.9030, 0.6825, 0.5725, 0.4755]])
    >>>
    >>> g2 = dgl.rand_graph(4, 6)  # g2 is a random graph with 4 nodes and 6 edges
    >>> g2_node_feats = th.rand(4, 5)  # feature size is 5
    >>> g2_node_feats
    tensor([[0.2053, 0.2426, 0.4111, 0.9028, 0.5658],
            [0.5278, 0.6365, 0.9990, 0.2351, 0.8945],
            [0.3134, 0.0580, 0.4349, 0.7949, 0.3891],
            [0.0142, 0.2709, 0.3330, 0.8521, 0.6925]])
    >>>
    >>> gate_nn = th.nn.Linear(5, 1)  # the gate layer that maps node feature to scalar
    >>> gap = GlobalAttentionPooling(gate_nn)  # create a Global Attention Pooling layer

    Case 1: Input a single graph

    >>> gap(g1, g1_node_feats)
    tensor([[0.7410, 0.6032, 0.8111, 0.5942, 0.4762]],
           grad_fn=<SegmentReduceBackward>)

    Case 2: Input a batch of graphs

    Build a batch of DGL graphs and concatenate all graphs' node features into one tensor.

    >>> batch_g = dgl.batch([g1, g2])
    >>> batch_f = th.cat([g1_node_feats, g2_node_feats], 0)
    >>>
    >>> gap(batch_g, batch_f)
    tensor([[0.7410, 0.6032, 0.8111, 0.5942, 0.4762],
            [0.2417, 0.2743, 0.5054, 0.7356, 0.6146]],
           grad_fn=<SegmentReduceBackward>)
    Notes
    -----
    See our `GGNN example <https://github.com/dmlc/dgl/tree/master/examples/pytorch/ggnn>`_
    on how to use GatedGraphConv and GlobalAttentionPooling layer to build a Graph Neural
    Networks that can solve Soduku.
    """

    def __init__(self, gate_nn, feat_nn=None):
        super(GlobalAttentionPooling, self).__init__()
        self.gate_nn = gate_nn
        self.feat_nn = feat_nn

    def forward(self, graph, feat, get_attention=False):
        r"""

        Compute global attention pooling.

        Parameters
        ----------
        graph : DGLGraph
            A DGLGraph or a batch of DGLGraphs.
        feat : torch.Tensor
            The input node feature with shape :math:`(N, D)` where :math:`N` is the
            number of nodes in the graph, and :math:`D` means the size of features.
        get_attention : bool, optional
            Whether to return the attention values from gate_nn. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, D)`, where :math:`B` refers
            to the batch size.
        torch.Tensor, optional
            The attention values of shape :math:`(N, 1)`, where :math:`N` is the number of
            nodes in the graph. This is returned only when :attr:`get_attention` is ``True``.
        """
        with graph.local_scope():
            gate = self.gate_nn(feat)
            assert (
                gate.shape[-1] == 1
            ), "The output of gate_nn should have size 1 at the last axis."
            feat = self.feat_nn(feat) if self.feat_nn else feat

            graph.ndata["gate"] = gate
            gate = softmax_nodes(graph, "gate")
            graph.ndata.pop("gate")

            graph.ndata["r"] = feat * gate
            readout = sum_nodes(graph, "r")
            graph.ndata.pop("r")

            if get_attention:
                return readout, gate
            else:
                return readout


class Set2Set(nn.Module):
    r"""Set2Set operator from `Order Matters: Sequence to sequence for sets
    <https://arxiv.org/pdf/1511.06391.pdf>`__

    For each individual graph in the batch, set2set computes

    .. math::
        q_t &= \mathrm{LSTM} (q^*_{t-1})

        \alpha_{i,t} &= \mathrm{softmax}(x_i \cdot q_t)

        r_t &= \sum_{i=1}^N \alpha_{i,t} x_i

        q^*_t &= q_t \Vert r_t

    for this graph.

    Parameters
    ----------
    input_dim : int
        The size of each input sample.
    n_iters : int
        The number of iterations.
    n_layers : int
        The number of recurrent layers.

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import Set2Set
    >>>
    >>> g1 = dgl.rand_graph(3, 4)  # g1 is a random graph with 3 nodes and 4 edges
    >>> g1_node_feats = th.rand(3, 5)  # feature size is 5
    >>> g1_node_feats
    tensor([[0.8948, 0.0699, 0.9137, 0.7567, 0.3637],
            [0.8137, 0.8938, 0.8377, 0.4249, 0.6118],
            [0.5197, 0.9030, 0.6825, 0.5725, 0.4755]])
    >>>
    >>> g2 = dgl.rand_graph(4, 6)  # g2 is a random graph with 4 nodes and 6 edges
    >>> g2_node_feats = th.rand(4, 5)  # feature size is 5
    >>> g2_node_feats
    tensor([[0.2053, 0.2426, 0.4111, 0.9028, 0.5658],
            [0.5278, 0.6365, 0.9990, 0.2351, 0.8945],
            [0.3134, 0.0580, 0.4349, 0.7949, 0.3891],
            [0.0142, 0.2709, 0.3330, 0.8521, 0.6925]])
    >>>
    >>> s2s = Set2Set(5, 2, 1)  # create a Set2Set layer(n_iters=2, n_layers=1)

    Case 1: Input a single graph

    >>> s2s(g1, g1_node_feats)
        tensor([[-0.0235, -0.2291,  0.2654,  0.0376,  0.1349,  0.7560,  0.5822,  0.8199,
                  0.5960,  0.4760]], grad_fn=<CatBackward>)

    Case 2: Input a batch of graphs

    Build a batch of DGL graphs and concatenate all graphs' node features into one tensor.

    >>> batch_g = dgl.batch([g1, g2])
    >>> batch_f = th.cat([g1_node_feats, g2_node_feats], 0)
    >>>
    >>> s2s(batch_g, batch_f)
    tensor([[-0.0235, -0.2291,  0.2654,  0.0376,  0.1349,  0.7560,  0.5822,  0.8199,
              0.5960,  0.4760],
            [-0.0483, -0.2010,  0.2324,  0.0145,  0.1361,  0.2703,  0.3078,  0.5529,
              0.6876,  0.6399]], grad_fn=<CatBackward>)

    Notes
    -----
    Set2Set is widely used in molecular property predictions, see
    `dgl-lifesci's MPNN example <https://github.com/awslabs/dgl-lifesci/blob/
    ecd95c905479ec048097777039cf9a19cfdcf223/python/dgllife/model/model_zoo/
    mpnn_predictor.py>`__
    on how to use DGL's Set2Set layer in graph property prediction applications.
    """

    def __init__(self, input_dim, n_iters, n_layers):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = th.nn.LSTM(self.output_dim, self.input_dim, n_layers)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        self.lstm.reset_parameters()

    def forward(self, graph, feat):
        r"""
        Compute set2set pooling.

        Parameters
        ----------
        graph : DGLGraph
            The input graph.
        feat : torch.Tensor
            The input feature with shape :math:`(N, D)` where  :math:`N` is the
            number of nodes in the graph, and :math:`D` means the size of features.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, D)`, where :math:`B` refers to
            the batch size, and :math:`D` means the size of features.
        """
        with graph.local_scope():
            batch_size = graph.batch_size

            h = (
                feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
                feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
            )

            q_star = feat.new_zeros(batch_size, self.output_dim)

            for _ in range(self.n_iters):
                q, h = self.lstm(q_star.unsqueeze(0), h)
                q = q.view(batch_size, self.input_dim)
                e = (feat * broadcast_nodes(graph, q)).sum(dim=-1, keepdim=True)
                graph.ndata["e"] = e
                alpha = softmax_nodes(graph, "e")
                graph.ndata["r"] = feat * alpha
                readout = sum_nodes(graph, "r")
                q_star = th.cat([q, readout], dim=-1)

            return q_star

    def extra_repr(self):
        """Set the extra representation of the module.
        which will come into effect when printing the model.
        """
        summary = "n_iters={n_iters}"
        return summary.format(**self.__dict__)


def _gen_mask(lengths_x, lengths_y, max_len_x, max_len_y):
    """Generate binary mask array for given x and y input pairs.

    Parameters
    ----------
    lengths_x : Tensor
        The int tensor indicates the segment information of x.
    lengths_y : Tensor
        The int tensor indicates the segment information of y.
    max_len_x : int
        The maximum element in lengths_x.
    max_len_y : int
        The maximum element in lengths_y.

    Returns
    -------
    Tensor
        the mask tensor with shape (batch_size, 1, max_len_x, max_len_y)
    """
    device = lengths_x.device
    # x_mask: (batch_size, max_len_x)
    x_mask = th.arange(max_len_x, device=device).unsqueeze(
        0
    ) < lengths_x.unsqueeze(1)
    # y_mask: (batch_size, max_len_y)
    y_mask = th.arange(max_len_y, device=device).unsqueeze(
        0
    ) < lengths_y.unsqueeze(1)
    # mask: (batch_size, 1, max_len_x, max_len_y)
    mask = (x_mask.unsqueeze(-1) & y_mask.unsqueeze(-2)).unsqueeze(1)
    return mask


class MultiHeadAttention(nn.Module):
    r"""Multi-Head Attention block, used in Transformer, Set Transformer and so on

    Parameters
    ----------
    d_model : int
        The feature size (input and output) in Multi-Head Attention layer.
    num_heads : int
        The number of heads.
    d_head : int
        The hidden size per head.
    d_ff : int
        The inner hidden size in the Feed-Forward Neural Network.
    dropouth : float
        The dropout rate of each sublayer.
    dropouta : float
        The dropout rate of attention heads.

    Notes
    -----
    This module was used in SetTransformer layer.
    """

    def __init__(
        self, d_model, num_heads, d_head, d_ff, dropouth=0.0, dropouta=0.0
    ):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.proj_q = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.proj_k = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.proj_v = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.proj_o = nn.Linear(num_heads * d_head, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropouth),
            nn.Linear(d_ff, d_model),
        )
        self.droph = nn.Dropout(dropouth)
        self.dropa = nn.Dropout(dropouta)
        self.norm_in = nn.LayerNorm(d_model)
        self.norm_inter = nn.LayerNorm(d_model)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mem, lengths_x, lengths_mem):
        """
        Compute multi-head self-attention.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor used to compute queries.
        mem : torch.Tensor
            The memory tensor used to compute keys and values.
        lengths_x : list
            The array of node numbers, used to segment x.
        lengths_mem : list
            The array of node numbers, used to segment mem.
        """
        batch_size = len(lengths_x)
        max_len_x = max(lengths_x)
        max_len_mem = max(lengths_mem)
        device = x.device
        lengths_x = th.as_tensor(lengths_x, dtype=th.int64, device=device)
        lengths_mem = th.as_tensor(lengths_mem, dtype=th.int64, device=device)

        queries = self.proj_q(x).view(-1, self.num_heads, self.d_head)
        keys = self.proj_k(mem).view(-1, self.num_heads, self.d_head)
        values = self.proj_v(mem).view(-1, self.num_heads, self.d_head)

        # padding to (B, max_len_x/mem, num_heads, d_head)
        queries = F.pad_packed_tensor(queries, lengths_x, 0)
        keys = F.pad_packed_tensor(keys, lengths_mem, 0)
        values = F.pad_packed_tensor(values, lengths_mem, 0)

        # attention score with shape (B, num_heads, max_len_x, max_len_mem)
        e = th.einsum("bxhd,byhd->bhxy", queries, keys)
        # normalize
        e = e / np.sqrt(self.d_head)

        # generate mask
        mask = _gen_mask(lengths_x, lengths_mem, max_len_x, max_len_mem)
        e = e.masked_fill(mask == 0, -float("inf"))

        # apply softmax
        alpha = th.softmax(e, dim=-1)
        # the following line addresses the NaN issue, see
        # https://github.com/dmlc/dgl/issues/2657
        alpha = alpha.masked_fill(mask == 0, 0.0)

        # sum of value weighted by alpha
        out = th.einsum("bhxy,byhd->bxhd", alpha, values)
        # project to output
        out = self.proj_o(
            out.contiguous().view(
                batch_size, max_len_x, self.num_heads * self.d_head
            )
        )
        # pack tensor
        out = F.pack_padded_tensor(out, lengths_x)

        # intra norm
        x = self.norm_in(x + out)

        # inter norm
        x = self.norm_inter(x + self.ffn(x))

        return x


class SetAttentionBlock(nn.Module):
    r"""SAB block from `Set Transformer: A Framework for Attention-based
    Permutation-Invariant Neural Networks <https://arxiv.org/abs/1810.00825>`__

    Parameters
    ----------
    d_model : int
        The feature size (input and output) in Multi-Head Attention layer.
    num_heads : int
        The number of heads.
    d_head : int
        The hidden size per head.
    d_ff : int
        The inner hidden size in the Feed-Forward Neural Network.
    dropouth : float
        The dropout rate of each sublayer.
    dropouta : float
        The dropout rate of attention heads.

    Notes
    -----
    This module was used in SetTransformer layer.
    """

    def __init__(
        self, d_model, num_heads, d_head, d_ff, dropouth=0.0, dropouta=0.0
    ):
        super(SetAttentionBlock, self).__init__()
        self.mha = MultiHeadAttention(
            d_model,
            num_heads,
            d_head,
            d_ff,
            dropouth=dropouth,
            dropouta=dropouta,
        )

    def forward(self, feat, lengths):
        """
        Compute a Set Attention Block.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature.
        lengths : list
            The array of node numbers, used to segment feat tensor.
        """
        return self.mha(feat, feat, lengths, lengths)


class InducedSetAttentionBlock(nn.Module):
    r"""ISAB block from `Set Transformer: A Framework for Attention-based
    Permutation-Invariant Neural Networks <https://arxiv.org/abs/1810.00825>`__

    Parameters
    ----------
    m : int
        The number of induced vectors.
    d_model : int
        The feature size (input and output) in Multi-Head Attention layer.
    num_heads : int
        The number of heads.
    d_head : int
        The hidden size per head.
    d_ff : int
        The inner hidden size in the Feed-Forward Neural Network.
    dropouth : float
        The dropout rate of each sublayer.
    dropouta : float
        The dropout rate of attention heads.

    Notes
    -----
    This module was used in SetTransformer layer.
    """

    def __init__(
        self, m, d_model, num_heads, d_head, d_ff, dropouth=0.0, dropouta=0.0
    ):
        super(InducedSetAttentionBlock, self).__init__()
        self.m = m
        if m == 1:
            dgl_warning(
                "if m is set to 1, the parameters corresponding to query and key "
                "projections would not get updated during training."
            )
        self.d_model = d_model
        self.inducing_points = nn.Parameter(th.FloatTensor(m, d_model))
        self.mha = nn.ModuleList(
            [
                MultiHeadAttention(
                    d_model,
                    num_heads,
                    d_head,
                    d_ff,
                    dropouth=dropouth,
                    dropouta=dropouta,
                )
                for _ in range(2)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        nn.init.xavier_uniform_(self.inducing_points)

    def forward(self, feat, lengths):
        """
        Compute an Induced Set Attention Block.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature.
        lengths : list
            The array of node numbers, used to segment feat tensor.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        batch_size = len(lengths)
        query = self.inducing_points.repeat(batch_size, 1)
        memory = self.mha[0](query, feat, [self.m] * batch_size, lengths)
        return self.mha[1](feat, memory, lengths, [self.m] * batch_size)

    def extra_repr(self):
        """Set the extra representation of the module.
        which will come into effect when printing the model.
        """
        shape_str = "({}, {})".format(
            self.inducing_points.shape[0], self.inducing_points.shape[1]
        )
        return "InducedVector: " + shape_str


class PMALayer(nn.Module):
    r"""Pooling by Multihead Attention from `Set Transformer: A Framework for Attention-based
    Permutation-Invariant Neural Networks <https://arxiv.org/abs/1810.00825>`__

    Parameters
    ----------
    k : int
        The number of seed vectors.
    d_model : int
        The feature size (input and output) in Multi-Head Attention layer.
    num_heads : int
        The number of heads.
    d_head : int
        The hidden size per head.
    d_ff : int
        The kernel size in FFN (Positionwise Feed-Forward Network) layer.
    dropouth : float
        The dropout rate of each sublayer.
    dropouta : float
        The dropout rate of attention heads.

    Notes
    -----
    This module was used in SetTransformer layer.
    """

    def __init__(
        self, k, d_model, num_heads, d_head, d_ff, dropouth=0.0, dropouta=0.0
    ):
        super(PMALayer, self).__init__()
        self.k = k
        if k == 1:
            dgl_warning(
                "if k is set to 1, the parameters corresponding to query and key "
                "projections would not get updated during training."
            )
        self.d_model = d_model
        self.seed_vectors = nn.Parameter(th.FloatTensor(k, d_model))
        self.mha = MultiHeadAttention(
            d_model,
            num_heads,
            d_head,
            d_ff,
            dropouth=dropouth,
            dropouta=dropouta,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropouth),
            nn.Linear(d_ff, d_model),
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        nn.init.xavier_uniform_(self.seed_vectors)

    def forward(self, feat, lengths):
        """
        Compute Pooling by Multihead Attention.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature.
        lengths : list
            The array of node numbers, used to segment feat tensor.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        batch_size = len(lengths)
        query = self.seed_vectors.repeat(batch_size, 1)
        return self.mha(query, self.ffn(feat), [self.k] * batch_size, lengths)

    def extra_repr(self):
        """Set the extra representation of the module.
        which will come into effect when printing the model.
        """
        shape_str = "({}, {})".format(
            self.seed_vectors.shape[0], self.seed_vectors.shape[1]
        )
        return "SeedVector: " + shape_str


class SetTransformerEncoder(nn.Module):
    r"""The Encoder module from `Set Transformer: A Framework for Attention-based
    Permutation-Invariant Neural Networks <https://arxiv.org/pdf/1810.00825.pdf>`__

    Parameters
    ----------
    d_model : int
        The hidden size of the model.
    n_heads : int
        The number of heads.
    d_head : int
        The hidden size of each head.
    d_ff : int
        The kernel size in FFN (Positionwise Feed-Forward Network) layer.
    n_layers : int
        The number of layers.
    block_type : str
        Building block type: 'sab' (Set Attention Block) or 'isab' (Induced
        Set Attention Block).
    m : int or None
        The number of induced vectors in ISAB Block. Set to None if block type
        is 'sab'.
    dropouth : float
        The dropout rate of each sublayer.
    dropouta : float
        The dropout rate of attention heads.

    Examples
    --------
    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import SetTransformerEncoder
    >>>
    >>> g1 = dgl.rand_graph(3, 4)  # g1 is a random graph with 3 nodes and 4 edges
    >>> g1_node_feats = th.rand(3, 5)  # feature size is 5
    >>> g1_node_feats
    tensor([[0.8948, 0.0699, 0.9137, 0.7567, 0.3637],
            [0.8137, 0.8938, 0.8377, 0.4249, 0.6118],
            [0.5197, 0.9030, 0.6825, 0.5725, 0.4755]])
    >>>
    >>> g2 = dgl.rand_graph(4, 6)  # g2 is a random graph with 4 nodes and 6 edges
    >>> g2_node_feats = th.rand(4, 5)  # feature size is 5
    >>> g2_node_feats
    tensor([[0.2053, 0.2426, 0.4111, 0.9028, 0.5658],
            [0.5278, 0.6365, 0.9990, 0.2351, 0.8945],
            [0.3134, 0.0580, 0.4349, 0.7949, 0.3891],
            [0.0142, 0.2709, 0.3330, 0.8521, 0.6925]])
    >>>
    >>> set_trans_enc = SetTransformerEncoder(5, 4, 4, 20)  # create a settrans encoder.

    Case 1: Input a single graph

    >>> set_trans_enc(g1, g1_node_feats)
    tensor([[ 0.1262, -1.9081,  0.7287,  0.1678,  0.8854],
            [-0.0634, -1.1996,  0.6955, -0.9230,  1.4904],
            [-0.9972, -0.7924,  0.6907, -0.5221,  1.6211]],
           grad_fn=<NativeLayerNormBackward>)

    Case 2: Input a batch of graphs

    Build a batch of DGL graphs and concatenate all graphs' node features into one tensor.

    >>> batch_g = dgl.batch([g1, g2])
    >>> batch_f = th.cat([g1_node_feats, g2_node_feats])
    >>>
    >>> set_trans_enc(batch_g, batch_f)
    tensor([[ 0.1262, -1.9081,  0.7287,  0.1678,  0.8854],
            [-0.0634, -1.1996,  0.6955, -0.9230,  1.4904],
            [-0.9972, -0.7924,  0.6907, -0.5221,  1.6211],
            [-0.7973, -1.3203,  0.0634,  0.5237,  1.5306],
            [-0.4497, -1.0920,  0.8470, -0.8030,  1.4977],
            [-0.4940, -1.6045,  0.2363,  0.4885,  1.3737],
            [-0.9840, -1.0913, -0.0099,  0.4653,  1.6199]],
           grad_fn=<NativeLayerNormBackward>)

    See Also
    --------
    SetTransformerDecoder

    Notes
    -----
    SetTransformerEncoder is not a readout layer, the tensor it returned is nodewise
    representation instead out graphwise representation, and the SetTransformerDecoder
    would return a graph readout tensor.
    """

    def __init__(
        self,
        d_model,
        n_heads,
        d_head,
        d_ff,
        n_layers=1,
        block_type="sab",
        m=None,
        dropouth=0.0,
        dropouta=0.0,
    ):
        super(SetTransformerEncoder, self).__init__()
        self.n_layers = n_layers
        self.block_type = block_type
        self.m = m
        layers = []
        if block_type == "isab" and m is None:
            raise KeyError(
                "The number of inducing points is not specified in ISAB block."
            )

        for _ in range(n_layers):
            if block_type == "sab":
                layers.append(
                    SetAttentionBlock(
                        d_model,
                        n_heads,
                        d_head,
                        d_ff,
                        dropouth=dropouth,
                        dropouta=dropouta,
                    )
                )
            elif block_type == "isab":
                layers.append(
                    InducedSetAttentionBlock(
                        m,
                        d_model,
                        n_heads,
                        d_head,
                        d_ff,
                        dropouth=dropouth,
                        dropouta=dropouta,
                    )
                )
            else:
                raise KeyError(
                    "Unrecognized block type {}: we only support sab/isab"
                )

        self.layers = nn.ModuleList(layers)

    def forward(self, graph, feat):
        """
        Compute the Encoder part of Set Transformer.

        Parameters
        ----------
        graph : DGLGraph
            The input graph.
        feat : torch.Tensor
            The input feature with shape :math:`(N, D)`, where :math:`N` is the
            number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(N, D)`.
        """
        lengths = graph.batch_num_nodes()
        for layer in self.layers:
            feat = layer(feat, lengths)
        return feat


class SetTransformerDecoder(nn.Module):
    r"""The Decoder module from `Set Transformer: A Framework for Attention-based
    Permutation-Invariant Neural Networks <https://arxiv.org/pdf/1810.00825.pdf>`__

    Parameters
    ----------
    d_model : int
        Hidden size of the model.
    num_heads : int
        The number of heads.
    d_head : int
        Hidden size of each head.
    d_ff : int
        Kernel size in FFN (Positionwise Feed-Forward Network) layer.
    n_layers : int
        The number of layers.
    k : int
        The number of seed vectors in PMA (Pooling by Multihead Attention) layer.
    dropouth : float
        Dropout rate of each sublayer.
    dropouta : float
        Dropout rate of attention heads.

    Examples
    --------
    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import SetTransformerDecoder
    >>>
    >>> g1 = dgl.rand_graph(3, 4)  # g1 is a random graph with 3 nodes and 4 edges
    >>> g1_node_feats = th.rand(3, 5)  # feature size is 5
    >>> g1_node_feats
    tensor([[0.8948, 0.0699, 0.9137, 0.7567, 0.3637],
            [0.8137, 0.8938, 0.8377, 0.4249, 0.6118],
            [0.5197, 0.9030, 0.6825, 0.5725, 0.4755]])
    >>>
    >>> g2 = dgl.rand_graph(4, 6)  # g2 is a random graph with 4 nodes and 6 edges
    >>> g2_node_feats = th.rand(4, 5)  # feature size is 5
    >>> g2_node_feats
    tensor([[0.2053, 0.2426, 0.4111, 0.9028, 0.5658],
            [0.5278, 0.6365, 0.9990, 0.2351, 0.8945],
            [0.3134, 0.0580, 0.4349, 0.7949, 0.3891],
            [0.0142, 0.2709, 0.3330, 0.8521, 0.6925]])
    >>>
    >>> set_trans_dec = SetTransformerDecoder(5, 4, 4, 20, 1, 3)  # define the layer

    Case 1: Input a single graph

    >>> set_trans_dec(g1, g1_node_feats)
    tensor([[-0.5538,  1.8726, -1.0470,  0.0276, -0.2994, -0.6317,  1.6754, -1.3189,
              0.2291,  0.0461, -0.4042,  0.8387, -1.7091,  1.0845,  0.1902]],
           grad_fn=<ViewBackward>)

    Case 2: Input a batch of graphs

    Build a batch of DGL graphs and concatenate all graphs' node features into one tensor.

    >>> batch_g = dgl.batch([g1, g2])
    >>> batch_f = th.cat([g1_node_feats, g2_node_feats])
    >>>
    >>> set_trans_dec(batch_g, batch_f)
    tensor([[-0.5538,  1.8726, -1.0470,  0.0276, -0.2994, -0.6317,  1.6754, -1.3189,
              0.2291,  0.0461, -0.4042,  0.8387, -1.7091,  1.0845,  0.1902],
            [-0.5511,  1.8869, -1.0156,  0.0028, -0.3231, -0.6305,  1.6845, -1.3105,
              0.2136,  0.0428, -0.3820,  0.8043, -1.7138,  1.1126,  0.1789]],
           grad_fn=<ViewBackward>)

    See Also
    --------
    SetTransformerEncoder
    """

    def __init__(
        self,
        d_model,
        num_heads,
        d_head,
        d_ff,
        n_layers,
        k,
        dropouth=0.0,
        dropouta=0.0,
    ):
        super(SetTransformerDecoder, self).__init__()
        self.n_layers = n_layers
        self.k = k
        self.d_model = d_model
        self.pma = PMALayer(
            k,
            d_model,
            num_heads,
            d_head,
            d_ff,
            dropouth=dropouth,
            dropouta=dropouta,
        )
        layers = []
        for _ in range(n_layers):
            layers.append(
                SetAttentionBlock(
                    d_model,
                    num_heads,
                    d_head,
                    d_ff,
                    dropouth=dropouth,
                    dropouta=dropouta,
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, graph, feat):
        """
        Compute the decoder part of Set Transformer.

        Parameters
        ----------
        graph : DGLGraph
            The input graph.
        feat : torch.Tensor
            The input feature with shape :math:`(N, D)`, where :math:`N` is the
            number of nodes in the graph, and :math:`D` means the size of features.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, D)`, where :math:`B` refers to
            the batch size.
        """
        len_pma = graph.batch_num_nodes()
        len_sab = [self.k] * graph.batch_size
        feat = self.pma(feat, len_pma)
        for layer in self.layers:
            feat = layer(feat, len_sab)
        return feat.view(graph.batch_size, self.k * self.d_model)


class WeightAndSum(nn.Module):
    """Compute importance weights for atoms and perform a weighted sum.

    Parameters
    ----------
    in_feats : int
        Input atom feature size

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import WeightAndSum
    >>>
    >>> g1 = dgl.rand_graph(3, 4)  # g1 is a random graph with 3 nodes and 4 edges
    >>> g1_node_feats = th.rand(3, 5)  # feature size is 5
    >>> g1_node_feats
    tensor([[0.8948, 0.0699, 0.9137, 0.7567, 0.3637],
            [0.8137, 0.8938, 0.8377, 0.4249, 0.6118],
            [0.5197, 0.9030, 0.6825, 0.5725, 0.4755]])
    >>>
    >>> g2 = dgl.rand_graph(4, 6)  # g2 is a random graph with 4 nodes and 6 edges
    >>> g2_node_feats = th.rand(4, 5)  # feature size is 5
    >>> g2_node_feats
    tensor([[0.2053, 0.2426, 0.4111, 0.9028, 0.5658],
            [0.5278, 0.6365, 0.9990, 0.2351, 0.8945],
            [0.3134, 0.0580, 0.4349, 0.7949, 0.3891],
            [0.0142, 0.2709, 0.3330, 0.8521, 0.6925]])
    >>>
    >>> weight_and_sum = WeightAndSum(5)  # create a weight and sum layer(in_feats=16)

    Case 1: Input a single graph

    >>> weight_and_sum(g1, g1_node_feats)
    tensor([[1.2194, 0.9490, 1.3235, 0.9609, 0.7710]],
           grad_fn=<SegmentReduceBackward>)

    Case 2: Input a batch of graphs

    Build a batch of DGL graphs and concatenate all graphs' node features into one tensor.

    >>> batch_g = dgl.batch([g1, g2])
    >>> batch_f = th.cat([g1_node_feats, g2_node_feats])
    >>>
    >>> weight_and_sum(batch_g, batch_f)
    tensor([[1.2194, 0.9490, 1.3235, 0.9609, 0.7710],
            [0.5322, 0.5840, 1.0729, 1.3665, 1.2360]],
           grad_fn=<SegmentReduceBackward>)

    Notes
    -----
    WeightAndSum module was commonly used in molecular property prediction networks,
    see the GCN predictor in `dgl-lifesci <https://github.com/awslabs/dgl-lifesci/blob/
    ae0491431804611ba466ff413f69d435789dbfd5/python/dgllife/model/model_zoo/
    gcn_predictor.py>`__
    to understand how to use WeightAndSum layer to get the graph readout output.
    """

    def __init__(self, in_feats):
        super(WeightAndSum, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1), nn.Sigmoid()
        )

    def forward(self, g, feats):
        """Compute molecule representations out of atom representations

        Parameters
        ----------
        g : DGLGraph
            DGLGraph with batch size B for processing multiple molecules in parallel
        feats : FloatTensor of shape (N, self.in_feats)
            Representations for all atoms in the molecules
            * N is the total number of atoms in all molecules

        Returns
        -------
        FloatTensor of shape (B, self.in_feats)
            Representations for B molecules
        """
        with g.local_scope():
            g.ndata["h"] = feats
            g.ndata["w"] = self.atom_weighting(g.ndata["h"])
            h_g_sum = sum_nodes(g, "h", "w")

        return h_g_sum
