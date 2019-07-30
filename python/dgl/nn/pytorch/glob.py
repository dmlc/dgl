"""Torch modules for graph global pooling."""
# pylint: disable= no-member, arguments-differ, C0103, W0235
import torch as th
import torch.nn as nn
import numpy as np

from ..utils import _create_fully_connected_graph, _create_bipartite_graph, \
    _create_graph_from_num_nodes
from .softmax import edge_softmax
from ... import function as fn, BatchedDGLGraph
from ...utils import get_ndata_name
from ...batched_graph import sum_nodes, mean_nodes, max_nodes, broadcast_nodes,\
    softmax_nodes, topk_nodes


__all__ = ['SumPooling', 'AvgPooling', 'MaxPooling', 'SortPooling',
           'GlobalAttentionPooling', 'Set2Set',
           'SetTransformerEncoder', 'SetTransformerDecoder']

class SumPooling(nn.Module):
    r"""Apply sum pooling over the graph.
    """
    _feat_name = '_gpool_feat'
    def __init__(self):
        super(SumPooling, self).__init__()

    def forward(self, feat, graph):
        r"""Compute sum pooling.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature
        graph : DGLGraph
            The graph.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        _feat_name = get_ndata_name(graph, self._feat_name)
        graph.ndata[_feat_name] = feat
        readout = sum_nodes(graph, _feat_name)
        graph.ndata.pop(_feat_name)
        return readout


class AvgPooling(nn.Module):
    r"""Apply average pooling over the graph.
    """
    _feat_name = '_gpool_avg'
    def __init__(self):
        super(AvgPooling, self).__init__()

    def forward(self, feat, graph):
        r"""Compute average pooling.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature
        graph : DGLGraph
            The graph.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        _feat_name = get_ndata_name(graph, self._feat_name)
        graph.ndata[_feat_name] = feat
        readout = mean_nodes(graph, _feat_name)
        graph.ndata.pop(_feat_name)
        return readout


class MaxPooling(nn.Module):
    r"""Apply max pooling over the graph.
    """
    _feat_name = '_gpool_max'
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, feat, graph):
        r"""Compute max pooling.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature
        graph : DGLGraph
            The graph.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        _feat_name = get_ndata_name(graph, self._feat_name)
        graph.ndata[_feat_name] = feat
        readout = max_nodes(graph, _feat_name)
        graph.ndata.pop(_feat_name)
        return readout


class SortPooling(nn.Module):
    r"""Apply sort pooling (f"An End-to-End Deep Learning Architecture
    for Graph Classification") over the graph.

    Parameters
    ----------
    k : int
        The number of nodes to hold for each graph.
    """
    _feat_name = '_gpool_sort'
    def __init__(self, k):
        super(SortPooling, self).__init__()
        self.k = k

    def forward(self, feat, graph):
        r"""Compute sort pooling.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature
        graph : DGLGraph
            The graph.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        # Sort the feature of each node in ascending order.
        feat, _ = feat.sort(dim=-1)
        graph.ndata[self._feat_name] = feat
        # Sort nodes according to their last features.
        ret = topk_nodes(graph, self._feat_name, self.k)[0].view(
            -1, self.k * feat.shape[-1])
        graph.ndata.pop(self._feat_name)
        if isinstance(graph, BatchedDGLGraph):
            return ret
        else:
            return ret.squeeze(0)


class GlobalAttentionPooling(nn.Module):
    r"""Apply global attention pooling over the graph.

    Parameters
    ----------
    gate_nn : torch.nn.Module
        A neural network that computes attention scores for each feature.
    feat_nn : torch.nn.Module, optional
        A neural network applied to each feature before combining them
        with attention scores.
    """
    _gate_name = '_gpool_attn_gate'
    _readout_name = '_gpool_attn_readout'
    def __init__(self, gate_nn, feat_nn=None):
        super(GlobalAttentionPooling, self).__init__()
        self.gate_nn = gate_nn
        self.feat_nn = feat_nn
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.gate_nn.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if self.feat_nn:
            for p in self.feat_nn.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, feat, graph):
        r"""Compute global attention pooling.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature
        graph : DGLGraph
            The graph.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        gate = self.gate_nn(feat)
        assert gate.shape[-1] == 1, "The output of gate_nn should have size 1 at the last axis."
        feat = self.feat_nn(feat) if self.feat_nn else feat

        feat_name = get_ndata_name(graph, self._gate_name)
        graph.ndata[feat_name] = gate
        gate = softmax_nodes(graph, feat_name)
        graph.ndata.pop(feat_name)

        feat_name = get_ndata_name(graph, self._readout_name)
        graph.ndata[feat_name] = feat * gate
        readout = sum_nodes(graph, feat_name)
        graph.ndata.pop(feat_name)

        return readout


class Set2Set(nn.Module):
    r"""Apply Set2Set (f"Order Matters: Sequence to sequence for sets") over the graph.

    Parameters
    ----------
    input_dim : int
        Size of each input sample
    n_iters : int
        Number of iterations.
    n_layers : int
        Number of recurrent layers.
    """
    _score_name = '_gpool_s2s_score'
    _readout_name = '_gpool_s2s_readout'
    def __init__(self, input_dim, n_iters, n_layers):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = th.nn.LSTM(self.output_dim, self.input_dim, n_layers)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.lstm.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat, graph):
        r"""Compute set2set pooling.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature
        graph : DGLGraph
            The graph.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        batch_size = 1
        if isinstance(graph, BatchedDGLGraph):
            batch_size = graph.batch_size

        h = (feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
             feat.new_zeros((self.n_layers, batch_size, self.input_dim)))
        q_star = feat.new_zeros(batch_size, self.output_dim)

        for _ in range(self.n_iters):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.input_dim)

            score = (feat * broadcast_nodes(graph, q)).sum(dim=-1, keepdim=True)
            feat_name = get_ndata_name(graph, self._score_name)
            graph.ndata[feat_name] = score
            score = softmax_nodes(graph, feat_name)
            graph.ndata.pop(feat_name)

            feat_name = get_ndata_name(graph, self._readout_name)
            graph.ndata[feat_name] = feat * score
            readout = sum_nodes(graph, feat_name)
            graph.ndata.pop(feat_name)

            if readout.dim() == 1: # graph is not a BatchedDGLGraph
                readout = readout.unsqueeze(0)

            q_star = th.cat([q, readout], dim=-1)

        if isinstance(graph, BatchedDGLGraph):
            return q_star
        else:
            return q_star.squeeze(0)

    def extra_repr(self):
        """Set the extra representation of the module.
        which will come into effect when printing the model.
        """
        summary = 'n_iters={n_iters}'
        return summary.format(**self.__dict__)


class MultiHeadAttention(nn.Module):
    r""" Multi-Head Attention block, used in Transformer, Set Transformer and so on."""
    _query_name = '_gpool_mha_query'
    _key_name = '_gpool_mha_key'
    _value_name = '_gpool_mha_value'
    _score_name = '_gpool_mha_score'
    _att_name = '_gpool_mha_att'
    _out_name = '_gpool_mha_out'
    _feat_name = '_gpool_mha_feat'
    def __init__(self, d_model, num_heads, d_head, d_ff, dropouth=0., dropouta=0.):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.W_q = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.W_o = nn.Linear(num_heads * d_head, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropouth),
            nn.Linear(d_ff, d_model)
        )
        self.droph = nn.Dropout(dropouth)
        self.dropa = nn.Dropout(dropouta)
        self.norm_in = nn.LayerNorm(d_model)
        self.norm_inter = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, q_feat, kv_feat, q_nids, kv_nids):
        """
        Compute multi-head self-attention.
        """
        feat_name = get_ndata_name(graph, self._feat_name)
        query_name = get_ndata_name(graph, self._query_name)
        key_name = get_ndata_name(graph, self._key_name)
        value_name = get_ndata_name(graph, self._value_name)
        score_name = get_ndata_name(graph, self._score_name)
        att_name = get_ndata_name(graph, self._att_name)
        out_name = get_ndata_name(graph, self._out_name)

        # Copy q_feat and kv_feat to graph data frame
        graph.nodes[q_nids].data[feat_name] = q_feat
        graph.nodes[kv_nids].data[feat_name] = kv_feat

        # Compute queries, keys and values.
        graph.nodes[q_nids].data[query_name] =\
            self.W_q(graph.nodes[q_nids].data[feat_name]).view(-1, self.num_heads, self.d_head)
        graph.nodes[kv_nids].data[key_name] =\
            self.W_k(graph.nodes[kv_nids].data[feat_name]).view(-1, self.num_heads, self.d_head)
        graph.nodes[kv_nids].data[value_name] =\
            self.W_v(graph.nodes[kv_nids].data[feat_name]).view(-1, self.num_heads, self.d_head)

        # Free node features.
        graph.ndata.pop(feat_name)

        # Compute attention score.
        graph.apply_edges(fn.u_mul_v(key_name, query_name, score_name))
        e = graph.edata.pop(score_name).sum(dim=-1, keepdim=True) / np.sqrt(self.d_head)
        graph.edata[att_name] = self.dropa(edge_softmax(graph, e))
        graph.pull(q_nids,
                   fn.u_mul_e(value_name, att_name, 'm'),
                   fn.sum('m', out_name))
        sa = self.W_o(graph.nodes[q_nids].data[out_name].view(-1, self.num_heads * self.d_head))
        feat = self.norm_in(q_feat + sa)

        # Free queries, keys, values, outputs and attention weights.
        graph.ndata.pop(query_name)
        graph.ndata.pop(key_name)
        graph.ndata.pop(value_name)
        graph.ndata.pop(out_name)
        graph.edata.pop(att_name)

        # Position-wise Feed Forward Network
        feat = self.norm_inter(feat + self.ffn(feat))

        return feat


class SetAttentionBlock(nn.Module):
    r""" SAB block mentioned in Set-Transformer paper."""
    def __init__(self, d_model, num_heads, d_head, d_ff, dropouth=0., dropouta=0.):
        super(SetAttentionBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, d_head, d_ff,
                                      dropouth=dropouth, dropouta=dropouta)

    def forward(self, feat, graph, sab_graph=None):
        """
        Compute a Set Attention Block.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature
        graph : DGLGraph
            The graph.
        sab_graph: DGLGraph or None
            The graph to apply Message Passing on, set to None if not
            specified.
        """
        if sab_graph is None:
            sab_graph = _create_fully_connected_graph(graph)

        q_nids = th.arange(sab_graph.number_of_nodes())
        kv_nids = q_nids

        return self.mha(sab_graph, feat, feat, q_nids, kv_nids)

class InducedSetAttentionBlock(nn.Module):
    r""" ISAB block mentioned in Set-Transformer paper."""
    def __init__(self, m, d_model, num_heads, d_head, d_ff, dropouth=0., dropouta=0.):
        super(InducedSetAttentionBlock, self).__init__()
        self.m = m
        self.I = nn.Parameter(
            th.FloatTensor(m, d_model)
        )
        self.mha = nn.ModuleList([
            MultiHeadAttention(d_model, num_heads, d_head, d_ff,
                               dropouth=dropouth, dropouta=dropouta) for _ in range(2)])
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.I)

    def forward(self, feat, graph, isab_graph=None):
        """
        Compute an Induced Set Attention Block.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature
        graph : DGLGraph
            The graph.
        isab_graph: DGLGraph or None
            The graph to apply Message Passing on, set to None if not
            specified.
        """
        query = self.I
        batch_size = 1
        if isinstance(graph, BatchedDGLGraph):
            batch_size = graph.batch_size

        if isab_graph is None:
            induced_graph = _create_graph_from_num_nodes([self.m] * batch_size)
            isab_graph = [
                _create_bipartite_graph(graph, induced_graph),
                _create_bipartite_graph(induced_graph, graph)
            ]

        query = query.repeat(batch_size, 1)
        for mha, (g, kv_nids, q_nids) in zip(self.mha, isab_graph):
            rst = mha(g, query, feat, q_nids, kv_nids)
            query, feat = feat, rst

        return rst

    def extra_repr(self):
        """Set the extra representation of the module.
        which will come into effect when printing the model.
        """
        shape_str = '({}, {})'.format(self.I.shape[0], self.I.shape[1])
        return 'InducedVector: ' + shape_str


class PMALayer(nn.Module):
    r"""Pooling by Multihead Attention, used in the Decoder Module of Set Transformer."""
    def __init__(self, k, d_model, num_heads, d_head, d_ff, dropouth=0., dropouta=0.):
        self.k = k
        super(PMALayer, self).__init__()
        self.S = nn.Parameter(
            th.FloatTensor(k, d_model)
        )
        self.mha = MultiHeadAttention(d_model, num_heads, d_head, d_ff,
                                      dropouth=dropouth, dropouta=dropouta)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.S)

    def forward(self, feat, graph, pma_graph=None):
        """
        Compute Pooling by Multihead Attention.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature
        graph : DGLGraph
            The graph.
        pma_graph: DGLGraph or None
            The graph to apply Message Passing on, set to None if not
            specified.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        query = self.S
        batch_size = 1
        if isinstance(graph, BatchedDGLGraph):
            batch_size = graph.batch_size

        if pma_graph is None:
            induced_graph = _create_graph_from_num_nodes([self.k] * batch_size)
            pma_graph = _create_bipartite_graph(graph, induced_graph)

        g, kv_nids, q_nids = pma_graph
        query = query.repeat(batch_size, 1)
        return self.mha(g, query, feat, q_nids, kv_nids)

    def extra_repr(self):
        """Set the extra representation of the module.
        which will come into effect when printing the model.
        """
        shape_str = '({}, {})'.format(self.S.shape[0], self.S.shape[1])
        return 'SeedVector: ' + shape_str


class SetTransformerEncoder(nn.Module):
    r"""(experimental) The Encoder module in Set Transformer paper.

    Parameters
    ----------
    d_model : int
        Hidden size of the model.
    n_heads : int
        Number of heads.
    d_head : int
        Hidden size of each head.
    d_ff : int
        Kernel size in FFN (Positionwise Feed-Forward Network) layer.
    n_layers : int
        Number of layers.
    block_type : str
        Building block type: 'sab' (Set Attention Block) or 'isab' (Induced
        Set Attention Block).
    m : int or None
        Number of induced vectors in ISAB Block, set to None if block type
        is 'sab'.
    dropouth : float
        Dropout rate of each sublayer.
    dropouta : float
        Dropout rate of attention heads.
    """
    def __init__(self, d_model, n_heads, d_head, d_ff,
                 n_layers=1, block_type='sab', m=None, dropouth=0., dropouta=0.):
        super(SetTransformerEncoder, self).__init__()
        self.n_layers = n_layers
        self.block_type = block_type
        self.m = m
        layers = []
        if block_type == 'isab' and m is None:
            raise KeyError('The number of inducing points is not specified in ISAB block.')

        for _ in range(n_layers):
            if block_type == 'sab':
                layers.append(
                    SetAttentionBlock(d_model, n_heads, d_head, d_ff,
                                      dropouth=dropouth, dropouta=dropouta))
            elif block_type == 'isab':
                layers.append(
                    InducedSetAttentionBlock(m, d_model, n_heads, d_head, d_ff,
                                             dropouth=dropouth, dropouta=dropouta))
            else:
                raise KeyError("Unrecognized block type {}: we only support sab/isab")

        self.layers = nn.ModuleList(layers)

    def forward(self, feat, graph):
        """
        Compute the Encoder part of Set Transformer.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature
        graph : DGLGraph
            The graph.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        batch_size = 1
        if isinstance(graph, BatchedDGLGraph):
            batch_size = graph.batch_size

        if self.block_type == 'sab':
            att_graph = _create_fully_connected_graph(graph)
        else:
            induced_graph = _create_graph_from_num_nodes([self.m] * batch_size)
            att_graph = [
                _create_bipartite_graph(graph, induced_graph),
                _create_bipartite_graph(induced_graph, graph)
            ]

        for layer in self.layers:
            feat = layer(feat, graph, att_graph)

        return feat


class SetTransformerDecoder(nn.Module):
    r"""(experimental) The Decoder module in Set Transformer paper.

    Parameters
    ----------
    d_model : int
        Hidden size of the model.
    num_heads : int
        Number of heads.
    d_head : int
        Hidden size of each head.
    d_ff : int
        Kernel size in FFN (Positionwise Feed-Forward Network) layer.
    n_layers : int
        Number of layers.
    k : int
        Number of seed vectors in PMA (Pooling by Multihead Attention) layer.
    dropouth : float
        Dropout rate of each sublayer.
    dropouta : float
        Dropout rate of attention heads.
    """
    def __init__(self, d_model, num_heads, d_head, d_ff, n_layers, k, dropouth=0., dropouta=0.):
        super(SetTransformerDecoder, self).__init__()
        self.n_layers = n_layers
        self.k = k
        self.d_model = d_model
        self.pma = PMALayer(k, d_model, num_heads, d_head, d_ff,
                            dropouth=dropouth, dropouta=dropouta)
        layers = []
        for _ in range(n_layers):
            layers.append(
                SetAttentionBlock(d_model, num_heads, d_head, d_ff,
                                  dropouth=dropouth, dropouta=dropouta))

        self.layers = nn.ModuleList(layers)

    def forward(self, feat, graph):
        """
        Compute the decoder part of Set Transformer.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature
        graph : DGLGraph
            The graph.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        if isinstance(graph, BatchedDGLGraph):
            induced_graph = _create_graph_from_num_nodes([self.k] * graph.batch_size)
        else:
            induced_graph = _create_graph_from_num_nodes([self.k])

        pma_graph = _create_bipartite_graph(graph, induced_graph)
        feat = self.pma(feat, graph, pma_graph=pma_graph)

        sab_graph = _create_fully_connected_graph(induced_graph)
        for layer in self.layers:
            feat = layer(feat, graph, sab_graph=sab_graph)

        if isinstance(graph, BatchedDGLGraph):
            return feat.view(graph.batch_size, self.k * self.d_model)
        else:
            return feat.view(self.k * self.d_model)
