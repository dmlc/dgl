"""Torch modules for graph global pooling."""
# pylint: disable= no-member, arguments-differ, C0103
import torch as th
import torch.nn as nn
import numpy as np

from ..utils import _create_fully_connected_graph, _create_bipartite_graph, \
    _create_batched_graph_from_num_nodes
from torch.nn import init
from .softmax import edge_softmax
from ... import function as fn, BatchedDGLGraph
from ...utils import get_ndata_name
from ...batched_graph import sum_nodes, mean_nodes, max_nodes, broadcast_nodes,\
    softmax_nodes, topk_nodes


__all__ = ['SumPooling', 'AvgPooling', 'MaxPooling', 'SortPooling',
           'GlobAttnPooling', 'Set2Set', 'MultiHeadAttention',
           'SetAttnBlock', 'InducedSetAttnBlock', 'SetTransEncoder', 'SetTransDecoder']

class SumPooling(nn.Module):
    r"""Apply sum pooling over the graph.
    """
    _feat_name = '_gpool_feat'
    def __init__(self):
        super(SumPooling, self).__init__()

    def forward(self, feat, graph):
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
        _feat_name = get_ndata_name(graph, self._feat_name)
        graph.ndata[_feat_name] = feat
        readout = max_nodes(graph, _feat_name)
        graph.ndata.pop(_feat_name)
        return readout


class SortPooling(nn.Module):
    r"""Apply sort pooling (f"An End-to-End Deep Learning Architecture
    for Graph Classification") over the graph.
    """
    _feat_name = '_gpool_sort'
    def __init__(self, k):
        super(SortPooling, self).__init__()
        self.k = k

    def forward(self, feat, graph):
        # Sort the feature of each node in ascending order.
        feat, _ = feat.sort(dim=-1)
        graph.ndata[self._feat_name] = feat
        # Sort nodes according to their last features.
        ret = topk_nodes(graph, self._feat_name, self.k).view(-1, self.k * feat.shape[-1])
        graph.ndata.pop(self._feat_name)
        return ret


class GlobAttnPooling(nn.Module):
    r"""Apply global attention pooling over the graph.
    """
    _gate_name = '_gpool_attn_gate'
    _readout_name = '_gpool_attn_readout'
    def __init__(self, gate_nn, feat_nn=None):
        super(GlobAttnPooling, self).__init__()
        self.gate_nn = gate_nn
        self.feat_nn = feat_nn
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.gate_nn.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)
        if self.feat_nn:
            for p in self.feat_nn.parameters():
                if p.dim() > 1:
                    init.xavier_uniform_(p)

    def forward(self, feat, graph):
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
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.lstm.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)

    def forward(self, feat, graph):
        batch_size = 1
        if isinstance(graph, BatchedDGLGraph):
            batch_size = graph.batch_size

        h = (feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
             feat.new_zeros((self.n_layers, batch_size, self.input_dim)))
        q_star = feat.new_zeros(batch_size, self.output_dim)

        for i in range(self.n_iters):
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

            q_star = th.cat([q, readout], dim=-1)

        return q_star

    def extra_repr(self):
        """Set the extra representation of the module.
        which will come into effect when printing the model.
        """
        summary = 'input_dim={input_dim}, out_dim={out_dim}' +\
            'n_iters={n_iters}, n_layers={n_layers}'
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
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)

    def forward(self, graph, q_feat, kv_feat, q_nids, kv_nids):
        feat_name, query_name, key_name, value_name, score_name, att_name, out_name = map()
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


class SetAttnBlock(nn.Module):
    r""" SAB block mentioned in Set-Transformer paper."""
    def __init__(self, d_model, num_heads, d_head, d_ff, dropouth=0., dropouta=0.):
        super(SetAttnBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, d_head, d_ff,
                                      dropouth=dropouth, dropouta=dropouta)

    def forward(self, graph, feat, sab_graph=None):
        if sab_graph is None:
            sab_graph = _create_fully_connected_graph(graph)

        q_nids = th.arange(sab_graph.number_of_nodes())
        kv_nids = q_nids

        return self.mha(sab_graph, feat, feat, q_nids, kv_nids)

class InducedSetAttnBlock(nn.Module):
    r""" ISAB block mentioned in Set-Transformer paper."""
    def __init__(self, m, d_model, num_heads, d_head, d_ff, dropouth=0., dropouta=0.):
        super(InducedSetAttnBlock, self).__init__()
        self.m = m
        self.I = nn.Parameter(
            th.FloatTensor(m, d_model)
        )
        self.mha = nn.ModuleList([
            MultiHeadAttention(d_model, num_heads, d_head, d_ff,
                               dropouth=dropouth, dropouta=dropouta) for _ in range(2)])
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.I)

    def forward(self, graph, feat, isab_graph=None):
        query = self.I
        if isab_graph is None:
            if isinstance(graph, BatchedDGLGraph):
                induced_graph = _create_batched_graph_from_num_nodes([self.m] * graph.batch_size)
            else:
                induced_graph = _create_batched_graph_from_num_nodes([self.m])

            isab_graph = [
                _create_bipartite_graph(graph, induced_graph),
                _create_bipartite_graph(induced_graph, graph)
            ]

            query = query.repeat(graph.batch_size, 1)

        for mha, (g, kv_nids, q_nids) in zip(self.mha, isab_graph):
            rst = mha(g, query, feat, q_nids, kv_nids)
            query, feat = feat, rst

        return rst


class _PMALayer(nn.Module):
    r"""Pooling by Multihead Attention, used in the Decoder Module of Set Transformer."""
    def __init__(self, k, d_model, num_heads, d_head, d_ff, dropouth=0., dropouta=0.):
        self.k = k
        super(_PMALayer, self).__init__()
        self.S = nn.Parameter(
            nn.FloatTensor(k, d_model)
        )
        self.mha = MultiHeadAttention(d_model, num_heads, d_head, d_ff,
                                      dropouth=dropouth, dropouta=dropouta)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.S)

    def forward(self, graph, feat, pma_graph=None):
        query = self.S
        if pma_graph is None:
            if isinstance(graph, BatchedDGLGraph):
                induced_graph = _create_batched_graph_from_num_nodes([self.k] * graph.batch_size)
            else:
                induced_graph = _create_batched_graph_from_num_nodes([self.k])
            pma_graph = _create_bipartite_graph(graph, induced_graph)

        g, kv_nids, q_nids = pma_graph

        return self.mha(g, query, feat, q_nids, kv_nids)


class SetTransEncoder(nn.Module):
    r""" The Encoder module in Set Transformer paper. """
    def __init__(self, d_model, num_heads, d_head, d_ff,
                 n_layers=1, block_type='sab', m=None, dropouth=0., dropouta=0.):
        super(SetTransEncoder, self).__init__()
        self.n_layers = n_layers
        self.block_type = block_type
        self.m = m
        layers = []
        if block_type == 'isab' and m is None:
            raise KeyError('The number of inducing points is not specified in ISAB block.')

        for _ in range(n_layers):
            if block_type == 'sab':
                layers.append(
                    SetAttnBlock(d_model, num_heads, d_head, d_ff,
                                 dropouth=dropouth, dropouta=dropouta))
            elif block_type == 'isab':
                layers.append(
                    InducedSetAttnBlock(m, d_model, num_heads, d_head, d_ff,
                                        dropouth=dropouth, dropouta=dropouta))
            else:
                raise KeyError("Unrecognized block type {}: we only support sab/isab")

        self.layers = nn.ModuleList(layers)

    def forward(self, graph, feat):
        if self.block_type == 'sab':
            att_graph = _create_fully_connected_graph(graph)
        else:
            if isinstance(graph, BatchedDGLGraph):
                induced_graph = _create_batched_graph_from_num_nodes([self.m] * graph.batch_size)
            else:
                induced_graph = _create_batched_graph_from_num_nodes([self.m])

            att_graph = [
                _create_bipartite_graph(graph, induced_graph),
                _create_bipartite_graph(induced_graph, graph)
            ]

        for layer in self.layers:
            feat = layer(graph, feat, att_graph)

        return feat


class SetTransDecoder(nn.Module):
    r""" The Decoder module in Set Transformer paper. """
    def __init__(self, k, d_model, num_heads, d_head, d_ff, n_layers, dropouth=0., dropouta=0.):
        super(SetTransDecoder, self).__init__()
        self.n_layers = n_layers
        self.k = k
        self.d_model = d_model
        self.pma = _PMALayer(k, d_model, num_heads, d_head, d_ff,
                             dropouth=dropouth, dropouta=dropouta)
        layers = []
        for _ in range(n_layers):
            layers.append(
                SetAttnBlock(d_model, num_heads, d_head, d_ff,
                             dropouth=dropouth, dropouta=dropouta))

        self.layers = nn.ModuleList(layers)

    def forward(self, graph, feat):
        if isinstance(graph, BatchedDGLGraph):
            induced_graph = _create_batched_graph_from_num_nodes([self.k] * graph.batch_size)
        else:
            induced_graph = _create_batched_graph_from_num_nodes([self.k])

        pma_graph = _create_bipartite_graph(graph, induced_graph)
        feat = self.pma(graph, feat, pma_graph=pma_graph)

        sab_graph = _create_fully_connected_graph(induced_graph)
        for layer in self.layers:
            feat = layer(graph, feat, sab_graph=sab_graph)

        if isinstance(graph, BatchedDGLGraph):
            return feat.view(graph.batch_size, self.k, self.d_model)
        else:
            return feat.view(self.k, self.d_model)
