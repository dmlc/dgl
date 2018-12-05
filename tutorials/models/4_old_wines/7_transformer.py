"""
.. _model-transformer:

Transformer DGL Tutorial
=========================

**Author**: aaa, bbb, ccc, ddd
"""

import torch as th
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from dgl.contrib.transformer.layers import * 
from dgl.contrib.transformer.functions import *
from dgl.contrib.transformer.embedding import *
from dgl.contrib.transformer.viz import att_animation, get_attention_map, draw_g
from dgl.contrib.transformer.optims import NoamOpt
from dgl.contrib.transformer.loss import LabelSmoothing, SimpleLossCompute
from dgl.contrib.transformer.dataset import get_dataset, GraphPool

"""
Start of model
"""
VIZ_IDX = 3

import dgl.function as fn
import torch.nn as nn
import torch.nn.init as INIT
from torch.nn import LayerNorm


class MultiHeadAttention(nn.Module):
    "Multi-Head Attention"
    def __init__(self, h, dim_model, dropout=0.1):
        '''
        h: number of heads
        dim_model: hidden dimension
        '''
        super(MultiHeadAttention, self).__init__()
        assert dim_model % h == 0
        self.d_k = dim_model // h
        self.h = h

        # W_q, W_k, W_v, W_o
        self.linears = clones(
            nn.Linear(dim_model, dim_model), 4
        )
        self.dropout = nn.Dropout(dropout)

    def get(self, x, fields='qkv'):
        "Return a dict of queries / keys / values."
        batch_size = x.shape[0]
        ret = {}
        if 'q' in fields:
            ret['q'] = self.linears[0](x).view(batch_size, self.h, self.d_k)
        if 'k' in fields:
            ret['k'] = self.linears[1](x).view(batch_size, self.h, self.d_k)
        if 'v' in fields:
            ret['v'] = self.linears[2](x).view(batch_size, self.h, self.d_k)
        return ret

    def get_o(self, x):
        "get output of the multi-head attention"
        batch_size = x.shape[0]
        return self.linears[3](x.view(batch_size, -1))


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def pre_func(self, i, fields='qkv'):
        layer = self.layers[i]
        def func(nodes):
            x = nodes.data['x']
            norm_x = layer.sublayer[0].norm(x)
            return layer.self_attn.get(norm_x, fields=fields)
        return func

    def post_func(self, i):
        layer = self.layers[i]
        def func(nodes):
            x, wv, z = nodes.data['x'], nodes.data['wv'], nodes.data['z']
            o = layer.self_attn.get_o(wv / z)
            x = x + layer.sublayer[0].dropout(o)
            x = layer.sublayer[1](x, layer.feed_forward)
            return {'x': x if i < self.N - 1 else self.norm(x)}
            # return {'x': x}
        return func


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def pre_func(self, i, fields='qkv', l=0):
        layer = self.layers[i]
        def func(nodes):
            x = nodes.data['x']
            if fields == 'kv':
                norm_x = x # In enc-dec attention, x has already been normalized.
            else:
                norm_x = layer.sublayer[l].norm(x)
            return layer.self_attn.get(norm_x, fields)
        return func

    def post_func(self, i, l=0):
        layer = self.layers[i]
        def func(nodes):
            x, wv, z = nodes.data['x'], nodes.data['wv'], nodes.data['z']
            o = layer.self_attn.get_o(wv / z)
            x = x + layer.sublayer[l].dropout(o)
            if l == 1:
                x = layer.sublayer[2](x, layer.feed_forward)
            return {'x': x if i < self.N - 1 else self.norm(x)}
        return func

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, pos_enc, generator, h, d_k):
        super(Transformer, self).__init__()
        self.encoder,  self.decoder = encoder, decoder
        self.src_embed, self.tgt_embed = src_embed, tgt_embed
        self.pos_enc = pos_enc
        self.generator = generator
        self.h, self.d_k = h, d_k
        self.att_weight_map = None

    def propagate_attention(self, g, eids):
        # Compute attention score
        g.apply_edges(src_dot_dst('k', 'q', 'score'), eids)
        g.apply_edges(scaled_exp('score', np.sqrt(self.d_k)), eids)
        # Send weighted values to target nodes
        g.send_and_recv(eids,
                        [fn.src_mul_edge('v', 'score', 'v'), fn.copy_edge('score', 'score')],
                        [fn.sum('v', 'wv'), fn.sum('score', 'z')])

    def update_graph(self, g, eids, pre_pairs, post_pairs):
        "Update the node states and edge states of the graph."

        # Pre-compute queries and key-value pairs.
        for pre_func, nids in pre_pairs:
            g.apply_nodes(pre_func, nids)
        self.propagate_attention(g, eids)
        # Further calculation after attention mechanism
        for post_func, nids in post_pairs:
            g.apply_nodes(post_func, nids)

    def forward(self, graph):
        g = graph.g
        nids, eids = graph.nids, graph.eids

        # embed
        src_embed, src_pos = self.src_embed(graph.src[0]), self.pos_enc(graph.src[1])
        tgt_embed, tgt_pos = self.tgt_embed(graph.tgt[0]), self.pos_enc(graph.tgt[1])
        g.nodes[nids['enc']].data['x'] = self.pos_enc.dropout(src_embed + src_pos)
        g.nodes[nids['dec']].data['x'] = self.pos_enc.dropout(tgt_embed + tgt_pos)

        for i in range(self.encoder.N):
            pre_func = self.encoder.pre_func(i, 'qkv')
            post_func = self.encoder.post_func(i)
            nodes, edges = nids['enc'], eids['ee']
            self.update_graph(g, edges, [(pre_func, nodes)], [(post_func, nodes)])

        for i in range(self.decoder.N):
            pre_func, post_func = self.decoder.pre_func(i, 'qkv'), self.decoder.post_func(i)
            nodes, edges = nids['dec'], eids['dd']
            self.update_graph(g, edges, [(pre_func, nodes)], [(post_func, nodes)])
            pre_q, pre_kv = self.decoder.pre_func(i, 'q', 1), self.decoder.pre_func(i, 'kv', 1)
            post_func = self.decoder.post_func(i, 1)
            nodes_e, nodes_d, edges = nids['enc'], nids['dec'], eids['ed']
            self.update_graph(g, edges, [(pre_q, nodes_d), (pre_kv, nodes_e)], [(post_func, nodes_d)])

        # visualize attention
        if self.att_weight_map is None:
            self._register_att_map(g, graph.nid_arr['enc'][VIZ_IDX], graph.nid_arr['dec'][VIZ_IDX])

        return self.generator(g.ndata['x'][nids['dec']])

    def _register_att_map(self, g, enc_ids, dec_ids):
        self.att_weight_map = [
            get_attention_map(g, enc_ids, enc_ids, self.h),
            get_attention_map(g, enc_ids, dec_ids, self.h),
            get_attention_map(g, dec_ids, dec_ids, self.h),
        ]


def make_model(src_vocab, tgt_vocab, N=6,
                   dim_model=512, dim_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h, dim_model, dropout)
    ff = PositionwiseFeedForward(dim_model, dim_ff, dropout)
    pos_enc = PositionalEncoding(dim_model, dropout)

    encoder = Encoder(EncoderLayer(dim_model, c(attn), c(ff), dropout), N)
    decoder = Decoder(DecoderLayer(dim_model, c(attn), c(attn), c(ff), dropout), N)
    src_embed = Embeddings(src_vocab, dim_model)
    tgt_embed = Embeddings(tgt_vocab, dim_model)
    generator = Generator(dim_model, tgt_vocab)
    model = Transformer(
        encoder, decoder, src_embed, tgt_embed, pos_enc, generator, h, dim_model // h)
    # xavier init
    for p in model.parameters():
        if p.dim() > 1:
            INIT.xavier_uniform_(p)
    return model
"""
End of model
"""


def run_epoch(data_iter, model, loss_compute, is_train=True):
    for i, g in enumerate(data_iter):
        with th.set_grad_enabled(is_train):
            output = model(g)
            loss = loss_compute(output, g.tgt_y, g.n_tokens)
    print('average loss: {}'.format(loss_compute.avg_loss))
    print('accuracy: {}'.format(loss_compute.accuracy))


np.random.seed(1111)
N = 1
batch_size = 128 
devices = ['cuda' if th.cuda.is_available() else 'cpu']

dataset = get_dataset("copy")
V = dataset.vocab_size
criterion = LabelSmoothing(V, padding_idx=dataset.pad_id, smoothing=0.1)
dim_model = 128

graph_pool = GraphPool()

"""
Display dataset
"""
data_iter = dataset(graph_pool, mode='train', batch_size=1, devices=devices)
for graph in data_iter:
    print(graph.nids['enc'])
    print(graph.nids['dec'])
    draw_g(graph)
    break

# Create model
model = make_model(V, V, N=N, dim_model=128, dim_ff=128, h=1)

# Sharing weights between Encoder & Decoder
model.src_embed.lut.weight = model.tgt_embed.lut.weight
model.generator.proj.weight = model.tgt_embed.lut.weight

model, criterion = model.to(devices[0]), criterion.to(devices[0])
model_opt = NoamOpt(dim_model, 1, 400,
                    th.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9))
loss_compute = SimpleLossCompute

att_maps = []
for epoch in range(4):
    train_iter = dataset(graph_pool, mode='train', batch_size=batch_size, devices=devices)
    valid_iter = dataset(graph_pool, mode='valid', batch_size=batch_size, devices=devices)
    print('Epoch: {} Training...'.format(epoch))
    model.train(True)
    run_epoch(train_iter, model,
              loss_compute(criterion, model_opt), is_train=True)
    print('Epoch: {} Evaluating...'.format(epoch))
    model.att_weight_map = None
    model.eval()
    run_epoch(valid_iter, model,
              loss_compute(criterion, None), is_train=False)
    att_maps.append(model.att_weight_map)

src_seq = dataset.get_seq_by_id(VIZ_IDX, mode='valid', field='src')
tgt_seq = dataset.get_seq_by_id(VIZ_IDX, mode='valid', field='tgt')[:-1]
print(src_seq, '\n', tgt_seq)
# visualize head 0 of encoder-decoder attention
anim = att_animation(att_maps, 'e2d', src_seq, tgt_seq, 0)
#anim.save('att.gif', dpi=150, writer='imagemagick')
