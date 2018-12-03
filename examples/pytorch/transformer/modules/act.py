from .layers import *
from .functions import *
from .embedding import *
import torch as th

class UEncoder(nn.Module):
    def __init__(self, layer):
        super(UEncoder, self).__init__()
        self.N = N
        self.layer = layer
        self.norm = LayerNorm(layer.size)

    def pre_func(self, fields='qkv'):
        layer = self.layer
        def func(nodes):
            x = nodes.data['x']
            norm_x = layer.sublayer[0].norm(x)
            return layer.self_attn.get(norm_x, fields=fields)
        return func

    def post_func(self):
        layer = self.layer
        def func(nodes):
            pass
        return func


class UDecoder(nn.Module):
    def __init__(self, layer):
        super(UDecoder, self).__init__()
        self.N = N
        self.layer = layer
        self.norm = LayerNorm(layer.size)

    def pre_func(self, fields='qkv', l=0):
        layer = self.layer
        def func(nodes):
            x = nodes.data['x']
            if fields == 'kv':
                norm_x = x
            else:
                norm_x = layer.sublayer[l].norm(x)
            return layer.self_attn.get(norm_x, fields)
        return func

    def post_func(self, l=0):
        layer = self.layer
        def func(nodes):
            pass
        return func


class HaltingUnit(nn.Module):
    halting_bias_init = 1.0
    def __init___(self, dim_model):
        self.linear = nn.Linear(dim_model, 1)
        INIT.constant_(self.linear, self.halting_bias_init)

    def forward(x):
        return th.sigmoid(self.linear(x))

class UTransformer(nn.Module):
    '''
    The Basic Universal Transformer(https://arxiv.org/pdf/1807.03819.pdf) with ACT based on remainder-distribution ACT(https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/universal_transformer_util.py).
    '''
    MAX_DEPTH = 8
    thres = 0.99
    act_loss_weight = 0.01
    def __init__(self, encoder, decoder, src_embed, tgt_embed, pos_enc, time_enc, generator, h, d_k):
        # TODO
        self.time_enc = time_enc
        self.pos_enc = pos_enc
        self.halt_enc = HaltingUnit(h * d_k)
        self.halt_dec = HaltingUnit(h * d_k)

    def add_time_pos_enc(self, nodes):
        x = nodes['x']
        step = nodes['step']
        pos = nodes['pos']
        return {'x_p': x + self.pos_enc(pos) + self.time_enc(step)}

    def update_graph(self, g, pre_pairs, post_pair):
        "Update the node status and edge status of the graph."

        # Pre-compute queries and key-value pairs and add time encoding.
        for pre_func, nodes in pre_pairs:
            g.apply_nodes(pre_func, nodes)

        post_func, edges = post_pair[0]
        # Compute attention score
        g.apply_edges(src_dot_dst('k', 'q', 'score'), edges)
        g.apply_edges(scaled_exp('score', np.sqrt(self.d_k)))

        # Update node state
        g.send_and_recv(edges,
                        [fn.src_mul_edge('v', 'score', 'v'), fn.copy_edge('score', 'score') ],
                        [fn.sum('v', 'wv'), fn.sum('score', 'z')],
                        post_func)

        # Compute halting prob
        # TODO: some other issues

    def forward(self, graph):
        g = graph.g
        N = graph.n_nodes
        E = graph.n_edges
        h = self.h
        d_k = self.d_k
        nids = graph.nids
        eids = graph.eids

        # embed
        src_embed, src_pos = self.src_embed[0](graph.src[0]), self.src_embed[1](graph.src[1])
        tgt_embed, tgt_pos = self.tgt_embed[0](graph.tgt[0]), self.tgt_embed[1](graph.tgt[1])
        g.nodes[nids['enc']].data['x'] = self.src_embed[1].dropout(src_embed + src_pos)
        g.nodes[nids['dec']].data['x'] = self.tgt_embed[1].dropout(tgt_embed + tgt_pos)

        # init ponder value
        device = next(self.parameters()).device
        g.ndata['ponder'] = th.zeros(N, dtype=th.float, device=device)
        g.ndata['step'] = th.zeros(N, dtype=th.long, device=device)

        for i in range(self.MAX_DEPTH):
            pre_func, post_func = self.encoder.pre_func('qkv'), self.encoder.post_func()
            nodes, edges = nids['enc'], eids['ee']
            self.update_graph(g, [(pre_func, nodes)], (post_func, edges))

        for i in range(self.MAX_DEPTH):
            pre_func, post_func = self.decoder.pre_func('qkv'), self.decoder.post_func()
            nodes, edges = nids['dec'], eids['dd']
            self.update_graph(g, [(pre_func, nodes)], (post_func, edges))
            pre_q, pre_kv, post_func = self.decoder.pre_func('q', 1), self.decoder.pre_func('kv', 1)
            nodes_e, nodes_d, edges = nids['enc'], nids['dec'], eids['ed']
            self.update_graph(g, [(pre_q, nodes_d), (pre_kv, nodes_e)], (post_func, edges))

        return self.generator(g.ndata['x'][nids['dec']])


def make_universal_model(src_vocab, tgt_vocab, dim_model=512, dim_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h, dim_model, dropout)
    ff = PositionwiseFeedForward(dim_model, dim_ff, dropout)
    pos_enc = PositionalEncoding(dim_model, dropout)
    encoder = UEncoder(EncoderLayer(dim_model), c(attn), c(ff), dropout)
    decoder = UDecoder(DecoderLayer(dim_model), c(attn), c(attn), c(ff), dropout)
    src_embed = nn.ModuleList([Embeddings(src_vocab, dim_model), c(pos_enc)])
    tgt_embed = nn.ModuleList([Embeddings(tgt_vocab, dim_model), c(pos_enc)])
    generator = Generator(dim_model, tgt_vocab)
    model = UTransformer(
        encoder, decoder, src_embed, tgt_embed, generator, h, dim_model // h)
    # xavier init
    for p in model.parameters():
        if p.dim() > 1:
            INIT.xavier_uniform_(p)
    return model
