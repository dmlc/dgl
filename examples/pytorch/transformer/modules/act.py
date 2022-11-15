from .attention import *
from .layers import *
from .functions import *
from .embedding import *
import torch as th
import dgl.function as fn
import torch.nn.init as INIT

class UEncoder(nn.Module):
    def __init__(self, layer):
        super(UEncoder, self).__init__()
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
            x, wv, z = nodes.data['x'], nodes.data['wv'], nodes.data['z']
            o = layer.self_attn.get_o(wv / z)
            x = x + layer.sublayer[0].dropout(o)
            x = layer.sublayer[1](x, layer.feed_forward)
            return {'x': x}
        return func


class UDecoder(nn.Module):
    def __init__(self, layer):
        super(UDecoder, self).__init__()
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
            x, wv, z = nodes.data['x'], nodes.data['wv'], nodes.data['z']
            o = layer.self_attn.get_o(wv / z)
            x = x + layer.sublayer[l].dropout(o)
            if l == 1:
                x = layer.sublayer[2](x, layer.feed_forward)
            return {'x': x}
        return func


class HaltingUnit(nn.Module):
    halting_bias_init = 1.0
    def __init__(self, dim_model):
        super(HaltingUnit, self).__init__()
        self.linear = nn.Linear(dim_model, 1)
        self.norm = LayerNorm(dim_model)
        INIT.constant_(self.linear.bias, self.halting_bias_init)

    def forward(self, x):
        return th.sigmoid(self.linear(self.norm(x)))

class UTransformer(nn.Module):
    "Universal Transformer(https://arxiv.org/pdf/1807.03819.pdf) with ACT(https://arxiv.org/pdf/1603.08983.pdf)."
    MAX_DEPTH = 8
    thres = 0.99
    act_loss_weight = 0.01 
    def __init__(self, encoder, decoder, src_embed, tgt_embed, pos_enc, time_enc, generator, h, d_k):
        super(UTransformer, self).__init__()
        self.encoder,  self.decoder = encoder, decoder
        self.src_embed, self.tgt_embed = src_embed, tgt_embed
        self.pos_enc, self.time_enc = pos_enc, time_enc
        self.halt_enc = HaltingUnit(h * d_k)
        self.halt_dec = HaltingUnit(h * d_k)
        self.generator = generator
        self.h, self.d_k = h, d_k
        self.reset_stat()

    def reset_stat(self):
        self.stat = [0] * (self.MAX_DEPTH + 1)

    def step_forward(self, nodes):
        x = nodes.data['x']
        step = nodes.data['step']
        pos = nodes.data['pos']
        return {'x': self.pos_enc.dropout(x + self.pos_enc(pos.view(-1)) + self.time_enc(step.view(-1))),
                'step': step + 1}

    def halt_and_accum(self, name, end=False):
        "field: 'enc' or 'dec'"
        halt = self.halt_enc if name == 'enc' else self.halt_dec
        thres = self.thres
        def func(nodes):
            p = halt(nodes.data['x'])
            sum_p = nodes.data['sum_p'] + p
            active = (sum_p < thres) & (1 - end)
            _continue = active.float()
            r = nodes.data['r'] * (1 - _continue) + (1 - sum_p) * _continue
            s = nodes.data['s'] + ((1 - _continue) * r + _continue * p) * nodes.data['x']
            return {'p': p, 'sum_p': sum_p, 'r': r, 's': s, 'active': active}
        return func

    def propagate_attention(self, g, eids):
        # Compute attention score
        g.apply_edges(src_dot_dst('k', 'q', 'score'), eids)
        g.apply_edges(scaled_exp('score', np.sqrt(self.d_k)), eids)
        # Send weighted values to target nodes
        g.send_and_recv(eids,
                        [fn.u_mul_e('v', 'score', 'v'), fn.copy_e('score', 'score')],
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
        N, E = graph.n_nodes, graph.n_edges
        nids, eids = graph.nids, graph.eids

        # embed & pos
        g.nodes[nids['enc']].data['x'] = self.src_embed(graph.src[0])
        g.nodes[nids['dec']].data['x'] = self.tgt_embed(graph.tgt[0])
        g.nodes[nids['enc']].data['pos'] = graph.src[1]
        g.nodes[nids['dec']].data['pos'] = graph.tgt[1]

        # init step
        device = next(self.parameters()).device
        g.ndata['s'] = th.zeros(N, self.h * self.d_k, dtype=th.float, device=device)    # accumulated state
        g.ndata['p'] = th.zeros(N, 1, dtype=th.float, device=device)                    # halting prob 
        g.ndata['r'] = th.ones(N, 1, dtype=th.float, device=device)                     # remainder
        g.ndata['sum_p'] = th.zeros(N, 1, dtype=th.float, device=device)                # sum of pondering values
        g.ndata['step'] = th.zeros(N, 1, dtype=th.long, device=device)                  # step
        g.ndata['active'] = th.ones(N, 1, dtype=th.uint8, device=device)                # active

        for step in range(self.MAX_DEPTH):
            pre_func = self.encoder.pre_func('qkv')
            post_func = self.encoder.post_func()
            nodes = g.filter_nodes(lambda v: v.data['active'].view(-1), nids['enc'])
            if len(nodes) == 0: break
            edges = g.filter_edges(lambda e: e.dst['active'].view(-1), eids['ee'])
            end = step == self.MAX_DEPTH - 1
            self.update_graph(g, edges,
                              [(self.step_forward, nodes), (pre_func, nodes)],
                              [(post_func, nodes), (self.halt_and_accum('enc', end), nodes)])

        g.nodes[nids['enc']].data['x'] = self.encoder.norm(g.nodes[nids['enc']].data['s'])

        for step in range(self.MAX_DEPTH):
            pre_func = self.decoder.pre_func('qkv')
            post_func = self.decoder.post_func()
            nodes = g.filter_nodes(lambda v: v.data['active'].view(-1), nids['dec'])
            if len(nodes) == 0: break
            edges = g.filter_edges(lambda e: e.dst['active'].view(-1), eids['dd'])
            self.update_graph(g, edges,
                              [(self.step_forward, nodes), (pre_func, nodes)],
                              [(post_func, nodes)])

            pre_q = self.decoder.pre_func('q', 1)
            pre_kv = self.decoder.pre_func('kv', 1)
            post_func = self.decoder.post_func(1)
            nodes_e = nids['enc']
            edges = g.filter_edges(lambda e: e.dst['active'].view(-1), eids['ed'])
            end = step == self.MAX_DEPTH - 1
            self.update_graph(g, edges,
                              [(pre_q, nodes), (pre_kv, nodes_e)],
                              [(post_func, nodes), (self.halt_and_accum('dec', end), nodes)])

        g.nodes[nids['dec']].data['x'] = self.decoder.norm(g.nodes[nids['dec']].data['s'])
        act_loss = th.mean(g.ndata['r']) # ACT loss

        self.stat[0] += N
        for step in range(1, self.MAX_DEPTH + 1):
            self.stat[step] += th.sum(g.ndata['step'] >= step).item()

        return self.generator(g.ndata['x'][nids['dec']]), act_loss * self.act_loss_weight

    def infer(self, *args, **kwargs):
        raise NotImplementedError


def make_universal_model(src_vocab, tgt_vocab, dim_model=512, dim_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h, dim_model)
    ff = PositionwiseFeedForward(dim_model, dim_ff)
    pos_enc = PositionalEncoding(dim_model, dropout)
    time_enc = PositionalEncoding(dim_model, dropout)
    encoder = UEncoder(EncoderLayer((dim_model), c(attn), c(ff), dropout))
    decoder = UDecoder(DecoderLayer((dim_model), c(attn), c(attn), c(ff), dropout))
    src_embed = Embeddings(src_vocab, dim_model)
    tgt_embed = Embeddings(tgt_vocab, dim_model)
    generator = Generator(dim_model, tgt_vocab)
    model = UTransformer(
        encoder, decoder, src_embed, tgt_embed, pos_enc, time_enc, generator, h, dim_model // h)
    # xavier init
    for p in model.parameters():
        if p.dim() > 1:
            INIT.xavier_uniform_(p)
    return model
