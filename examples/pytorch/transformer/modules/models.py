from .layers import *
from .functions import *
import torch as th

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def prep(self, i, fields='qkv'):
        layer = self.layers[i]
        def func(nodes):
            x = nodes.data['x']
            norm_x = layer.sublayer[0].norm(x)
            return layer.self_attn.prep(norm_x, fields=fields)
        return func

    def upd(self, i):
        layer = self.layers[i]
        h = layer.self_attn.h
        def func(nodes):
            x, wv, z = nodes.data['x'], nodes.data['wv'], nodes.data['z']
            o = layer.self_attn.get_o(wv / z)
            x = x + layer.sublayer[0].dropout(o)
            x = layer.sublayer[1](x, layer.feed_forward)
            return {'x': x if i < self.N - 1 else self.norm(x)}
        return func

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def prep(self, i, fields='qkv'):
        layer = self.layers[i]
        def func(nodes):
            x = nodes.data['x']
            norm_x = layer.sublayer[0].norm(x)
            return layer.self_attn.prep(norm_x, fields)
        return func

    def upd(self, i, l=0):
        layer = self.layers[i]
        def func(nodes):
            x, wv, z = nodes.data['x'], nodes.data['wv'], nodes.data['z']
            o = layer.self_attn.get_o(wv / z)
            x = x + layer.sublayer[l].dropout(o)
            if l == 1:
                x = layer.sublayer[2](x, layer.feed_forward)
            return {'x': x if i < self.N - 1 else self.norm(x)}
        return func

class Generator(nn.Module):
    """
    Generate next token from the representation. This part is separated from the decoder, mostly for the convenience of sharing weight between embedding and generator. """
    def __init__(self, dim_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(dim_model, vocab_size)

    def forward(self, x):
        return T.log_softmax(
            self.proj(x), dim=-1
        )

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, h, d_k, universal=False):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.h = h
        self.d_k = d_k

    def forward_layer(self, g, pairs, upd_func, edges):
        # Prepare queries and key-value pairs.
        for prep_i, nodes_i in pairs:
            g.apply_nodes(prep_i, nodes_i)
        # Compute attention score
        g.apply_edges(src_dot_dst('k', 'q', 'score'), edges)
        g.apply_edges(scaled_exp('score', np.sqrt(self.d_k)))
        # Update node state
        g.send_and_recv(edges,
                        [fn.src_mul_edge('v', 'score', 'v'), fn.copy_edge('score', 'score') ],
                        [fn.sum('v', 'wv'), fn.sum('score', 'z')],
                        upd_func)

    def forward(self, graph):
        g = graph.g
        N = graph.n_nodes
        E = graph.n_edges
        h = self.h
        d_k = self.d_k

        # init
        try:
            device = next(self.parameters()).get_device()
        except RuntimeError:
            device = th.device('cpu')
        g.ndata['x'] = th.zeros(N, h * d_k, device=device)
        g.edata['score'] = th.zeros(E, h, 1, device=device)
        # embed
        g.nodes[graph.nids['enc']].data['x'] = self.src_embed[1].dropout(sum(self.src_embed[i](graph.src[i]) for i in range(2)))
        g.nodes[graph.nids['dec']].data['x'] = self.tgt_embed[1].dropout(sum(self.tgt_embed[i](graph.tgt[i]) for i in range(2)))

        for i in range(self.encoder.N):
            prep, upd = self.encoder.prep(i, 'qkv'), self.encoder.upd(i)
            nodes, edges = graph.nids['enc'], graph.eids['ee']
            self.forward_layer(g, [(prep, nodes)], upd, edges)

        for i in range(self.decoder.N):
            prep, upd = self.decoder.prep(i, 'qkv'), self.decoder.upd(i, 0)
            nodes, edges = graph.nids['dec'], graph.eids['dd']
            self.forward_layer(g, [(prep, nodes)], upd, edges)
            prep_q, prep_kv, upd = self.decoder.prep(i, 'q'), self.decoder.prep(i, 'kv'), self.decoder.upd(i, 1)
            nodes_e, nodes_d, edges = graph.nids['enc'], graph.nids['dec'], graph.eids['ed']
            self.forward_layer(g, [(prep_q, nodes_d), (prep_kv, nodes_e)], upd, edges)

        return self.generator(g.nodes[graph.nids['dec']].data['x'])

    def infer(self, graph, max_len, eos_id, k): # k: beam size
        g = graph.g
        N = graph.n_nodes
        E = graph.n_edges
        h = self.h
        d_k = self.d_k

        # init
        try:
            device = next(self.parameters()).get_device()
        except RuntimeError:
            device = th.device('cpu')
        g.ndata['x'] = th.zeros(N, h * d_k, device=device)
        g.edata['score'] = th.zeros(E, h, 1, device=device)
        g.ndata['pos'] = th.zeros(N, dtype=th.long, device=device)
        g.ndata['mask'] = th.zeros(N, dtype=th.uint8, device=device)

        # embed
        src_embed = self.src_embed[0](graph.src[0])
        src_pos = self.src_embed[1](graph.src[1])
        g.nodes[graph.nids['enc']].data['pos'] = graph.src[1]
        g.nodes[graph.nids['enc']].data['x'] = self.src_embed[1].dropout(src_embed + src_pos)
        tgt_pos = self.tgt_embed[1](graph.tgt[1])
        g.nodes[graph.nids['dec']].data['pos'] = graph.tgt[1]

        for i in range(self.encoder.N):
            prep, upd = self.encoder.prep(i, 'qkv'), self.encoder.upd(i)
            nodes, edges = graph.nids['enc'], graph.eids['ee']
            self.forward_layer(g, [(prep, nodes)], upd, edges)

        log_prob = None
        y = graph.tgt[0]
        for step in range(1, max_len):
            y = y.view(-1)
            tgt_embed = self.tgt_embed[0](y)
            g.nodes[graph.nids['dec']].data['x'] = self.tgt_embed[1].dropout(tgt_embed + tgt_pos)
            edges_ed = g.filter_edges(lambda e: (e.dst['pos'] < step) & ~e.dst['mask'] , graph.eids['ed'])
            edges_dd = g.filter_edges(lambda e: (e.dst['pos'] < step) & ~e.dst['mask'], graph.eids['dd'])
            nodes_d = g.filter_nodes(lambda v: (v.data['pos'] < step) & ~v.data['mask'], graph.nids['dec'])
            for i in range(self.decoder.N):
                prep, upd = self.decoder.prep(i, 'qkv'), self.decoder.upd(i, 0)
                nodes, edges = nodes_d, edges_dd
                self.forward_layer(g, [(prep, nodes)], upd, edges)
                prep_q, prep_kv, upd = self.decoder.prep(i, 'q'), self.decoder.prep(i, 'kv'), self.decoder.upd(i, 1)
                nodes_e, nodes_d, edges = graph.nids['enc'], nodes_d, edges_ed
                self.forward_layer(g, [(prep_q, nodes_d), (prep_kv, nodes_e)], upd, edges)

            frontiers = g.filter_nodes(lambda v: v.data['pos'] == step - 1, graph.nids['dec'])
            out = self.generator(g.nodes[frontiers].data['x'])
            batch_size = frontiers.shape[0] // k
            vocab_size = out.shape[-1]
            if log_prob is None:
                log_prob, pos = out.view(batch_size, k, -1)[:, 0, :].topk(k, dim=-1)
                eos = th.zeros(batch_size).byte()
            else:
                log_prob, pos = (out.view(batch_size, k, -1) + log_prob.unsqueeze(-1)).view(batch_size, -1).topk(k, dim=-1)

            _y = y.view(batch_size * k, -1)
            y = th.zeros_like(_y)
            for i in range(batch_size):
                if not eos[i]:
                    for j in range(k):
                        _j = pos[i, j].item() // vocab_size
                        token = pos[i, j].item() % vocab_size
                        y[i*k+j,:] = _y[i*k+_j, :]
                        y[i*k+j, step] = token
                        if j == 0:
                            eos[i] = eos[i] | (token == eos_id)
                else:
                    y[i*k:(i+1)*k, :] = _y[i*k:(i+1)*k, :]

            if eos.all():
                break
            else:
                g.nodes[graph.nids['dec']].data['mask'] = eos.unsqueeze(-1).repeat(1, k * max_len).view(-1).to(device)
        return y.view(batch_size, k, -1)[:, 0, :].tolist()

from .attention import *
from .embedding import *
import torch.nn.init as INIT

def make_model(src_vocab, tgt_vocab, N=6,
                   dim_model=512, dim_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h, dim_model, dropout)
    ff = PositionwiseFeedForward(dim_model, dim_ff, dropout)
    pos_enc = PositionalEncoding(dim_model, dropout)

    encoder = Encoder(
        EncoderLayer(dim_model, c(attn), c(ff), dropout), N
    )
    decoder = Decoder(
        DecoderLayer(dim_model, c(attn), c(attn), c(ff), dropout), N
    )
    src_embed = nn.ModuleList([
        Embeddings(src_vocab, dim_model),
        c(pos_enc)
    ])
    tgt_embed = nn.ModuleList([
        Embeddings(tgt_vocab, dim_model),
        c(pos_enc)
    ])
    generator = Generator(dim_model, tgt_vocab)
    model = Transformer(
        encoder, decoder, src_embed, tgt_embed, generator, h, dim_model // h)
    # xavier init
    for p in model.parameters():
        if p.dim() > 1:
            INIT.xavier_uniform_(p)
    return model
