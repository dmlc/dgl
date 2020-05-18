from .attention import *
from .layers import *
from .embedding import *
from dataset.datasets import VertexDataset
import threading
import torch as th
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn.init as INIT

class Decoder(nn.Module):
    def __init__(self, layer, N):
        '''
        args:
            N: number of blocks.
        '''
        super(Decoder, self).__init__()
        self.N = N
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def pre_func(self, i, fields='qkv'):
        layer = self.layers[i]
        def func(nodes):
            x = nodes.data['x']
            norm_x = layer.sublayer[0].norm(x)
            return layer.self_attn.get(norm_x, fields)
        return func

    def post_func(self, i):
        layer = self.layers[i]
        def func(nodes):
            x, z = nodes.data['x'], nodes.data['z']
            o = layer.self_attn.get_o(z)
            x = x + layer.sublayer[0].dropout(o)
            x = layer.sublayer[1](x, layer.feed_forward)
            # We apply norm at the begining for each block. Then output of transformer need an additional norm.
            return {'x': x if i < self.N - 1 else self.norm(x)}
        return func

class Transformer(nn.Module):
    def __init__(self, decoder, coord_embed, pos_embed, value_embed, generator, h, d_k):
        '''
        args:
            decoder: decoder module
            coord_embed: coordinate embedding module
            pos_embed: position encoding module
            value_embed: value embedding module
            generator: generate next token from representation
            h: number of heads
            d_k: dim per head
        '''
        super(Transformer, self).__init__()
        self.decoder = decoder
        self.coord_embed = coord_embed
        self.pos_embed = pos_embed
        self.value_embed = value_embed
        self.generator = generator
        self.h, self.d_k = h, d_k
        self.att_weight_map = None

    def propagate_attention(self, g, eids):
        # Compute attention score
        g.apply_edges(fn.u_dot_v('k', 'q', 'score'), eids)
        # Softmax
        g.edata['softmax_score'] = th.zeros_like(g.edata['score'].unsqueeze(-1))
        g.edata['softmax_score'][eids] = dglnn.edge_softmax(g, g.edata['score'][eids], eids).unsqueeze(-1)
        # sum up v
        g.send_and_recv(eids,
                        [fn.u_mul_e('v', 'softmax_score', 'z')],
                        [fn.sum('z', 'z')])

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
        coord_embed = self.coord_embed(graph.tgt[1]%3)
        pos_embed = self.pos_embed(graph.tgt[1]//3)
        value_embed = self.value_embed(graph.tgt[0]) 
        g.nodes[nids['dec']].data['x'] = coord_embed + pos_embed + value_embed

        for i in range(self.decoder.N):
            pre_func = self.decoder.pre_func(i, 'qkv')
            post_func = self.decoder.post_func(i)
            nodes, edges = nids['dec'], eids['dd']
            self.update_graph(g, edges, [(pre_func, nodes)], [(post_func, nodes)])

        return self.generator(g.ndata['x'][nids['dec']])

    def infer(self, graph, max_len, eos_id):
        '''
        This function implements nucleus sampling in DGL, which is required in inference phase.
        args:
            graph: a `Graph` object defined in `dgl.contrib.transformer.graph`.
            max_len: the maximum length of decoding.
            eos_id: the index of end-of-sequence symbol.
        return:
            ret: a list of index array correspond to the input sequence specified by `graph``.
        '''
        g = graph.g
        N, E = graph.n_nodes, graph.n_edges
        nids, eids = graph.nids, graph.eids
        
        # init mask
        device = next(self.parameters()).device

        g.ndata['mask'] = th.zeros(N, dtype=th.bool, device=device)
        g.nodes[nids['dec']].data['pos'] = graph.tgt[1]
        coord_embed = self.coord_embed(graph.tgt[1]%3)
        pos_embed = self.pos_embed(graph.tgt[1]//3) 
        res = graph.tgt_y

        # decode
        log_prob = None
        y = graph.tgt[0]
        for step in range(1, max_len):
            y = y.view(-1)
            value_embed = self.value_embed(y) 
            g.nodes[nids['dec']].data['x'] = self.pos_embed.dropout(coord_embed + pos_embed + value_embed)
            edges_dd = g.filter_edges(lambda e: (e.dst['pos'] < step) & ~e.dst['mask'], eids['dd'])
            nodes_d = g.filter_nodes(lambda v: (v.data['pos'] < step) & ~v.data['mask'], nids['dec'])

            for i in range(self.decoder.N):
                pre_func = self.decoder.pre_func(i, 'qkv')
                post_func = self.decoder.post_func(i)
                nodes, edges = nids['dec'], eids['dd']
                self.update_graph(g, edges, [(pre_func, nodes)], [(post_func, nodes)])

            frontiers = g.filter_nodes(lambda v: v.data['pos'] == step - 1, nids['dec'])
            out = self.generator(g.ndata['x'][frontiers])
            batch_size = frontiers.shape[0] // k
            vocab_size = out.shape[-1]
            # Mask output for complete sequence
            one_hot = th.zeros(vocab_size).fill_(-1e9).to(device)
            one_hot[eos_id] = 0
            mask = g.ndata['mask'][frontiers].unsqueeze(-1).float()
            out = out * (1 - mask) + one_hot.unsqueeze(0) * mask

            if log_prob is None:
                log_prob, pos = out.view(batch_size, k, -1)[:, 0, :].topk(k, dim=-1)
                eos = th.zeros(batch_size, k).bool()
            else:
                norm_old = eos.float().to(device) + (1 - eos.float().to(device)) * np.power((4. + step) / 6, alpha)
                norm_new = eos.float().to(device) + (1 - eos.float().to(device)) * np.power((5. + step) / 6, alpha)
                log_prob, pos = ((out.view(batch_size, k, -1) + (log_prob * norm_old).unsqueeze(-1)) / norm_new.unsqueeze(-1)).view(batch_size, -1).topk(k, dim=-1)

            _y = y.view(batch_size * k, -1)
            y = th.zeros_like(_y)
            _eos = eos.clone()
            for i in range(batch_size):
                for j in range(k):
                    _j = pos[i, j].item() // vocab_size
                    token = pos[i, j].item() % vocab_size
                    y[i*k+j, :] = _y[i*k+_j, :]
                    y[i*k+j, step] = token
                    eos[i, j] = _eos[i, _j] | (token == eos_id)
            if eos.all():
                break
            else:
                g.ndata['mask'][nids['dec']] = eos.unsqueeze(-1).repeat(1, 1, max_len).view(-1).to(device)
        return y.view(batch_size, k, -1)[:, 0, :].tolist()

    def _register_att_map(self, g, enc_ids, dec_ids):
        self.att_weight_map = [
            get_attention_map(g, enc_ids, enc_ids, self.h),
            get_attention_map(g, enc_ids, dec_ids, self.h),
            get_attention_map(g, dec_ids, dec_ids, self.h),
        ]


def make_vertex_model(N=18, dim_model=256, dim_ff=1024, h=8, dropout=0.1, universal=False):
    '''
    args:
        N: transformer block number.
        dim_model: hidden layer dimention of transformer.
        dim_ff: feedforward layer dimention.
        h: number of head for multihead attention
        dropout: dropout rate
    '''
    c = copy.deepcopy
    attn = MultiHeadAttention(h, dim_model)
    ff = PositionwiseFeedForward(dim_model, dim_ff)
    # coord only have x, y, z
    # According to the paper Child, et 19, learning embedding works well
    coord_embed = Embeddings(3, dim_model)
    pos_embed = Embeddings(VertexDataset.MAX_VERT_LENGTH+1, dim_model)
    # Do we need to consider INIT_BIN?
    tgt_vocab = VertexDataset.COORD_BIN + 3
    value_embed = Embeddings(tgt_vocab, dim_model)
    decoder = Decoder(DecoderLayer(dim_model, c(attn), None, c(ff), dropout), N)
    generator = VertexGenerator(dim_model, tgt_vocab)
    model = Transformer(
        decoder, coord_embed, pos_embed, value_embed, generator, h, dim_model // h)
    # xavier init
    for p in model.parameters():
        if p.dim() > 1:
            INIT.xavier_uniform_(p)
    return model