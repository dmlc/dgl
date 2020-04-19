import dgl
import torch as th
import numpy as np
import itertools
import time
from collections import *

VertexGraph = namedtuple('Graph',
                   ['g', 'tgt', 'tgt_y', 'nids', 'eids',  'nid_arr', 'n_nodes', 'n_edges', 'n_tokens'])

class VertexNetGraphPool:
    "Create a graph pool in advance to accelerate graph building phase in Transformer."
    def __init__(self, n=300):
        '''
        args:
            n: maximum number of vertexes
        '''
        print('start creating graph pool...')
        tic = time.time()
        self.n = n
        g_pool = [dgl.DGLGraph() for _ in range(n)]
        num_edges = {
            'dd': np.zeros(n).astype(int)
        }
        for i in range(n):
            n_vertex = i + 1

            g_pool[i].add_nodes(n_vertex)
            dec_nodes = th.arange(n_vertex, dtype=th.long)

            # dec -> dec
            indices = th.triu(th.ones(n_vertex, n_vertex)) == 1
            us = dec_nodes.unsqueeze(-1).repeat(1, n_vertex)[indices]
            vs = dec_nodes.unsqueeze(0).repeat(n_vertex, 1)[indices]
            g_pool[i].add_edges(us, vs)
            num_edges['dd'][i] = len(us)

        print('successfully created graph pool, time: {0:0.3f}s'.format(time.time() - tic))
        self.g_pool = g_pool
        self.num_edges = num_edges

    def beam(self, src_buf, start_sym, max_len, k, device='cpu'):
        '''
        Return a batched graph for beam search during inference of Transformer.
        args:
            src_buf: a list of input sequence
            start_sym: the index of start-of-sequence symbol
            max_len: maximum length for decoding
            k: beam size
            device: 'cpu' or 'cuda:*' 
        '''
        g_list = []
        src_lens = [len(_) for _ in src_buf]
        tgt_lens = [max_len] * len(src_buf)
        num_edges = {'ee': [], 'ed': [], 'dd': []}
        for src_len, tgt_len in zip(src_lens, tgt_lens):
            i, j = src_len - 1, tgt_len - 1
            for _ in range(k):
                g_list.append(self.g_pool[i][j])
            for key in ['ee', 'ed', 'dd']:
                num_edges[key].append(int(self.num_edges[key][i][j]))

        g = dgl.batch(g_list)
        src, tgt = [], []
        src_pos, tgt_pos = [], []
        enc_ids, dec_ids = [], []
        e2e_eids, e2d_eids, d2d_eids = [], [], []
        n_nodes, n_edges, n_tokens = 0, 0, 0
        for src_sample, n, n_ee, n_ed, n_dd in zip(src_buf, src_lens, num_edges['ee'], num_edges['ed'], num_edges['dd']):
            for _ in range(k):
                src.append(th.tensor(src_sample, dtype=th.long, device=device))
                src_pos.append(th.arange(n, dtype=th.long, device=device))
                enc_ids.append(th.arange(n_nodes, n_nodes + n, dtype=th.long, device=device))
                n_nodes += n
                e2e_eids.append(th.arange(n_edges, n_edges + n_ee, dtype=th.long, device=device))
                n_edges += n_ee
                tgt_seq = th.zeros(max_len, dtype=th.long, device=device)
                tgt_seq[0] = start_sym
                tgt.append(tgt_seq)
                tgt_pos.append(th.arange(max_len, dtype=th.long, device=device))

                dec_ids.append(th.arange(n_nodes, n_nodes + max_len, dtype=th.long, device=device))
                n_nodes += max_len
                e2d_eids.append(th.arange(n_edges, n_edges + n_ed, dtype=th.long, device=device))
                n_edges += n_ed
        g.set_e_initiddt
        alizer(dgl.init.zero_initializer)

        return Graph(g=g,
                     src=(th.cat(src), th.cat(src_pos)),
                     tgt=(th.cat(tgt), th.cat(tgt_pos)),
                     tgt_y=None,
                     nids = {'enc': th.cat(enc_ids), 'dec': th.cat(dec_ids)},
                     eids = {'ee': th.cat(e2e_eids), 'ed': th.cat(e2d_eids), 'dd': th.cat(d2d_eids)},
                     nid_arr = {'enc': enc_ids, 'dec': dec_ids},
                     n_nodes=n_nodes,
                     n_edges=n_edges,
                     n_tokens=n_tokens)

    def __call__(self, tgt_buf, device='cpu'):
        '''
        Return a batched graph for the training phase of Transformer.
        args:
            tgt_buf: a set of output sequence arrays.
            device: 'cpu' or 'cuda:*'
        '''
        g_list = []
        tgt_lens = [len(_) - 1 for _ in tgt_buf]
        num_edges = {'dd': []}
        for tgt_len in tgt_lens:
            i = tgt_len - 1
            g_list.append(self.g_pool[i])
            num_edges['dd'].append(int(self.num_edges['dd'][i]))

        g = dgl.batch(g_list)
        tgt, tgt_y = [], []
        tgt_pos = []
        dec_ids = []
        d2d_eids = []
        n_nodes, n_edges, n_tokens = 0, 0, 0
        for tgt_sample, n, n_dd in zip(tgt_buf, tgt_lens, num_edges['dd']):
            tgt.append(th.tensor(tgt_sample[:-1], dtype=th.long, device=device))
            tgt_y.append(th.tensor(tgt_sample[1:], dtype=th.long, device=device))
            tgt_pos.append(th.arange(n, dtype=th.long, device=device))
            dec_ids.append(th.arange(n_nodes, n_nodes + n, dtype=th.long, device=device))
            d2d_eids.append(th.arange(n_edges, n_edges + n_dd, dtype=th.long, device=device))
            n_nodes += n
            n_tokens += n
            n_edges += n_dd

        g.set_n_initializer(dgl.init.zero_initializer)
        g.set_e_initializer(dgl.init.zero_initializer)

        return VertexGraph(g=g,
                     tgt=(th.cat(tgt), th.cat(tgt_pos)),
                     tgt_y=th.cat(tgt_y),
                     nids = {'dec': th.cat(dec_ids)},
                     eids = {'dd': th.cat(d2d_eids)},
                     nid_arr = {'dec': dec_ids},
                     n_nodes=n_nodes,
                     n_edges=n_edges,
                     n_tokens=n_tokens)
