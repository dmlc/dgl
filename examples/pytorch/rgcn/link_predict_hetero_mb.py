"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn

Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""

import argparse
import itertools
import numpy as np
import time, gc
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dataset import load_entity
from functools import partial

from dgl.contrib.data import load_data
from model import RelGraphEmbedLayer, RelGraphConvLayer
from utils import build_heterograph_in_homogeneous_from_triplets, build_multi_ntype_heterograph_in_homogeneous_from_triplets
from utils import thread_wrapped_func
from utils import DrkgDataset

class LinkPredict(nn.Module):
    def __init__(self,
                 device,
                 h_dim,
                 num_rels,
                 edge_rels,
                 num_bases=-1,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False,
                 low_mem=False,
                 regularization_coef=0):
        super(LinkPredict, self).__init__()
        self.device = th.device(device if device >= 0 else 'cpu')
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.edge_rels = edge_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.regularization_coef = regularization_coef

        self.w_relation = nn.Parameter(th.Tensor(num_rels, h_dim).to(self.device))
        nn.init.xavier_uniform_(self.w_relation)
        
        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConvLayer(
            self.h_dim, self.h_dim, self.edge_rels, "basis",
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            low_mem=low_mem, dropout=self.dropout))
        # h2h
        for idx in range(self.num_hidden_layers):
            self.layers.append(RelGraphConvLayer(
                self.h_dim, self.h_dim, self.edge_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                low_mem=low_mem, dropout=self.dropout))

    def forward_full(self, blocks, feats):
        h = feats
        for layer, block in zip(self.layers, blocks):
            block = block.to(self.device)
            h = layer(block, h)

        return h

    def forward(self, p_blocks, p_feats, n_blocks=None, n_feats=None, p_g=None, n_g=None):
        p_h = p_feats
        for layer, block in zip(self.layers, p_blocks):
            block = block.to(self.device)
            p_h = layer(block, p_h)

        if n_blocks is None:
            return p_h

        n_h = n_feats
        for layer, block in zip(self.layers, n_blocks):
            block = block.to(self.device)
            n_h = layer(block, n_h)

        head, tail, eid = p_g.all_edges(form='all')
        p_head_emb = p_h[head]
        p_tail_emb = p_h[tail]
        rids = p_g.edata['etype']
        r_emb = self.w_relation[rids]
        head, tail = n_g.all_edges()
        n_head_emb = n_h[head]
        n_tail_emb = n_h[tail]
        return (p_head_emb, p_tail_emb, r_emb, n_head_emb, n_tail_emb)

def regularization_loss(h_emb, t_emb, r_emb, nh_emb, nt_emb):
    return th.mean(h_emb.pow(2)) + \
            th.mean(t_emb.pow(2)) + \
            th.mean(nh_emb.pow(2)) + \
            th.mean(nt_emb.pow(2)) + \
            th.mean(r_emb.pow(2))

def calc_pos_score(h_emb, t_emb, r_emb):
    # DistMult
    score = th.sum(h_emb * r_emb * t_emb, dim=-1)
    return score

def calc_neg_tail_score(heads, tails, r_emb, num_chunks, chunk_size, neg_sample_size, device=None):
    hidden_dim = heads.shape[1]
    r = r_emb

    if device is not None:
        r = r.to(device)
        heads = heads.to(device)
        tails = tails.to(device)
    tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
    tails = th.transpose(tails, 1, 2)
    tmp = (heads * r).reshape(num_chunks, chunk_size, hidden_dim)
    return th.bmm(tmp, tails)

def calc_neg_head_score(heads, tails, r_emb, num_chunks, chunk_size, neg_sample_size, device=None):
    hidden_dim = tails.shape[1]
    r = r_emb
    if device is not None:
        r = r.to(device)
        heads = heads.to(device)
        tails = tails.to(device)
    heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
    heads = th.transpose(heads, 1, 2)
    tmp = (tails * r).reshape(num_chunks, chunk_size, hidden_dim)
    return th.bmm(tmp, heads)

def get_loss(h_emb, t_emb, nh_emb, nt_emb, r_emb,
    num_chunks, chunk_size, neg_sample_size, regularization_coef):
    # triplets is a list of data samples (positive and negative)
    # each row in the triplets is a 3-tuple of (source, relation, destination)
    pos_score = calc_pos_score(h_emb, t_emb, r_emb)
    t_neg_score = calc_neg_tail_score(h_emb, nt_emb, r_emb, num_chunks, chunk_size, neg_sample_size)
    h_neg_score = calc_neg_head_score(nh_emb, t_emb, r_emb, num_chunks, chunk_size, neg_sample_size)
    pos_score = F.logsigmoid(pos_score)
    h_neg_score = h_neg_score.reshape(-1, neg_sample_size)
    t_neg_score = t_neg_score.reshape(-1, neg_sample_size)
    h_neg_score = F.logsigmoid(-h_neg_score).mean(dim=1)
    t_neg_score = F.logsigmoid(-t_neg_score).mean(dim=1)

    pos_score = pos_score.mean()
    h_neg_score = h_neg_score.mean()
    t_neg_score = t_neg_score.mean()
    predict_loss = -(2 * pos_score + h_neg_score + t_neg_score)

    reg_loss = regularization_loss(h_emb, t_emb, r_emb, nh_emb, nt_emb)

    print("pos loss {}, neg loss {}|{}, reg_loss {}".format(pos_score.detach(),
                                                            h_neg_score.detach(),
                                                            t_neg_score.detach(),
                                                            regularization_coef * reg_loss.detach()))
    return predict_loss + regularization_coef * reg_loss

class NodeSampler:
    def __init__(self, g, fanouts):
        self.g = g
        self.fanouts = fanouts

    def sample_nodes(self, seeds):
        seeds = th.tensor(seeds).long()
        curr = seeds

        blocks = []
        for fanout in self.fanouts:
            if fanout is None or fanout == -1:
                frontier = dgl.in_subgraph(self.g, curr)
            else:
                frontier = dgl.sampling.sample_neighbors(self.g, curr, fanout)
            etypes = self.g.edata['etype'][frontier.edata[dgl.EID]]
            norm = self.g.edata['norm'][frontier.edata[dgl.EID]]
            block = dgl.to_block(frontier, curr)
            block.srcdata['ntype'] = self.g.ndata['ntype'][block.srcdata[dgl.NID]]
            block.edata['etype'] = etypes
            block.edata['norm'] = norm
            curr = block.srcdata[dgl.NID]
            blocks.insert(0, block)

        return blocks
            

class LinkRankSampler:
    def __init__(self, g, neg_edges, num_edges, num_neg, fanouts, 
        is_train=True, keep_pos_edges=False):
        self.g = g
        self.neg_edges = neg_edges
        self.num_edges = num_edges
        self.num_neg = num_neg
        self.fanouts = fanouts
        self.is_train = is_train
        self.keep_pos_edges = keep_pos_edges

    def sample_blocks(self, seeds):
        pseeds = th.tensor(seeds).long()
        bsize = pseeds.shape[0]
        if self.num_neg is not None:
            nseeds = th.randint(self.num_edges, (self.num_neg,))
            nseeds = self.neg_edges[nseeds]
        else:
            nseeds = th.randint(self.num_edges, (bsize,))
            nseeds = self.neg_edges[nseeds]

        g = self.g
        fanouts = self.fanouts
        assert len(g.canonical_etypes) == 1
        p_subg = g.edge_subgraph({g.canonical_etypes[0] : pseeds})
        n_subg = g.edge_subgraph({g.canonical_etypes[0] : nseeds})

        p_g = dgl.compact_graphs(p_subg)
        n_g = dgl.compact_graphs(n_subg)
        p_g.edata['etype'] = g.edata['etype'][p_subg.edata[dgl.EID]]

        pg_seed = p_subg.ndata[dgl.NID][p_g.ndata[dgl.NID]]
        ng_seed = n_subg.ndata[dgl.NID][n_g.ndata[dgl.NID]]

        p_blocks = []
        n_blocks = []
        p_curr = pg_seed
        n_curr = ng_seed
        for i, fanout in enumerate(fanouts):
            if fanout is None:
                p_frontier = dgl.in_subgraph(g, p_curr)
                n_frontier = dgl.in_subgraph(g, n_curr)
            else:
                p_frontier = dgl.sampling.sample_neighbors(g, p_curr, fanout)
                n_frontier = dgl.sampling.sample_neighbors(g, n_curr, fanout)

            '''
            if self.keep_pos_edges is False and \
                self.is_train and i == 0:
                old_frontier = p_frontier
                p_frontier = dgl.remove_edges(old_frontier, pseeds)
            '''
            p_etypes = g.edata['etype'][p_frontier.edata[dgl.EID]]
            # print(p_etypes)
            n_etypes = g.edata['etype'][n_frontier.edata[dgl.EID]]
            p_norm = g.edata['norm'][p_frontier.edata[dgl.EID]]
            n_norm = g.edata['norm'][n_frontier.edata[dgl.EID]]
            p_block = dgl.to_block(p_frontier, p_curr)
            n_block = dgl.to_block(n_frontier, n_curr)
            p_block.srcdata['ntype'] = g.ndata['ntype'][p_block.srcdata[dgl.NID]]
            n_block.srcdata['ntype'] = g.ndata['ntype'][n_block.srcdata[dgl.NID]]
            p_block.edata['etype'] = p_etypes
            n_block.edata['etype'] = n_etypes
            p_block.edata['norm'] = p_norm
            n_block.edata['norm'] = n_norm
            p_curr = p_block.srcdata[dgl.NID]
            n_curr = n_block.srcdata[dgl.NID]
            p_blocks.insert(0, p_block)
            n_blocks.insert(0, n_block)

        return (bsize, p_g, n_g, p_blocks, n_blocks)

def evaluate(embed_layer, model, dataloader, node_feats, bsize, neg_cnt, queue):
    logs = []
    model.eval()
    t0 = time.time()

    with th.no_grad():
        for i, sample_data in enumerate(dataloader):
            bsize, p_g, n_g, p_blocks, n_blocks = sample_data
            p_feats = embed_layer(p_blocks[0].srcdata[dgl.NID],
                                p_blocks[0].srcdata['ntype'],
                                node_feats)
            n_feats = embed_layer(n_blocks[0].srcdata[dgl.NID],
                                n_blocks[0].srcdata['ntype'],
                                node_feats)
            p_head_emb, p_tail_emb, r_emb, n_head_emb, n_tail_emb = \
                model(p_blocks, p_feats, n_blocks, n_feats, p_g, n_g)

            pos_score = calc_pos_score(p_head_emb, p_tail_emb, r_emb)
            t_neg_score = calc_neg_tail_score(p_head_emb,
                                              n_tail_emb,
                                              r_emb,
                                              1,
                                              bsize,
                                              neg_cnt)
            h_neg_score = calc_neg_head_score(n_head_emb,
                                              p_tail_emb,
                                              r_emb,
                                              1,
                                              bsize,
                                              neg_cnt)
            pos_scores = F.logsigmoid(pos_score).reshape(bsize, -1)
            t_neg_score = F.logsigmoid(t_neg_score).reshape(bsize, neg_cnt)
            h_neg_score = F.logsigmoid(h_neg_score).reshape(bsize, neg_cnt)
            neg_scores = th.cat([h_neg_score, t_neg_score], dim=1)
            rankings = th.sum(neg_scores >= pos_scores, dim=1) + 1
            rankings = rankings.cpu().detach().numpy()
            for ranking in rankings:
                logs.append({
                    'MRR': 1.0 / ranking,
                    'MR': float(ranking),
                    'HITS@1': 1.0 if ranking <= 1 else 0.0,
                    'HITS@3': 1.0 if ranking <= 3 else 0.0,
                    'HITS@10': 1.0 if ranking <= 10 else 0.0
                })

    t1 = time.time()
    print("Eval {} samples takes {} seconds".format(i * bsize, t1 - t0))

    metrics = {}
    if len(logs) > 0:
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
    if queue is not None:
        queue.put(logs)
    else:
        for k, v in metrics.items():
            print('[{}]{} average {}: {}'.format(rank, mode, k, v))

def fullgraph_eval(g, embed_layer, model, device, node_feats, dim_size,
    minibatch_blocks, pos_eids, neg_eids, neg_cnt, queue):
    model.eval()
    t0 = time.time()

    logs = []
    embs = th.empty(g.number_of_nodes(), dim_size)
    with th.no_grad():
        for i, blocks in enumerate(minibatch_blocks):
            in_feats = embed_layer(blocks[0].srcdata[dgl.NID].to(device),
                                   blocks[0].srcdata['ntype'].to(device),
                                   node_feats)
            mb_feats = model(blocks, in_feats)
            embs[blocks[-1].dstdata[dgl.NID]] = mb_feats.cpu()

        mrr = 0
        mr = 0
        hit1 = 0
        hit3 = 0
        hit10 = 0
        pos_batch_size = 1024
        pos_cnt = pos_eids.shape[0]
        total_cnt = 0

        u, v = g.find_edges(pos_eids)
        # randomly select neg_cnt edges and corrupt them int neg heads and neg tails
        if neg_cnt > 0:
            total_neg_seeds = th.randint(neg_eids.shape[0], (neg_cnt * ((pos_cnt // pos_batch_size) + 1),))
        # treat all nodes in head or tail as neg nodes
        else:
            n_u, n_v = g.find_edges(neg_eids)
            n_u = th.unique(n_u)
            n_v = th.unique(n_v)

        for p_i in range(int((pos_cnt + pos_batch_size - 1) // pos_batch_size)):
            print("Eval {}-{}".format(p_i * pos_batch_size,
                                      (p_i + 1) * pos_batch_size \
                                      if (p_i + 1) * pos_batch_size < pos_cnt \
                                      else pos_cnt))
            eids = pos_eids[p_i * pos_batch_size : \
                            (p_i + 1) * pos_batch_size \
                            if (p_i + 1) * pos_batch_size < pos_cnt \
                            else pos_cnt]
            su = u[p_i * pos_batch_size : \
                   (p_i + 1) * pos_batch_size \
                   if (p_i + 1) * pos_batch_size < pos_cnt \
                   else pos_cnt]
            sv = v[p_i * pos_batch_size : \
                   (p_i + 1) * pos_batch_size \
                   if (p_i + 1) * pos_batch_size < pos_cnt \
                   else pos_cnt]

            eids_t = g.edata['etype'][eids].to(device)
            phead_emb = embs[su].to(device)
            ptail_emb = embs[sv].to(device)
            rel_emb = model.module.w_relation[eids_t]

            pos_score = calc_pos_score(phead_emb, ptail_emb, rel_emb)
            pos_score = F.logsigmoid(pos_score).reshape(phead_emb.shape[0], -1)

            if neg_cnt > 0:
                n_eids = neg_eids[total_neg_seeds[p_i * neg_cnt:(p_i + 1) * neg_cnt]]
                n_u, n_v = g.find_edges(n_eids)

            neg_batch_size = 10000
            head_neg_cnt = n_u.shape[0]
            tail_neg_cnt = n_v.shape[0]
            t_neg_score = []
            h_neg_score = []
            for n_i in range(int((head_neg_cnt + neg_batch_size - 1)//neg_batch_size)):
                sn_u = n_u[n_i * neg_batch_size : \
                           (n_i + 1) * neg_batch_size \
                           if (n_i + 1) * neg_batch_size < head_neg_cnt
                           else head_neg_cnt]
                nhead_emb = embs[sn_u].to(device)
                h_neg_score.append(calc_neg_head_score(nhead_emb,
                                                       ptail_emb,
                                                       rel_emb,
                                                       1,
                                                       ptail_emb.shape[0],
                                                       nhead_emb.shape[0]).reshape(-1, nhead_emb.shape[0]))

            for n_i in range(int((tail_neg_cnt + neg_batch_size - 1)//neg_batch_size)):
                sn_v = n_v[n_i * neg_batch_size : \
                           (n_i + 1) * neg_batch_size \
                           if (n_i + 1) * neg_batch_size < tail_neg_cnt
                           else tail_neg_cnt]
                ntail_emb = embs[sn_v].to(device)
                t_neg_score.append(calc_neg_tail_score(phead_emb,
                                                       ntail_emb,
                                                       rel_emb,
                                                       1,
                                                       phead_emb.shape[0],
                                                       ntail_emb.shape[0]).reshape(-1, ntail_emb.shape[0]))
            t_neg_score = th.cat(t_neg_score, dim=1)
            h_neg_score = th.cat(h_neg_score, dim=1)
            t_neg_score = F.logsigmoid(t_neg_score)
            h_neg_score = F.logsigmoid(h_neg_score)

            # exclude false neg edges
            for idx in range(eids.shape[0]):
                tail_pos = g.has_edges_between(th.full((n_v.shape[0],), su[idx], dtype=th.long), n_v)
                head_pos = g.has_edges_between(n_u, th.full((n_u.shape[0],), sv[idx], dtype=th.long))
                etype = g.edata['etype'][eids[idx]]

                loc = tail_pos == 1
                u_ec = su[idx]
                n_v_ec = n_v[loc]
                false_neg_comp = th.full((n_v_ec.shape[0],), 0, device=device)
                for idx_ec in range(n_v_ec.shape[0]):
                    sn_v_ec = n_v_ec[idx_ec]
                    eid_ec = g.edge_id(u_ec, sn_v_ec, return_array=True)
                    etype_ec = g.edata['etype'][eid_ec]
                    loc_ec = etype_ec == etype
                    eid_ec = eid_ec[loc_ec]
                    # has edge
                    if eid_ec.shape[0] > 0:
                        false_neg_comp[idx_ec] = pos_score[idx]
                t_neg_score[idx][loc] += false_neg_comp

                loc = head_pos == 1
                n_u_ec = n_u[loc]
                v_ec = sv[idx]
                false_neg_comp = th.full((n_u_ec.shape[0],), 0, device=device)
                for idx_ec in range(n_u_ec.shape[0]):
                    sn_u_ec = n_u_ec[idx_ec]
                    eid_ec = g.edge_id(sn_u_ec, v_ec, return_array=True)
                    etype_ec = g.edata['etype'][eid_ec]
                    loc_ec = etype_ec == etype
                    eid_ec = eid_ec[loc_ec]
                    # has edge
                    if eid_ec.shape[0] > 0:
                        false_neg_comp[idx_ec] = pos_score[idx]
                h_neg_score[idx][loc] += false_neg_comp

            neg_score = th.cat([h_neg_score, t_neg_score], dim=1)
            rankings = th.sum(neg_score >= pos_score, dim=1) + 1
            rankings = rankings.cpu().detach().numpy()
            for ranking in rankings:
                logs.append({
                    'MRR': 1.0 / ranking,
                    'MR': float(ranking),
                    'HITS@1': 1.0 if ranking <= 1 else 0.0,
                    'HITS@3': 1.0 if ranking <= 3 else 0.0,
                    'HITS@10': 1.0 if ranking <= 10 else 0.0
                })

    metrics = {}
    if len(logs) > 0:
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
    if queue is not None:
        queue.put(logs)
    else:
        for k, v in metrics.items():
            print('[{}]{} average {}: {}'.format(rank, mode, k, v))
    t1 = time.time()
    print("Full eval {} exmpales takes {} seconds".format(pos_eids.shape[0], t1 - t0))

@thread_wrapped_func
def run(proc_id, n_gpus, args, devices, dataset, pos_seeds, neg_seeds, queue=None):
    dev_id = devices[proc_id]
    train_g, valid_g, test_g, num_rels, edge_rels = dataset
    train_seeds, valid_seeds, test_seeds = pos_seeds
    train_neg_seeds, valid_neg_seeds, test_neg_seeds = neg_seeds

    batch_size = args.batch_size
    chunk_size = args.chunk_size
    valid_batch_size = args.valid_batch_size
    fanouts = [args.fanout if args.fanout > 0 else None] * args.n_layers

    num_train_edges = train_seeds.shape[0]
    num_valid_edges = valid_neg_seeds.shape[0]
    num_test_edges = test_neg_seeds.shape[0]
    node_tids = test_g.ndata['ntype']
    num_of_ntype = int(th.max(node_tids).numpy()) + 1

    # check cuda
    use_cuda = dev_id >= 0
    if use_cuda:
        th.cuda.set_device(dev_id)
        node_tids = node_tids.to(dev_id)

    train_sampler = LinkRankSampler(train_g,
                                    train_seeds,
                                    num_train_edges,
                                    num_neg=None,
                                    fanouts=fanouts)
    dataloader = DataLoader(dataset=train_seeds,
                            batch_size=batch_size,
                            collate_fn=train_sampler.sample_blocks,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            num_workers=args.num_workers)
    valid_sampler = LinkRankSampler(valid_g,
                                    valid_neg_seeds,
                                    num_valid_edges,
                                    num_neg=args.valid_neg_cnt,
                                    fanouts=[None] * args.n_layers,
                                    is_train=False)
    valid_dataloader = DataLoader(dataset=valid_seeds,
                                  batch_size=args.valid_batch_size,
                                  collate_fn=valid_sampler.sample_blocks,
                                  shuffle=False,
                                  pin_memory=True,
                                  drop_last=False,
                                  num_workers=args.num_workers)
    test_sampler = NodeSampler(test_g, fanouts=[None] * args.n_layers)
    test_dataloader = DataLoader(dataset=th.arange(test_g.number_of_nodes()),
                                 batch_size=4 * 1024,
                                 collate_fn=test_sampler.sample_nodes,
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=False,
                                 num_workers=args.num_workers)
    
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=dev_id)

    embed_layer = RelGraphEmbedLayer(dev_id,
                                     test_g.number_of_nodes(),
                                     node_tids,
                                     num_of_ntype,
                                     [None] * num_of_ntype,
                                     args.n_hidden)
    
    model = LinkPredict(dev_id,
                        args.n_hidden,
                        num_rels,
                        edge_rels,
                        num_bases=args.n_bases,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        use_self_loop=args.use_self_loop,
                        low_mem=args.low_mem,
                        regularization_coef=args.regularization_coef)
    if dev_id >= 0:
        model.cuda(dev_id)
        if args.mix_cpu_gpu is False:
            embed_layer.cuda(dev_id)

    if n_gpus > 1:
        embed_layer = DistributedDataParallel(embed_layer, device_ids=[dev_id], output_device=dev_id)
        model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id, find_unused_parameters=True)

    # optimizer
    all_params = itertools.chain(model.parameters(), embed_layer.parameters())
    optimizer = th.optim.Adam(all_params, lr=args.lr)
    node_feats = [None] * num_of_ntype

    print("start training...")
    for epoch in range(args.n_epochs):
        model.train()
        if epoch > 1:
            t0 = time.time()
        for i, sample_data in enumerate(dataloader):
            bsize, p_g, n_g, p_blocks, n_blocks = sample_data
            p_feats = embed_layer(p_blocks[0].srcdata[dgl.NID].to(dev_id),
                                  p_blocks[0].srcdata['ntype'].to(dev_id),
                                  node_feats)
            n_feats = embed_layer(n_blocks[0].srcdata[dgl.NID].to(dev_id),
                                  n_blocks[0].srcdata['ntype'].to(dev_id),
                                  node_feats)

            p_head_emb, p_tail_emb, r_emb, n_head_emb, n_tail_emb = \
                model(p_blocks, p_feats, n_blocks, n_feats, p_g, n_g)

            n_shuffle_seed = th.randperm(n_head_emb.shape[0])
            n_head_emb = n_head_emb[n_shuffle_seed]
            n_tail_emb = n_tail_emb[n_shuffle_seed]

            loss = get_loss(p_head_emb,
                            p_tail_emb,
                            n_head_emb,
                            n_tail_emb,
                            r_emb,
                            int(batch_size / chunk_size),
                            chunk_size,
                            chunk_size,
                            args.regularization_coef)
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            t1 = time.time()

            print("Epoch {}, Iter {}, Loss:{}".format(epoch, i, loss.detach()))

        if epoch > 1:
            dur = t1 - t0
            print("Epoch {} takes {} seconds".format(epoch, dur))

        evaluate(embed_layer,
                 model,
                 valid_dataloader,
                 node_feats,
                 args.valid_batch_size,
                 args.valid_neg_cnt,
                 queue)
        if proc_id == 0 and queue is not None:
            logs = []
            for i in range(n_gpus):
                log = queue.get()
                logs = logs + log
            
            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
            print("-------------- Eval result --------------")
            for k, v in metrics.items():
                print('Eval average {} : {}'.format(k, v))
            print("-----------------------------------------")

        if n_gpus > 1:
            th.distributed.barrier()

    print()
    if args.model_path is not None:
        th.save(model.state_dict(), args.model_path)

    fullgraph_eval(test_g,
                   embed_layer,
                   model,
                   dev_id,
                   node_feats,
                   args.n_hidden,
                   test_dataloader,
                   test_seeds,
                   test_neg_seeds,
                   args.test_neg_cnt,
                   queue)
    if proc_id == 0 and queue is not None:
        logs = []
        for i in range(n_gpus):
            log = queue.get()
            logs = logs + log
            
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        print("-------------- Test result --------------")
        for k, v in metrics.items():
            print('Test average {} : {}'.format(k, v))
        print("-----------------------------------------")

def prepare_mp(g):
    """
    Explicitly materialize the CSR, CSC and COO representation of the given graph
    so that they could be shared via copy-on-write to sampler workers and GPU
    trainers.
    This is a workaround before full shared memory support on heterogeneous graphs.
    """
    for etype in g.canonical_etypes:
        g.in_degree(0, etype=etype)
        g.out_degree(0, etype=etype)
        g.find_edges([0], etype=etype)

def main(args, devices):
    if args.dataset == "drkg":
        drkg_dataset = DrkgDataset()
        drkg_dataset.load_data()

        num_nodes = drkg_dataset.num_nodes
        num_rels = drkg_dataset.num_rels
        edge_rels = num_rels * 2 # we add reverse edges
        train_data = drkg_dataset.train
        valid_data = drkg_dataset.valid
        test_data = drkg_dataset.test
        node_types = drkg_dataset.node_types

        train_g = build_multi_ntype_heterograph_in_homogeneous_from_triplets(num_nodes, num_rels, node_types, [train_data])
        valid_g = build_multi_ntype_heterograph_in_homogeneous_from_triplets(num_nodes, num_rels, node_types, [train_data, valid_data])
        test_g = build_multi_ntype_heterograph_in_homogeneous_from_triplets(num_nodes, num_rels, node_types, [train_data, valid_data, test_data])
    else:
        data = load_data(args.dataset)
        num_nodes = data.num_nodes
        num_rels = data.num_rels
        edge_rels = num_rels * 2 # we add reverse edges
        train_data = data.train
        valid_data = data.valid
        test_data = data.test

        train_g = build_heterograph_in_homogeneous_from_triplets(num_nodes, num_rels, [train_data])
        valid_g = build_heterograph_in_homogeneous_from_triplets(num_nodes, num_rels, [train_data, valid_data])
        test_g = build_heterograph_in_homogeneous_from_triplets(num_nodes, num_rels, [train_data, valid_data, test_data])

    u, v, eid = train_g.all_edges(form='all')
    train_seed_idx = (train_g.edata['set'] == 0)
    train_seeds = eid[train_seed_idx]
    train_g.edata['norm'] = th.full((eid.shape[0],1), 1)

    for rel_id in range(edge_rels):
        idx = (train_g.edata['etype'] == rel_id)
        u_r = u[idx]
        v_r = v[idx]
        _, inverse_index, count = th.unique(v_r, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = th.ones(v_r.shape[0]) / degrees
        norm = norm.unsqueeze(1)
        train_g.edata['norm'][idx] = norm
    train_g.edata['norm'].share_memory_()
    train_g.edata['etype'].share_memory_()
    train_g.ndata['ntype'].share_memory_()
    train_g.edata.pop('set')

    u, v, eid = valid_g.all_edges(form='all')
    valid_seeds = eid[valid_g.edata['set'] == 1]
    valid_neg_seeds = eid[valid_g.edata['set'] > -1]
    valid_g.edata['norm'] = th.full((eid.shape[0],1), 1)

    for rel_id in range(edge_rels):
        idx = (valid_g.edata['etype'] == rel_id)
        u_r = u[idx]
        v_r = v[idx]
        _, inverse_index, count = th.unique(v_r, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = th.ones(v_r.shape[0]) / degrees
        norm = norm.unsqueeze(1)
        valid_g.edata['norm'][idx] = norm
    valid_g.edata['norm'].share_memory_()
    valid_g.edata['etype'].share_memory_()
    valid_g.ndata['ntype'].share_memory_()
    valid_g.edata.pop('set')

    u, v, eid = test_g.all_edges(form='all')
    test_seeds = eid[test_g.edata['set'] == 2]
    test_neg_seeds = eid[test_g.edata['set'] > -1]
    test_g.edata['norm'] = th.full((eid.shape[0],1), 1)

    for rel_id in range(edge_rels):
        idx = (test_g.edata['etype'] == rel_id)
        u_r = u[idx]
        v_r = v[idx]
        _, inverse_index, count = th.unique(v_r, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = th.ones(v_r.shape[0]) / degrees
        norm = norm.unsqueeze(1)
        test_g.edata['norm'][idx] = norm
    test_g.edata['norm'].share_memory_()
    test_g.edata['etype'].share_memory_()
    test_g.ndata['ntype'].share_memory_()
    test_g.edata.pop('set')

    n_gpus = len(devices)
    # cpu
    if devices[0] == -1:
        run(0, 0, args, ['cpu'], (train_g, valid_g, test_g, num_rels, edge_rels),
            (train_seeds, valid_seeds, test_seeds),
            (train_seeds, valid_neg_seeds, test_neg_seeds))
    # gpu
    elif n_gpus == 1:
        run(0, n_gpus, args, devices, (train_g, valid_g, test_g, num_rels, edge_rels),
            (train_seeds, valid_seeds, test_seeds),
            (train_seeds, valid_neg_seeds, test_neg_seeds))
    # multi gpu
    else:
        prepare_mp(train_g)
        prepare_mp(valid_g)
        prepare_mp(test_g)
        queue = mp.Queue(n_gpus)
        procs = []
        num_train_seeds = train_seeds.shape[0]
        num_valid_seeds = valid_seeds.shape[0]
        num_test_seeds = test_seeds.shape[0]
        tseeds_per_proc = num_train_seeds // n_gpus
        vseeds_per_proc = num_valid_seeds // n_gpus
        tstseeds_per_proc = num_test_seeds // n_gpus
        for proc_id in range(n_gpus):
            proc_train_seeds = train_seeds[proc_id * tseeds_per_proc :
                                           (proc_id + 1) * tseeds_per_proc \
                                           if (proc_id + 1) * tseeds_per_proc < num_train_seeds \
                                           else num_train_seeds]
            proc_valid_seeds = valid_seeds[proc_id * vseeds_per_proc :
                                           (proc_id + 1) * vseeds_per_proc \
                                           if (proc_id + 1) * vseeds_per_proc < num_train_seeds \
                                           else num_valid_seeds]
            proc_test_seeds = test_seeds[proc_id * tstseeds_per_proc :
                                         (proc_id + 1) * tstseeds_per_proc \
                                         if (proc_id + 1) * tstseeds_per_proc < num_train_seeds \
                                         else num_test_seeds]
            p = mp.Process(target=run, args=(proc_id, n_gpus, args, devices,
                           (train_g, valid_g, test_g, num_rels, edge_rels),
                           (proc_train_seeds, proc_valid_seeds, proc_test_seeds),
                           (train_seeds, valid_neg_seeds, test_neg_seeds),
                           queue))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

def config():
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.2,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=200,
            help="number of hidden units")
    parser.add_argument("--gpu", type=str, default='0',
            help="gpu")
    parser.add_argument("--keep_pos_edges", default=False, action='store_true',
            help="Whether delete positive edges during training in case of linkage leakage")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=10,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=20,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--model_path", type=str, default=None,
            help='path for save the model')
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    parser.add_argument("--batch-size", type=int, default=1024,
            help="Mini-batch size.")
    parser.add_argument("--chunk-size", type=int, default=32,
            help="Negative sample chunk size. In each chunk, positive pairs will share negative edges")
    parser.add_argument("--valid-batch-size", type=int, default=8,
            help="Mini-batch size for validation and test.")
    parser.add_argument("--fanout", type=int, default=10,
            help="Fan-out of neighbor sampling.")
    parser.add_argument("--regularization-coef", type=float, default=0.001,
            help="Regularization Coeffiency.")
    parser.add_argument("--grad-norm", type=float, default=1.0,
            help="norm to clip gradient to")
    parser.add_argument("--valid-neg-cnt", type=int, default=1000,
            help="Validation negative sample cnt.")
    parser.add_argument("--test-neg-cnt", type=int, default=-1,
            help="Test negative sample cnt.")
    parser.add_argument("--sample-based-eval", default=False, action='store_true',
            help="Use sample based evalution method or full-graph based evalution method")
    parser.add_argument("--num-workers", type=int, default=0,
            help="Number of workers for dataloader.")
    parser.add_argument("--low-mem", default=False, action='store_true',
            help="Whether use low mem RelGraphCov")
    parser.add_argument("--mix-cpu-gpu", default=False, action='store_true',
            help="Whether store node embeddins in cpu")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = config()
    devices = list(map(int, args.gpu.split(',')))
    print(args)
    assert args.batch_size > 0
    assert args.valid_batch_size > 0
    main(args, devices)