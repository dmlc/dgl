"""Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Reference Code: https://github.com/tkipf/relational-gcn
"""
import argparse
import itertools
import numpy as np
import time, gc
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial
import line_profiler

import dgl
import dgl.function as fn
from dgl.contrib.data import load_data
from entity_classify_mb import RelGraphConvHetero, RelGraphEmbed
from utils import build_graph_from_triplets, thread_wrapped_func

class LinkPredict(nn.Module):
    def __init__(self,
                 g,
                 device,
                 h_dim,
                 num_rels,
                 num_bases=-1,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False,
                 regularization_coef=0):
        super(LinkPredict, self).__init__()
        self.g = g
        self.device = device
        self.h_dim = h_dim

        self.rel_names = list(set(g.etypes))
        self.rel_names.sort()
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.regularization_coef = regularization_coef

        self.embed_layer = RelGraphEmbed(self.g, h_dim)
        self.w_relation = nn.Parameter(th.Tensor(num_rels, h_dim).to(self.device))
        nn.init.xavier_uniform_(self.w_relation)

        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConvHetero(
            self.h_dim, self.h_dim, self.rel_names, "basis",
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            dropout=self.dropout, use_weight=False))
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(RelGraphConvHetero(
                self.h_dim, self.h_dim, self.rel_names, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout))
        self.layers.to(self.device)

    def forward(self, p_blocks, n_blocks, p_g, n_g):
        emb = self.embed_layer()
        p_h = {}
        n_h = {}
        for k, e in emb.items():
            p_h[k] = e[p_blocks[0].srcnodes[k].data[dgl.NID]].to(self.device)
            n_h[k] = e[n_blocks[0].srcnodes[k].data[dgl.NID]].to(self.device)

        for layer, block in zip(self.layers, p_blocks):
            p_h = layer(block, p_h)
        for layer, block in zip(self.layers, n_blocks):
            n_h = layer(block, n_h)

        for ntype, emb in p_h.items():
            p_g.nodes[ntype].data['h'] = emb
        for ntype, emb in n_h.items():
            n_g.nodes[ntype].data['h'] = emb
        
        p_head_emb = []
        p_tail_emb = []
        rids = []
        for canonical_etype in p_g.canonical_etypes:
            head, tail = p_g.all_edges(etype=canonical_etype)
            head_emb = p_g.nodes[canonical_etype[0]].data['h'][head]
            tail_emb = p_g.nodes[canonical_etype[2]].data['h'][tail]
            idx = int(canonical_etype[1])
            rids.append(th.full((head_emb.shape[0],), idx, dtype=th.long))
            p_head_emb.append(head_emb)
            p_tail_emb.append(tail_emb)
        n_head_emb = []
        n_tail_emb = []
        for canonical_etype in n_g.canonical_etypes:
            head, tail = n_g.all_edges(etype=canonical_etype)
            head_emb = n_g.nodes[canonical_etype[0]].data['h'][head]
            tail_emb = n_g.nodes[canonical_etype[2]].data['h'][tail]
            n_head_emb.append(head_emb)
            n_tail_emb.append(tail_emb)
        p_head_emb = th.cat(p_head_emb, dim=0)
        p_tail_emb = th.cat(p_tail_emb, dim=0)
        rids = th.cat(rids, dim=0)
        n_head_emb = th.cat(n_head_emb, dim=0)
        n_tail_emb = th.cat(n_tail_emb, dim=0)

        return p_head_emb, p_tail_emb, rids, n_head_emb, n_tail_emb

    def regularization_loss(self, h_emb, t_emb, nh_emb, nt_emb):
        return th.mean(h_emb.pow(2)) + \
               th.mean(t_emb.pow(2)) + \
               th.mean(nh_emb.pow(2)) + \
               th.mean(nt_emb.pow(2)) + \
               th.mean(self.w_relation.pow(2))

    def calc_pos_score(self, h_emb, t_emb, rids):
        # DistMult
        r = self.w_relation[rids]
        score = th.sum(h_emb * r * t_emb, dim=-1)
        return score

    def calc_neg_tail_score(self, heads, tails, rids, num_chunks, chunk_size, neg_sample_size):
        hidden_dim = heads.shape[1]
        r = self.w_relation[rids]
        tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
        tails = th.transpose(tails, 1, 2)
        tmp = (heads * r).reshape(num_chunks, chunk_size, hidden_dim)
        return th.bmm(tmp, tails)

    def calc_neg_head_score(self, heads, tails, rids, num_chunks, chunk_size, neg_sample_size):
        hidden_dim = tails.shape[1]
        r = self.w_relation[rids]
        heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
        heads = th.transpose(heads, 1, 2)
        tmp = (tails * r).reshape(num_chunks, chunk_size, hidden_dim)
        return th.bmm(tmp, heads)
    
    def get_loss(self, h_emb, t_emb, nh_emb, nt_emb, rids, num_chunks, chunk_size, neg_sample_size):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        pos_score = self.calc_pos_score(h_emb, t_emb, rids)
        t_neg_score = self.calc_neg_tail_score(h_emb, nt_emb, rids, num_chunks, chunk_size, neg_sample_size)
        h_neg_score = self.calc_neg_head_score(nh_emb, t_emb, rids, num_chunks, chunk_size, neg_sample_size)
        pos_score = F.logsigmoid(pos_score)
        h_neg_score = h_neg_score.reshape(-1, neg_sample_size)
        t_neg_score = t_neg_score.reshape(-1, neg_sample_size)
        h_neg_score = F.logsigmoid(-h_neg_score).mean(dim=1)
        t_neg_score = F.logsigmoid(-t_neg_score).mean(dim=1)

        pos_score = pos_score.mean()
        h_neg_score = h_neg_score.mean()
        t_neg_score = t_neg_score.mean()
        predict_loss = -(2 * pos_score + h_neg_score + t_neg_score)

        reg_loss = self.regularization_loss(h_emb, t_emb, nh_emb, nt_emb)

        print("pos loss {}, neg loss {}|{}, reg_loss {}".format(pos_score.detach(),
                                                                h_neg_score.detach(),
                                                                t_neg_score.detach(),
                                                                self.regularization_coef * reg_loss.detach()))
        return predict_loss + self.regularization_coef * reg_loss

class RGCNLinkRankSampler:
    def __init__(self, g, num_edges, etypes, netypes, num_rels, phead_ids, ptail_ids, fanouts, 
        nhead_ids, ntail_ids, num_neg=None, is_train=True):
        self.g = g
        self.num_edges = num_edges
        self.etypes = etypes
        self.netypes = netypes
        self.num_rels = num_rels
        self.phead_ids = phead_ids
        self.ptail_ids = ptail_ids
        self.nhead_ids = nhead_ids
        self.ntail_ids = ntail_ids
        self.fanouts = fanouts
        self.num_neg = num_neg
        self.is_train = is_train

    def sample_blocks(self, seeds):
        pseed = th.stack(seeds)
        bsize = pseed.shape[0]
        if self.num_neg is not None:
            nseed = th.randint(self.num_edges, (self.num_neg,))
        else:
            nseed = th.randint(self.num_edges, (bsize,))
        g = self.g
        etypes = self.etypes
        netypes = self.netypes
        fanouts = self.fanouts
        phead_ids = self.phead_ids
        ptail_ids = self.ptail_ids
        nhead_ids = self.nhead_ids
        ntail_ids = self.ntail_ids
        
        p_etypes = etypes[pseed]
        phead_ids = phead_ids[pseed]
        ptail_ids = ptail_ids[pseed]
        n_etypes = netypes[nseed]
        nhead_ids = nhead_ids[nseed]
        ntail_ids = ntail_ids[nseed]

        p_edges = {}
        p_subg = []
        for key in range(self.num_rels):
            pe_loc = (p_etypes == key)
            p_head = phead_ids[pe_loc]
            p_tail = ptail_ids[pe_loc]
            if p_head.shape[0] == 0:
                continue
            canonical_etypes = ('node', str(key), 'node')
            p_edges[canonical_etypes] = (p_head, p_tail)
            p_subg.append(dgl.graph((p_head, p_tail),
                                     canonical_etypes[0],
                                     canonical_etypes[1],
                                     g.number_of_nodes(canonical_etypes[0])))
        n_subg = []
        for key in range(self.num_rels):
            ne_loc = (n_etypes == key)
            n_head = nhead_ids[ne_loc]
            n_tail = ntail_ids[ne_loc]
            if n_head.shape[0] == 0:
                continue

            canonical_etypes = ('node', str(key), 'node')
            n_subg.append(dgl.graph((n_head, n_tail),
                                     canonical_etypes[0],
                                     canonical_etypes[1],
                                     g.number_of_nodes(canonical_etypes[0])))

        p_g = dgl.hetero_from_relations(p_subg)
        n_g = dgl.hetero_from_relations(n_subg)
        p_g = dgl.compact_graphs(p_g)
        n_g = dgl.compact_graphs(n_g)

        pg_seed = {}
        ng_seed = {}
        for ntype in p_g.ntypes:
            pg_seed[ntype] = p_g.nodes[ntype].data[dgl.NID]
        for ntype in n_g.ntypes:
            ng_seed[ntype] = n_g.nodes[ntype].data[dgl.NID]

        p_blocks = []
        n_blocks = []
        p_cur = pg_seed
        n_cur = ng_seed
        for i, fanout in enumerate(fanouts):
            if fanout is None:
                p_frontier = dgl.in_subgraph(g, p_cur)
                n_frontier = dgl.in_subgraph(g, n_cur)
            else:
                p_frontier = dgl.sampling.sample_neighbors(g, p_cur, fanout)
                n_frontier = dgl.sampling.sample_neighbors(g, n_cur, fanout)
            if self.is_train and i == 0 and len(p_edges) > 0:
                # remove edges here
                edge_to_del = {}
                for canonical_etype, pairs in p_edges.items():
                    eid_to_del = p_frontier.edge_ids(pairs[0],
                                                     pairs[1],
                                                     return_uv=True,
                                                     etype=canonical_etype)[2]
                    if eid_to_del.shape[0] > 0:
                        edge_to_del[canonical_etype] = eid_to_del
                old_frontier = p_frontier
                p_frontier = dgl.remove_edges(old_frontier, edge_to_del)
            
            p_block = dgl.to_block(p_frontier, p_cur)
            p_cur = {}
            for ntype in p_block.srctypes:
                p_cur[ntype] = p_block.srcnodes[ntype].data[dgl.NID]
            p_blocks.insert(0, p_block)

            n_block = dgl.to_block(n_frontier, n_cur)
            n_cur = {}
            for ntype in n_block.srctypes:
                n_cur[ntype] = n_block.srcnodes[ntype].data[dgl.NID]
            n_blocks.insert(0, n_block)
        
        return (bsize, p_g, n_g, p_blocks, n_blocks)

def evaluate(model, dataloader, bsize, neg_cnt):
    logs = []
    model.eval()
    t0 = time.time()
    for i, sample_data in enumerate(dataloader):
        bsize, p_g, n_g, p_blocks, n_blocks = sample_data

        p_head_emb, p_tail_emb, rids, n_head_emb, n_tail_emb = \
            model(p_blocks, n_blocks, p_g, n_g)

        pos_score = model.calc_pos_score(p_head_emb, p_tail_emb, rids)
        t_neg_score = model.calc_neg_tail_score(p_head_emb,
                                                n_tail_emb,
                                                rids,
                                                1,
                                                bsize,
                                                neg_cnt)
        h_neg_score = model.calc_neg_head_score(n_head_emb,
                                                p_tail_emb,
                                                rids,
                                                1,
                                                bsize,
                                                neg_cnt)
        pos_scores = F.logsigmoid(pos_score).reshape(bsize, -1)
        t_neg_score = F.logsigmoid(t_neg_score).reshape(bsize, neg_cnt)
        h_neg_score = F.logsigmoid(t_neg_score).reshape(bsize, neg_cnt)
        neg_scores = th.cat([h_neg_score, t_neg_score], dim=1)
        rankings = th.sum(neg_scores >= pos_scores, dim=1) + 1
        rankings = rankings.cpu().detach().numpy()
        for idx in range(bsize):
            ranking = rankings[idx]
            logs.append({
                'MRR': 1.0 / ranking,
                'MR': float(ranking),
                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                'HITS@10': 1.0 if ranking <= 10 else 0.0
            })
        if i % 100 == 0:
            t1 = time.time()
            print("Eval {} samples takes {} seconds".format(i * bsize, t1 - t0))

    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

    for k, v in metrics.items():
        print('Test average {}: {}'.format(k, v))

def main(args):
    data = load_data(args.dataset)
    num_nodes = data.num_nodes
    print(num_nodes)
    train_data = data.train
    valid_data = data.valid
    test_data = data.test
    num_rels = data.num_rels

    train_g = build_graph_from_triplets(num_nodes, num_rels, [train_data])
    valid_g = build_graph_from_triplets(num_nodes, num_rels, [train_data, valid_data])
    test_g = build_graph_from_triplets(num_nodes, num_rels, [train_data, valid_data, test_data])

    batch_size = args.batch_size
    chunk_size = args.chunk_size
    valid_batch_size = args.valid_batch_size
    fanouts = [args.fanout] * args.n_layers

    train_src, train_rel, train_dst = train_data.transpose()
    head_ids = th.from_numpy(train_src)
    tail_ids = th.from_numpy(train_dst)
    etypes = th.from_numpy(train_rel)
    num_train_edges = train_data.shape[0]
    pos_seed = th.arange((num_train_edges//batch_size) * batch_size)

    # train dataloader
    sampler = RGCNLinkRankSampler(train_g,
                                  num_train_edges,
                                  etypes,
                                  etypes,
                                  num_rels,
                                  head_ids,
                                  tail_ids,
                                  fanouts,
                                  nhead_ids=head_ids,
                                  ntail_ids=tail_ids)

    dataloader = DataLoader(dataset=pos_seed,
                            batch_size=batch_size,
                            collate_fn=sampler.sample_blocks,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            num_workers=args.num_workers)

    # eval dataloader
    valid_src, valid_rel, valid_dst = valid_data.transpose()
    valid_head_ids = th.from_numpy(valid_src)
    valid_tail_ids = th.from_numpy(valid_dst)
    valid_etypes = th.from_numpy(valid_rel)
    valid_neg_head_ids = th.cat([head_ids, valid_head_ids])
    valid_neg_tail_ids = th.cat([tail_ids, valid_tail_ids])
    valid_neg_etypes = th.cat([etypes, valid_etypes])
    num_valid_edges = valid_data.shape[0] + num_train_edges
    valid_seed = th.arange(valid_etypes.shape[0])

    valid_sampler = RGCNLinkRankSampler(valid_g,
                                        num_valid_edges,
                                        valid_etypes,
                                        valid_neg_etypes,
                                        num_rels,
                                        valid_head_ids,
                                        valid_tail_ids,
                                        [None] * args.n_layers,
                                        nhead_ids=valid_neg_head_ids,
                                        ntail_ids=valid_neg_tail_ids,
                                        num_neg=args.valid_neg_cnt,
                                        is_train=False)
    valid_dataloader = DataLoader(dataset=valid_seed,
                                  batch_size=valid_batch_size,
                                  collate_fn=valid_sampler.sample_blocks,
                                  shuffle=False,
                                  pin_memory=True,
                                  drop_last=False,
                                  num_workers=args.num_workers)

    # test dataloader
    test_src, test_rel, test_dst = test_data.transpose()
    test_head_ids = th.from_numpy(test_src)
    test_tail_ids = th.from_numpy(test_dst)
    test_etypes = th.from_numpy(test_rel)
    test_neg_head_ids = th.cat([valid_neg_head_ids, test_head_ids])
    test_neg_tail_ids = th.cat([valid_neg_tail_ids, test_tail_ids])
    test_neg_etypes = th.cat([valid_neg_etypes, test_etypes])
    num_test_edges = test_data.shape[0] + num_valid_edges
    test_seed = th.arange(test_etypes.shape[0])

    test_sampler = RGCNLinkRankSampler(valid_g,
                                        num_test_edges,
                                        test_etypes,
                                        test_neg_etypes,
                                        num_rels,
                                        test_head_ids,
                                        test_tail_ids,
                                        [None] * args.n_layers,
                                        nhead_ids=test_neg_head_ids,
                                        ntail_ids=test_neg_tail_ids,
                                        num_neg=args.test_neg_cnt,
                                        is_train=False)
    test_dataloader = DataLoader(dataset=test_seed,
                                  batch_size=valid_batch_size,
                                  collate_fn=test_sampler.sample_blocks,
                                  shuffle=False,
                                  pin_memory=True,
                                  drop_last=False,
                                  num_workers=args.num_workers)

    # build input layer
    model = LinkPredict(test_g,
                        args.gpu if args.gpu >= 0 else 'cpu',
                        args.n_hidden,
                        num_rels,
                        num_bases=args.n_bases,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        use_self_loop=args.use_self_loop,
                        regularization_coef=args.regularization_coef)
    # optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)

    print("start training...")
    for epoch in range(args.n_epochs):
        model.train()
        if epoch > 1:
            t0 = time.time()
        for i, sample_data in enumerate(dataloader):
            bsize, p_g, n_g, p_blocks, n_blocks = sample_data

            p_head_emb, p_tail_emb, rids, n_head_emb, n_tail_emb = \
                model(p_blocks, n_blocks, p_g, n_g)
                
            n_shuffle_seed = th.randperm(n_head_emb.shape[0])
            n_head_emb = n_head_emb[n_shuffle_seed]
            n_tail_emb = n_tail_emb[n_shuffle_seed]

            loss = model.get_loss(p_head_emb,
                                  p_tail_emb,
                                  n_head_emb,
                                  n_tail_emb,
                                  rids,
                                  int(batch_size / chunk_size),
                                  chunk_size,
                                  chunk_size)
            loss.backward()
            optimizer.step()
            th.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            t1 = time.time()

            print("Epoch {}, Iter {}, Loss:{}".format(epoch, i, loss.detach()))

        if epoch > 1:
            dur = t1 - t0
            print("Epoch {} takes {} seconds".format(epoch, dur))
        gc.collect()
        evaluate(model, valid_dataloader, args.valid_batch_size, args.valid_neg_cnt)
    print()
    if args.model_path is not None:
        th.save(model.state_dict(), args.model_path)
    evaluate(model, test_dataloader, args.valid_batch_size, args.test_neg_cnt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.2,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=500,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=100,
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
    parser.add_argument("--valid-batch-size", type=int, default=16,
            help="Mini-batch size for validation and test.")
    parser.add_argument("--fanout", type=int, default=4,
            help="Fan-out of neighbor sampling.")
    parser.add_argument("--regularization-coef", type=float, default=0.001,
            help="Regularization Coeffiency.")
    parser.add_argument("--grad-norm", type=float, default=1.0,
            help="norm to clip gradient to")
    parser.add_argument("--valid-neg-cnt", type=int, default=1000,
            help="Validation negative sample cnt.")
    parser.add_argument("--test-neg-cnt", type=int, default=1000,
            help="Test negative sample cnt.")
    parser.add_argument("--num-workers", type=int, default=0,
            help="Number of workers for dataloader.")

    args = parser.parse_args()
    print(args)
    assert args.batch_size > 0
    assert args.valid_batch_size > 0
    main(args)