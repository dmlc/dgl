"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn

Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""

import argparse
import numpy as np
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dataset import load_entity
from functools import partial

from dgl.contrib.data import load_data
from model import RelGraphEmbedLayer, RelGraphConvLayer
from utils import build_heterograph_from_triplets

class LinkPredict(nn.Module):
    def __init__(self,
                 device,
                 h_dim,
                 num_train_rels,
                 num_rels,
                 num_bases=-1,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False,
                 regularization_coef=0):
        super(LinkPredict, self).__init__()
        self.device = device
        self.h_dim = h_dim
        self.num_rels = num_rels
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
            self.h_dim, self.h_dim, self.num_rels, "basis",
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            dropout=self.dropout))
        # h2h
        for idx in range(self.num_hidden_layers):
            self.layers.append(RelGraphConvLayer(
                self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout))

    def forward(self, p_blocks, n_blocks, p_feats, n_feats, p_g, n_g):
        p_h = p_feats
        n_h = n_feats
        for layer, block in zip(self.layers, p_blocks):
            block = block.to(self.device)
            p_h = layer(block, p_h)

        for layer, block in zip(self.layers, n_blocks):
            block = block.to(self.device)
            n_h = layer(block, n_h)

        head, tail, eid = p_g.all_edges(form='all')
        p_head_emb = p_h[head]
        p_tail_emb = p_h[tail]
        rids = p_g.edata[dgl.ETYPE]
        head, tail = n_g.all_edges()
        n_head_emb = n_h[head]
        n_tail_emb = n_h[tail]
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

    def calc_neg_tail_score(self, heads, tails, rids, num_chunks, chunk_size, neg_sample_size, device=None):
        hidden_dim = heads.shape[1]
        r = self.w_relation[rids]

        if device is not None:
            r = r.to(device)
            heads = heads.to(device)
            tails = tails.to(device)
        tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
        tails = th.transpose(tails, 1, 2)
        tmp = (heads * r).reshape(num_chunks, chunk_size, hidden_dim)
        return th.bmm(tmp, tails)

    def calc_neg_head_score(self, heads, tails, rids, num_chunks, chunk_size, neg_sample_size, device=None):
        hidden_dim = tails.shape[1]
        r = self.w_relation[rids]
        if device is not None:
            r = r.to(device)
            heads = heads.to(device)
            tails = tails.to(device)
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

class LinkRankSampler:
    def __init__(self, g, num_edges, num_neg, fanouts, 
        is_train=True, keep_pos_edges=False):
        self.g = g
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
        else:
            nseeds = th.randint(self.num_edges, (bsize,))

        g = self.g
        fanouts = self.fanouts
        assert len(g.canonical_etypes) == 1
        p_subg = g.edge_subgraph({g.canonical_etypes[0] : pseeds})
        n_subg = g.edge_subgraph({g.canonical_etypes[0] : nseeds})

        p_g = dgl.compact_graphs(p_subg)
        n_g = dgl.compact_graphs(n_subg)
        p_g.edata[dgl.ETYPE] = g.edata[dgl.ETYPE][p_subg.edata[dgl.EID]]

        assert len(g.ntypes) == 1
        pg_seed = p_g.ndata[dgl.NID]
        ng_seed = n_g.ndata[dgl.NID]

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

            if self.keep_pos_edges is False and \
                self.is_train and i == 0:
                old_frontier = p_frontier
                p_frontier = dgl.remove_edges(old_frontier, pseeds)

            p_etypes = g.edata[dgl.ETYPE][p_frontier.edata[dgl.EID]]
            # print(p_etypes)
            n_etypes = g.edata[dgl.ETYPE][n_frontier.edata[dgl.EID]]
            p_norm = g.edata['norm'][p_frontier.edata[dgl.EID]]
            n_norm = g.edata['norm'][n_frontier.edata[dgl.EID]]
            p_block = dgl.to_block(p_frontier, p_curr)
            n_block = dgl.to_block(n_frontier, n_curr)
            p_block.srcdata[dgl.NTYPE] = g.ndata[dgl.NTYPE][p_block.srcdata[dgl.NID]]
            n_block.srcdata[dgl.NTYPE] = g.ndata[dgl.NTYPE][n_block.srcdata[dgl.NID]]
            p_block.edata['etype'] = p_etypes
            n_block.edata['etype'] = n_etypes
            p_block.edata['norm'] = p_norm
            n_block.edata['norm'] = n_norm
            p_curr = p_block.srcdata[dgl.NID]
            n_curr = n_block.srcdata[dgl.NID]
            p_blocks.insert(0, p_block)
            n_blocks.insert(0, n_block)

        return (bsize, p_g, n_g, p_blocks, n_blocks)

def evaluate(embed_layer, model, dataloader, node_feats, bsize, neg_cnt):
    logs = []
    model.eval()
    t0 = time.time()

    for i, sample_data in enumerate(dataloader):
        bsize, p_g, n_g, p_blocks, n_blocks = sample_data
        p_feats = embed_layer(p_blocks[0].srcdata[dgl.NID],
                              p_blocks[0].srcdata[dgl.NTYPE],
                              node_feats)
        n_feats = embed_layer(n_blocks[0].srcdata[dgl.NID],
                              n_blocks[0].srcdata[dgl.NTYPE],
                              node_feats)
        p_head_emb, p_tail_emb, rids, n_head_emb, n_tail_emb = \
            model(p_blocks, n_blocks, p_feats, n_feats, p_g, n_g)

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
        h_neg_score = F.logsigmoid(h_neg_score).reshape(bsize, neg_cnt)
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
    if args.dataset == "drug":
        return
    else:
        data = load_data(args.dataset)
        num_nodes = data.num_nodes
        num_rels = data.num_rels
        train_data = data.train
        valid_data = data.valid
        test_data = data.test

        train_hg = build_heterograph_from_triplets(num_nodes, num_rels, [train_data])
        valid_hg = build_heterograph_from_triplets(num_nodes, num_rels, [train_data, valid_data])
        test_hg = build_heterograph_from_triplets(num_nodes, num_rels, [train_data, valid_data, test_data])

        train_data = train_data.transpose()
        valid_data = valid_data.transpose()
        test_data = test_data.transpose()

    device = th.device(args.gpu) if args.gpu >= 0 else th.device('cpu')
    batch_size = args.batch_size
    chunk_size = args.chunk_size
    valid_batch_size = args.valid_batch_size
    fanouts = [args.fanout if args.fanout > 0 else None] * args.n_layers

    # As we may add reserve edges, these edges will not be the seeds
    if len(train_data) == 3:

    # calculate norm for each edge type and store in edge
    for canonical_etypes in train_hg.canonical_etypes:
        
        u, v, eid = train_hg.all_edges(form='all', etype=canonical_etypes)
        _, inverse_index, count = th.unique(v, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = th.ones(eid.shape[0]) / degrees
        norm = norm.unsqueeze(1)
        train_hg.edges[canonical_etypes].data['norm'] = norm

    for canonical_etypes in valid_hg.canonical_etypes:
        u, v, eid = valid_hg.all_edges(form='all', etype=canonical_etypes)
        _, inverse_index, count = th.unique(v, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = th.ones(eid.shape[0]) / degrees
        norm = norm.unsqueeze(1)
        valid_hg.edges[canonical_etypes].data['norm'] = norm

    for canonical_etypes in test_hg.canonical_etypes:
        u, v, eid = test_hg.all_edges(form='all', etype=canonical_etypes)
        _, inverse_index, count = th.unique(v, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = th.ones(eid.shape[0]) / degrees
        norm = norm.unsqueeze(1)
        test_hg.edges[canonical_etypes].data['norm'] = norm
    assert len(train_hg.ntypes) == len(test_hg.ntypes)
    num_of_ntype = len(test_hg.ntypes)

    train_g = dgl.to_homo(train_hg)
    valid_g = dgl.to_homo(valid_hg)
    test_g = dgl.to_homo(test_hg)
    train_u, train_v, train_e = train_g.all_edges(form='all')
    valid_u, valid_v, valid_e = valid_g.all_edges(form='all')
    test_u, test_v, test_e = test_g.all_edges(form='all')
    num_train_edges = train_e.shape[0]
    num_valid_edges = valid_e.shape[0]
    num_test_edges = test_e.shape[0]
    node_tids = test_g.ndata[dgl.NTYPE]

    # check cuda
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)
        node_tids = node_tids.cuda()
        train_g.edata['norm'] = train_g.edata['norm'].cuda()
        valid_g.edata['norm'] = valid_g.edata['norm'].cuda()
        test_g.edata['norm'] = test_g.edata['norm'].cuda()

    train_seeds = th.arange(train_e.shape[0])
    valid_seeds = th.arange(valid_e.shape[0])
    test_seeds =  th.arange(test_e.shape[0])

    train_sampler = LinkRankSampler(train_g,
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
    embed_layer = RelGraphEmbedLayer(test_g.number_of_nodes(),
                                     node_tids,
                                     num_of_ntype,
                                     [None] * num_of_ntype,
                                     args.n_hidden)
    
    model = LinkPredict(device,
                        args.n_hidden,
                        num_rels * 2,
                        num_bases=args.n_bases,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        use_self_loop=args.use_self_loop,
                        regularization_coef=args.regularization_coef)
    if args.mix_cpu_gpu is False and args.gpu >= 0:
        embed_layer.cuda()
        model.cuda(device)

    # optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
    node_feats = [None] * num_of_ntype

    print("start training...")
    for epoch in range(args.n_epochs):
        model.train()
        if epoch > 1:
            t0 = time.time()
        for i, sample_data in enumerate(dataloader):
            bsize, p_g, n_g, p_blocks, n_blocks = sample_data
            p_feats = embed_layer(p_blocks[0].srcdata[dgl.NID].to(device),
                                  p_blocks[0].srcdata[dgl.NTYPE].to(device),
                                  node_feats)
            n_feats = embed_layer(n_blocks[0].srcdata[dgl.NID].to(device),
                                  n_blocks[0].srcdata[dgl.NTYPE].to(device),
                                  node_feats)

            p_head_emb, p_tail_emb, rids, n_head_emb, n_tail_emb = \
                model(p_blocks, n_blocks, p_feats, n_feats, p_g, n_g)

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

        evaluate(embed_layer, model, valid_dataloader, node_feats, args.valid_batch_size, args.valid_neg_cnt)

    print()
    if args.model_path is not None:
        th.save(model.state_dict(), args.model_path)

    evaluate(embed_layer, model, test_dataloader, node_feats, args.valid_batch_size, args.test_neg_cnt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.2,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=200,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--keep_pos_edges", default=False, action='store_true',
            help="Whether delete positive edges during training in case of linkage leakage")
    parser.add_argument("--mix_cpu_gpu", default=False, action='store_true',
            help="Mix CPU and GPU training")
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
    parser.add_argument("--test-neg-cnt", type=int, default=1000,
            help="Test negative sample cnt.")
    parser.add_argument("--sample-based-eval", default=False, action='store_true',
            help="Use sample based evalution method or full-graph based evalution method")
    parser.add_argument("--num-workers", type=int, default=0,
            help="Number of workers for dataloader.")

    args = parser.parse_args()
    print(args)
    assert args.batch_size > 0
    assert args.valid_batch_size > 0
    main(args)