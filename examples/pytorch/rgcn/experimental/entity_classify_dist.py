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
import time
import os
os.environ['DGLBACKEND']='pytorch'

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import dgl
from dgl import DGLGraph
from dgl.distributed import DistDataLoader
from functools import partial

from dgl.nn import RelGraphConv
import tqdm

from ogb.nodeproppred import DglNodePropPredDataset
from pyinstrument import Profiler

class EntityClassify(nn.Module):
    """ Entity classification class for RGCN
    Parameters
    ----------
    device : int
        Device to run the layer.
    num_nodes : int
        Number of nodes.
    h_dim : int
        Hidden dim size.
    out_dim : int
        Output dim size.
    num_rels : int
        Numer of relation types.
    num_bases : int
        Number of bases. If is none, use number of relations.
    num_hidden_layers : int
        Number of hidden RelGraphConv Layer
    dropout : float
        Dropout
    use_self_loop : bool
        Use self loop if True, default False.
    low_mem : bool
        True to use low memory implementation of relation message passing function
        trade speed with memory consumption
    """
    def __init__(self,
                 device,
                 h_dim,
                 out_dim,
                 num_rels,
                 num_bases=None,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False,
                 low_mem=False,
                 layer_norm=False):
        super(EntityClassify, self).__init__()
        self.device = device
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.low_mem = low_mem
        self.layer_norm = layer_norm

        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConv(
            self.h_dim, self.h_dim, self.num_rels, "basis",
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            low_mem=self.low_mem, dropout=self.dropout))
        # h2h
        for idx in range(self.num_hidden_layers):
            self.layers.append(RelGraphConv(
                self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                low_mem=self.low_mem, dropout=self.dropout))
        # h2o
        self.layers.append(RelGraphConv(
            self.h_dim, self.out_dim, self.num_rels, "basis",
            self.num_bases, activation=None,
            self_loop=self.use_self_loop,
            low_mem=self.low_mem))

    def forward(self, blocks, feats, norm=None):
        if blocks is None:
            # full graph training
            blocks = [self.g] * len(self.layers)
        h = feats
        for layer, block in zip(self.layers, blocks):
            block = block.to(self.device)
            h = layer(block, h, block.edata['etype'], block.edata['norm'])
        return h

def init_emb(shape, dtype):
    arr = th.zeros(shape, dtype=dtype)
    nn.init.uniform_(arr, -1.0, 1.0)
    return arr

class DistEmbedLayer(nn.Module):
    r"""Embedding layer for featureless heterograph.
    Parameters
    ----------
    dev_id : int
        Device to run the layer.
    g : DistGraph
        training graph
    num_of_ntype : int
        Number of node types
    embed_size : int
        Output embed size
    sparse_emb: bool
        Whether to use sparse embedding
        Default: False
    dgl_sparse_emb: bool
        Whether to use DGL sparse embedding
        Default: False
    embed_name : str, optional
        Embed name
    """
    def __init__(self,
                 dev_id,
                 g,
                 num_of_ntype,
                 embed_size,
                 sparse_emb=False,
                 dgl_sparse_emb=False,
                 embed_name='node_emb'):
        super(DistEmbedLayer, self).__init__()
        self.dev_id = dev_id
        self.num_of_ntype = num_of_ntype
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.sparse_emb = sparse_emb

        if sparse_emb:
            if dgl_sparse_emb:
                self.node_embeds = dgl.distributed.DistEmbedding(g.number_of_nodes(),
                                                                 self.embed_size,
                                                                 embed_name,
                                                                 init_emb)
            else:
                self.node_embeds = th.nn.Embedding(g.number_of_nodes(), self.embed_size, sparse=self.sparse_emb)
                nn.init.uniform_(self.node_embeds.weight, -1.0, 1.0)
        else:
            self.node_embeds = th.nn.Embedding(g.number_of_nodes(), self.embed_size)
            nn.init.uniform_(self.node_embeds.weight, -1.0, 1.0)

    def forward(self, node_ids, node_tids, features):
        """Forward computation
        Parameters
        ----------
        node_ids : tensor
            node ids to generate embedding for.
        node_ids : tensor
            node type ids
        features : list of features
            list of initial features for nodes belong to different node type.
            If None, the corresponding features is an one-hot encoding feature,
            else use the features directly as input feature and matmul a
            projection matrix.
        Returns
        -------
        tensor
            embeddings as the input of the next layer
        """
        embeds = th.empty(node_ids.shape[0], self.embed_size)
        for ntype in range(self.num_of_ntype):
            assert features[ntype] is None, 'Currently Dist RGCN only support non input feature'
            loc = node_tids == ntype
            embeds[loc] = self.node_embeds(node_ids[loc])

        return embeds

def compute_acc(results, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (results == labels).float().sum() / len(results)

def evaluate(g, model, embed_layer, labels, eval_loader, test_loader, node_feats, global_val_nid, global_test_nid):
    model.eval()
    embed_layer.eval()
    eval_logits = []
    eval_seeds = []

    global_results = dgl.distributed.DistTensor(labels.shape, th.long, 'results', persistent=True)

    with th.no_grad():
        for sample_data in tqdm.tqdm(eval_loader):
            seeds, blocks = sample_data
            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                blocks[0].srcdata[dgl.NTYPE],
                                node_feats)
            logits = model(blocks, feats)
            eval_logits.append(logits.cpu().detach())
            eval_seeds.append(seeds.cpu().detach())
    eval_logits = th.cat(eval_logits)
    eval_seeds = th.cat(eval_seeds)
    global_results[eval_seeds] = eval_logits.argmax(dim=1)

    test_logits = []
    test_seeds = []
    with th.no_grad():
        for sample_data in tqdm.tqdm(test_loader):
            seeds, blocks = sample_data
            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                blocks[0].srcdata[dgl.NTYPE],
                                node_feats)
            logits = model(blocks, feats)
            test_logits.append(logits.cpu().detach())
            test_seeds.append(seeds.cpu().detach())
    test_logits = th.cat(test_logits)
    test_seeds = th.cat(test_seeds)
    global_results[test_seeds] = test_logits.argmax(dim=1)

    g.barrier()
    if g.rank() == 0:
        return compute_acc(global_results[global_val_nid], labels[global_val_nid]), \
            compute_acc(global_results[global_test_nid], labels[global_test_nid])
    else:
        return -1, -1

class NeighborSampler:
    """Neighbor sampler
    Parameters
    ----------
    g : DGLHeterograph
        Full graph
    target_idx : tensor
        The target training node IDs in g
    fanouts : list of int
        Fanout of each hop starting from the seed nodes. If a fanout is None,
        sample full neighbors.
    """
    def __init__(self, g, fanouts, sample_neighbors):
        self.g = g
        self.fanouts = fanouts
        self.sample_neighbors = sample_neighbors

    def sample_blocks(self, seeds):
        """Do neighbor sample
        Parameters
        ----------
        seeds :
            Seed nodes
        Returns
        -------
        tensor
            Seed nodes, also known as target nodes
        blocks
            Sampled subgraphs
        """
        blocks = []
        etypes = []
        norms = []
        ntypes = []
        seeds = th.LongTensor(np.asarray(seeds))
        cur = seeds
        for fanout in self.fanouts:
            frontier = self.sample_neighbors(self.g, cur, fanout, replace=True)
            etypes = self.g.edata[dgl.ETYPE][frontier.edata[dgl.EID]]
            norm = self.g.edata['norm'][frontier.edata[dgl.EID]]
            block = dgl.to_block(frontier, cur)
            block.srcdata[dgl.NTYPE] = self.g.ndata[dgl.NTYPE][block.srcdata[dgl.NID]]
            block.edata['etype'] = etypes
            block.edata['norm'] = norm
            cur = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return seeds, blocks

def run(args, device, data):
    g, node_feats, num_of_ntype, num_classes, num_rels, \
        train_nid, val_nid, test_nid, labels, global_val_nid, global_test_nid = data

    fanouts = [int(fanout) for fanout in args.fanout.split(',')]
    val_fanouts = [int(fanout) for fanout in args.validation_fanout.split(',')]
    sampler = NeighborSampler(g, fanouts, dgl.distributed.sample_neighbors)
    # Create DataLoader for constructing blocks
    dataloader = DistDataLoader(
        dataset=train_nid.numpy(),
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False)

    valid_sampler = NeighborSampler(g, val_fanouts, dgl.distributed.sample_neighbors)
    # Create DataLoader for constructing blocks
    valid_dataloader = DistDataLoader(
        dataset=val_nid.numpy(),
        batch_size=args.batch_size,
        collate_fn=valid_sampler.sample_blocks,
        shuffle=False,
        drop_last=False)

    test_sampler = NeighborSampler(g, [-1] * args.n_layers, dgl.distributed.sample_neighbors)
    # Create DataLoader for constructing blocks
    test_dataloader = DistDataLoader(
        dataset=test_nid.numpy(),
        batch_size=args.batch_size,
        collate_fn=test_sampler.sample_blocks,
        shuffle=False,
        drop_last=False)

    embed_layer = DistEmbedLayer(device,
                                 g,
                                 num_of_ntype,
                                 args.n_hidden,
                                 sparse_emb=args.sparse_embedding,
                                 dgl_sparse_emb=args.dgl_sparse)

    model = EntityClassify(device,
                           args.n_hidden,
                           num_classes,
                           num_rels,
                           num_bases=args.n_bases,
                           num_hidden_layers=args.n_layers-2,
                           dropout=args.dropout,
                           use_self_loop=args.use_self_loop,
                           low_mem=args.low_mem,
                           layer_norm=args.layer_norm)
    model = model.to(device)
    if not args.standalone:
        model = th.nn.parallel.DistributedDataParallel(model)
        if args.sparse_embedding and not args.dgl_sparse:
            embed_layer = DistributedDataParallel(embed_layer, device_ids=None, output_device=None)

    if args.sparse_embedding:
        if args.dgl_sparse:
            emb_optimizer = dgl.distributed.SparseAdagrad([embed_layer.node_embeds], lr=args.sparse_lr)
        else:
            emb_optimizer = th.optim.SparseAdam(embed_layer.module.node_embeds.parameters(), lr=args.sparse_lr)
        optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)
    else:
        all_params = list(model.parameters()) + list(embed_layer.parameters())
        optimizer = th.optim.Adam(all_params, lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    for epoch in range(args.n_epochs):
        tic = time.time()

        sample_time = 0
        copy_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        number_train = 0

        step_time = []
        iter_t = []
        sample_t = []
        feat_copy_t = []
        forward_t = []
        backward_t = []
        update_t = []
        iter_tput = []

        start = time.time()
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        step_time = []
        for step, sample_data in enumerate(dataloader):
            seeds, blocks = sample_data
            number_train += seeds.shape[0]
            tic_step = time.time()
            sample_time += tic_step - start
            sample_t.append(tic_step - start)

            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                blocks[0].srcdata[dgl.NTYPE],
                                node_feats)
            label = labels[seeds]
            copy_time = time.time()
            feat_copy_t.append(copy_time - tic_step)

            # forward
            logits = model(blocks, feats)
            loss = F.cross_entropy(logits, label)
            forward_end = time.time()

            # backward
            optimizer.zero_grad()
            if args.sparse_embedding and not args.dgl_sparse:
                emb_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.sparse_embedding:
                emb_optimizer.step()
            compute_end = time.time()
            forward_t.append(forward_end - copy_time)
            backward_t.append(compute_end - forward_end)

            # Aggregate gradients in multiple nodes.
            optimizer.step()
            update_t.append(time.time() - compute_end)
            step_t = time.time() - start
            step_time.append(step_t)

            if step % args.log_every == 0:
                print('[{}] Epoch {:05d} | Step {:05d} | Loss {:.4f} | time {:.3f} s' \
                        '| sample {:.3f} | copy {:.3f} | forward {:.3f} | backward {:.3f} | update {:.3f}'.format(
                    g.rank(), epoch, step, loss.item(), np.sum(step_time[-args.log_every:]),
                    np.sum(sample_t[-args.log_every:]), np.sum(feat_copy_t[-args.log_every:]), np.sum(forward_t[-args.log_every:]),
                    np.sum(backward_t[-args.log_every:]), np.sum(update_t[-args.log_every:])))
            start = time.time()

        print('[{}]Epoch Time(s): {:.4f}, sample: {:.4f}, data copy: {:.4f}, forward: {:.4f}, backward: {:.4f}, update: {:.4f}, #number_train: {}'.format(
            g.rank(), np.sum(step_time), np.sum(sample_t), np.sum(feat_copy_t), np.sum(forward_t), np.sum(backward_t), np.sum(update_t), number_train))
        epoch += 1

        start = time.time()
        g.barrier()
        val_acc, test_acc = evaluate(g, model, embed_layer, labels,
            valid_dataloader, test_dataloader, node_feats, global_val_nid, global_test_nid)
        if val_acc >= 0:
            print('Val Acc {:.4f}, Test Acc {:.4f}, time: {:.4f}'.format(val_acc, test_acc,
                                                                         time.time() - start))

def main(args):
    dgl.distributed.initialize(args.ip_config, args.num_servers, num_workers=args.num_workers)
    if not args.standalone:
        th.distributed.init_process_group(backend='gloo')

    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.conf_path)
    print('rank:', g.rank())
    print('number of edges', g.number_of_edges())

    pb = g.get_partition_book()
    train_nid = dgl.distributed.node_split(g.ndata['train_mask'], pb, force_even=True)
    val_nid = dgl.distributed.node_split(g.ndata['val_mask'], pb, force_even=True)
    test_nid = dgl.distributed.node_split(g.ndata['test_mask'], pb, force_even=True)
    local_nid = pb.partid2nids(pb.partid).detach().numpy()
    print('part {}, train: {} (local: {}), val: {} (local: {}), test: {} (local: {})'.format(
          g.rank(), len(train_nid), len(np.intersect1d(train_nid.numpy(), local_nid)),
          len(val_nid), len(np.intersect1d(val_nid.numpy(), local_nid)),
          len(test_nid), len(np.intersect1d(test_nid.numpy(), local_nid))))
    device = th.device('cpu')
    labels = g.ndata['labels'][np.arange(g.number_of_nodes())]
    global_val_nid = th.LongTensor(np.nonzero(g.ndata['val_mask'][np.arange(g.number_of_nodes())])).squeeze()
    global_test_nid = th.LongTensor(np.nonzero(g.ndata['test_mask'][np.arange(g.number_of_nodes())])).squeeze()
    n_classes = len(th.unique(labels[labels >= 0]))
    print(labels.shape)
    print('#classes:', n_classes)

    # these two infor should have a better place to store and retrive
    num_of_ntype = len(th.unique(g.ndata[dgl.NTYPE][np.arange(g.number_of_nodes())]))
    num_rels = len(th.unique(g.edata[dgl.ETYPE][np.arange(g.number_of_edges())]))

    # no initial node features
    node_feats = [None] * num_of_ntype

    run(args, device, (g, node_feats, num_of_ntype, n_classes, num_rels,
                       train_nid, val_nid, test_nid, labels, global_val_nid, global_test_nid))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    # distributed training related
    parser.add_argument('--graph-name', type=str, help='graph name')
    parser.add_argument('--id', type=int, help='the partition id')
    parser.add_argument('--ip-config', type=str, help='The file for IP configuration')
    parser.add_argument('--conf-path', type=str, help='The path to the partition config file')
    parser.add_argument('--num-client', type=int, help='The number of clients')
    parser.add_argument('--num-servers', type=int, default=1, help='Server count on each machine.')

    # rgcn related
    parser.add_argument("--gpu", type=str, default='0',
            help="gpu")
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden units")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--sparse-lr", type=float, default=1e-2,
            help="sparse lr rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=50,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--relabel", default=False, action='store_true',
            help="remove untouched nodes and relabel")
    parser.add_argument("--fanout", type=str, default="4, 4",
            help="Fan-out of neighbor sampling.")
    parser.add_argument("--validation-fanout", type=str, default=None,
            help="Fan-out of neighbor sampling during validation.")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    parser.add_argument("--batch-size", type=int, default=100,
            help="Mini-batch size. ")
    parser.add_argument("--eval-batch-size", type=int, default=128,
            help="Mini-batch size. ")
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=1,
            help="Number of workers for distributed dataloader.")
    parser.add_argument("--low-mem", default=False, action='store_true',
            help="Whether use low mem RelGraphCov")
    parser.add_argument("--mix-cpu-gpu", default=False, action='store_true',
            help="Whether store node embeddins in cpu")
    parser.add_argument("--sparse-embedding", action='store_true',
            help='Use sparse embedding for node embeddings.')
    parser.add_argument("--dgl-sparse", action='store_true',
            help='Whether to use DGL sparse embedding')
    parser.add_argument('--node-feats', default=False, action='store_true',
            help='Whether use node features')
    parser.add_argument('--global-norm', default=False, action='store_true',
            help='User global norm instead of per node type norm')
    parser.add_argument('--layer-norm', default=False, action='store_true',
            help='Use layer norm')
    parser.add_argument('--local_rank', type=int, help='get rank of the process')
    parser.add_argument('--standalone', action='store_true', help='run in the standalone mode')
    args = parser.parse_args()

    # if validation_fanout is None, set it with args.fanout
    if args.validation_fanout is None:
        args.validation_fanout = args.fanout
    print(args)
    main(args)
