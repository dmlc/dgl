# /*!
#  *   Copyright (c) 2022, NVIDIA Corporation
#  *   All rights reserved.
#  *
#  *   Licensed under the Apache License, Version 2.0 (the "License");
#  *   you may not use this file except in compliance with the License.
#  *   You may obtain a copy of the License at
#  *
#  *       http://www.apache.org/licenses/LICENSE-2.0
#  *
#  *   Unless required by applicable law or agreed to in writing, software
#  *   distributed under the License is distributed on an "AS IS" BASIS,
#  *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  *   See the License for the specific language governing permissions and
#  *   limitations under the License.
#  *
#  * \file train_dist_layer.py
#  * \brief distributed cooperative minibatching example
#  */

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as thd
import torch.distributed.optim
import torchmetrics.functional as MF
from torch.multiprocessing import Process, Queue, Lock
import dgl
from dgl.base import NID
import dgl.nn as dglnn
from dgl.contrib.dist_sampling import DistConv, DistGraph, DistSampler, metis_partition, uniform_partition, reorder_graph_wrapper
from dgl.transforms.functional import add_self_loop, remove_self_loop
import time
import numpy as np
import tqdm
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from load_graph import load_reddit

import nvtx
    
class SAGE(nn.Module):
    def __init__(self, num_feats, replicated=False):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(num_feats) - 1):
            last = i == len(num_feats) - 2
            conv = dglnn.SAGEConv(num_feats[i], num_feats[i + 1], 'mean', feat_drop=0 if last else 0.5, activation=nn.Identity() if last else nn.ReLU())
            self.layers.append(DistConv(conv, i != 0 and not replicated))
        self.num_feats = num_feats
    
    def forward(self, blocks, h):
        # h is the dsttensor
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        return h

def producer(args, g, train_idx, device):
    fanouts = [int(_) for _ in args.fan_out.split(',')]

    sampler = DistSampler(g, dgl.dataloading.NeighborSampler, fanouts, ['features'], ['labels'])
    it = 0
    outputs = [None, None]
    for epoch in range(args.num_epochs):
        with nvtx.annotate("epoch: {}".format(epoch), color="orange"):
            if args.batch_size < train_idx.shape[0]:
                perm = th.randperm(train_idx.shape[0], device=device)
            elif epoch == 0:
                perm = th.arange(train_idx.shape[0], device=device)
            for i in range(0, train_idx.shape[0], args.batch_size):
                with nvtx.annotate("iteration: {}".format(it), color="yellow"):
                    seed_nodes = train_idx[perm[i: i + args.batch_size]].to(device)
                    input_nodes, output_nodes, blocks = sampler.sample(g, seed_nodes)
                    wait = blocks[0].slice_features(blocks[0])
                    blocks[-1].slice_labels(blocks[-1])
                    outputs[it % 2] = input_nodes, output_nodes, blocks, wait
                    it += 1
                    if it > 1:
                        input_nodes, output_nodes, blocks, wait = outputs[it % 2]
                        wait()
                        yield input_nodes, output_nodes, blocks
    it += 1
    input_nodes, output_nodes, blocks, wait = outputs[it % 2]
    wait()
    yield input_nodes, output_nodes, blocks

def train(local_rank, local_size, group_rank, world_size, g, parts, num_classes, args):
    th.set_num_threads(os.cpu_count() // local_size)
    th.cuda.set_device(local_rank)
    device = th.cuda.current_device()
    cpu_device = th.device('cpu')
    global_rank = group_rank * local_size + local_rank
    thd.init_process_group('nccl', 'env://', world_size=world_size, rank=global_rank)

    g = DistGraph(g, parts, args.replication)

    train_idx = th.nonzero(g.dstdata['train_mask'], as_tuple=True)[0] + g.l_offset
    val_idx = th.nonzero(g.dstdata['val_mask'], as_tuple=True)[0] + g.l_offset
    test_idx = th.nonzero(~(g.dstdata['train_mask'] | g.dstdata['val_mask']), as_tuple=True)[0] + g.l_offset
    reverse_eids = None if 'is_reverse' not in g.g.edata else th.nonzero(g.g.edata['is_reverse'], as_tuple=True)[0]
    
    num_layers = args.num_layers
    num_hidden = args.num_hidden

    model = SAGE([g.dstdata['features'].shape[1]] + [num_hidden for _ in range(num_layers - 1)] + [num_classes], args.replication == 1).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    opt = th.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    if not args.train:
        for epoch in range(args.num_epochs):
            for k in [3, 6, 9, 12, 15]:
                fanouts = [k for _ in range(num_layers)]
                samplers = [DistSampler(g, dgl.dataloading.NeighborSampler, fanouts)]
                if args.edge_pred:
                    samplers = [dgl.dataloading.as_edge_prediction_sampler(sampler, exclude='reverse_id', reverse_eids=reverse_eids,
                    negative_sampler=dgl.dataloading.negative_sampler.Uniform(1)) for sampler in samplers]
                sampler_names = ['NS']
                for batch_size in [1000, 2000, 4000, 8000, 16000, 32000, 64000]:
                    num_items = train_idx.shape[0] if not args.edge_pred else g.g.num_edges()
                    perm = th.randperm(num_items, device=device)
                    for i in range(0, num_items, batch_size):
                        seeds = train_idx[perm[i: i + batch_size]] if not args.edge_pred else perm[i: i + batch_size]
                        for sampler, name in zip(samplers, sampler_names):
                            if not args.edge_pred:
                                input_nodes, output_nodes, blocks = sampler.sample(g.g, seeds)
                                print("{}-{}-{}-{}".format(name, batch_size, k, global_rank), [(block.num_src_nodes(), block.num_dst_nodes(), block.num_edges()) for block in blocks])
                            else:
                                input_nodes, pair_graph, neg_graph, blocks = sampler.sample(g.g, seeds)
                                print("{}-{}-{}-{}".format(name, batch_size, k, global_rank), [(block.num_src_nodes(), block.num_dst_nodes(), block.num_edges()) for block in blocks])

    st, end = th.cuda.Event(enable_timing=True), th.cuda.Event(enable_timing=True)
    st.record()
    it = 0
    for input_nodes, output_nodes, blocks in producer(args, g, train_idx, device):
        x = blocks[0].srcdata.pop('features')
        y = blocks[-1].dstdata.pop('labels')
        with nvtx.annotate("forward", color="purple"):
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
        with nvtx.annotate("backward", color="purple"):
            opt.zero_grad()
            loss.backward()
        with nvtx.annotate("optimizer", color="purple"):
            opt.step()
        with nvtx.annotate("accuracy", color="purple"):
            acc = MF.accuracy(y_hat, y)
        end.record()
        mem = th.cuda.max_memory_allocated() >> 20
        block_stats = [(block.num_src_nodes(), block.num_dst_nodes(), block.num_edges()) for block in blocks]
        end.synchronize()
        print('rank: {}, it: {}, Loss: {:.4f}, Acc: {:.4f}, GPU Mem: {:.0f} MB, time: {:.3f}ms, stats: {}'.format(global_rank, it, loss.item(), acc.item(), mem, st.elapsed_time(end), block_stats))
        st, end = end, st
        it += 1

    thd.barrier()

def to_bidirected_with_reverse_mapping(g):
    """Makes a graph bidirectional, and returns a mapping array ``mapping`` where ``mapping[i]``
    is the reverse edge of edge ID ``i``.
    Does not work with graphs that have self-loops.
    """
    g_simple, mapping = dgl.to_simple(
        dgl.add_reverse_edges(g), return_counts='count', writeback_mapping=True)
    c = g_simple.edata['count']
    num_edges = g.num_edges()
    mapping_offset = torch.zeros(g_simple.num_edges() + 1, dtype=g_simple.idtype)
    mapping_offset[1:] = c.cumsum(0)
    idx = mapping.argsort()
    idx_uniq = idx[mapping_offset[:-1]]
    reverse_idx = torch.where(idx_uniq >= num_edges, idx_uniq - num_edges, idx_uniq + num_edges)
    reverse_mapping = mapping[reverse_idx]

    # Correctness check
    src1, dst1 = g_simple.edges()
    src2, dst2 = g_simple.find_edges(reverse_mapping)
    assert torch.equal(src1, dst2)
    assert torch.equal(src2, dst1)
    return g_simple, reverse_mapping

def main(args):
    # use all available CPUs
    th.set_num_threads(os.cpu_count())
    # use all available GPUs
    local_size = th.cuda.device_count()
    group_rank = int(os.environ["GROUP_RANK"])
    num_groups = int(os.environ["WORLD_SIZE"])
    world_size = local_size * num_groups
    if args.replication <= 0:
        args.replication = world_size

    g, n_classes = load_reddit()

    if args.undirected:
        g, reverse_eids = to_bidirected_with_reverse_mapping(remove_self_loop(g))
        g.edata['is_reverse'] = th.zeros(g.num_edges(), dtype=th.bool)
        g.edata['is_reverse'][reverse_eids] = True

    if args.partition == 'metis':
        parts = metis_partition(g, world_size)
    elif args.partition == 'random':
        parts = uniform_partition(g, world_size)
    else:
        parts = [th.arange(i * g.num_nodes() // world_size, (i + 1) * g.num_nodes() // world_size) for i in range(world_size)]
    g = reorder_graph_wrapper(g, parts)
    
    g.create_formats_()

    th.multiprocessing.spawn(train, args=(local_size, group_rank, world_size, g, [len(part) for part in parts], n_classes, args), nprocs=local_size)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--num-epochs', type=int, default=2)
    argparser.add_argument('--num-steps', type=int, default=5000)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='10,10,10')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts the graph, node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    argparser.add_argument('--edge-pred', action='store_true')
    argparser.add_argument('--logdir', type=str, default='tb_logs')
    argparser.add_argument('--partition', type=str, default='random')
    argparser.add_argument('--undirected', action='store_true')
    argparser.add_argument('--train', action='store_true')
    argparser.add_argument('--replication', type=int, default=0)
    args = argparser.parse_args()
    main(args)
